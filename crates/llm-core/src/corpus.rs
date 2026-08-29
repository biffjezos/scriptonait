//! In-memory training corpus built from the user's sources.
//!
//! Source *content* (raw text, URL, timestamps, ...) is owned by the
//! frontend and persisted in IndexedDB — this module only needs the
//! prepared token stream for each source so it can assemble training
//! batches. Call [`Corpus::upsert`] whenever a source is added or edited in
//! the UI, and [`Corpus::remove`] when it's deleted; the flattened token
//! stream used for sampling is rebuilt lazily on the next batch request.

use std::collections::{HashMap, VecDeque};

use crate::prep::{self, PreparedStats};
use crate::rng::Rng;
use crate::tokenizer::{self, Tokenizer};

/// Default fraction of sampled training windows that start exactly at a
/// source's beginning (its BOS token) instead of at the next window due in
/// its rotation. A control-tag preamble (see the frontend) only means
/// anything to the model if it's consistently seen at the *start* of a
/// window; never sampling one would place it there only by chance, roughly
/// `context_len` times less often than mid-window.
///
/// A training setting (see `set_boundary_sample_rate`) rather than only
/// ever this fixed default, because how much of a source's opening is
/// front matter — a title page, a table of contents, an epigraph — versus
/// actual prose varies by corpus, and a corpus with a lot of it is worth
/// turning this down for.
const DEFAULT_BOUNDARY_SAMPLE_RATE: f32 = 0.4;

/// Fraction of the corpus held out of training, to measure loss on text
/// the model has never been shown. Five percent is enough to be a
/// meaningful sample of a corpus this size while costing almost nothing
/// in training data.
const VALIDATION_FRACTION: f32 = 0.05;

pub struct Corpus {
    sources: HashMap<String, Vec<u32>>,
    cleaned_text: HashMap<String, String>,
    order: Vec<String>,
    flat_cache: Vec<u32>,
    /// Index into `flat_cache` of each source's first token (its BOS).
    boundaries: Vec<usize>,
    /// `(start, len)` of every source inside `flat_cache`, so a training
    /// window can be drawn from within one source rather than across the
    /// seam between two.
    spans: Vec<(usize, usize)>,
    /// The held-out stream: a slice taken from *every* source, not the
    /// tail of the corpus.
    ///
    /// A tail split holds out whichever source happens to be last, so
    /// its loss measures "text from a script we never saw" - a different
    /// distribution, not unseen text from the same one. Taking a slice
    /// out of each source instead means held-out loss answers the
    /// question it is supposed to: how does this model do on writing like
    /// the writing it was trained on.
    val_cache: Vec<u32>,
    /// `(start, len)` of every source's held-out slice inside
    /// `val_cache`, so a held-out window can be drawn from inside one
    /// source rather than across the seam between two — the same rule
    /// the training sampler has always had, and which the held-out
    /// sampler was missing.
    val_spans: Vec<(usize, usize)>,
    /// How many training windows have been drawn from each source, by id.
    ///
    /// Keyed by id rather than by `order` index: the index shifts whenever
    /// a source is added or removed, but a count belongs to the source, not
    /// to a position in the list. Read by the frontend to show which
    /// sources training has actually drawn from — useful for spotting a
    /// source that's barely been sampled (or one worth removing before it
    /// is).
    sample_counts: HashMap<String, u64>,
    /// How far each source has gotten through its own shuffled pass over
    /// its non-overlapping training windows, keyed by id for the same
    /// reason `sample_counts` is. See [`WindowCursor`] and
    /// [`Corpus::sample_batch`] for what this buys: every window in a
    /// source gets drawn once before any of them repeats, instead of
    /// sampling with replacement (which is what let a run replay the same
    /// handful of windows over and over, especially right after a
    /// reload — see `window_progress`/`set_window_progress`).
    window_cursors: HashMap<String, WindowCursor>,
    /// The order sources take a training window from, refilled with one
    /// entry per usable source (shuffled) whenever it runs out.
    ///
    /// This is what stops training from running through one document for
    /// many steps in a row: a full pass through this queue touches every
    /// usable source exactly once, in a random order, before any of them
    /// comes up again — rather than each step choosing a source at random
    /// (weighted by length) with nothing stopping the same long source
    /// from coming up several times in a row purely by chance.
    ///
    /// Not persisted: losing the exact position in this rotation across a
    /// reload only costs a little source-ordering variety for one
    /// session, which is nothing next to what `window_cursors` (which is
    /// persisted) actually protects against.
    rotation: VecDeque<String>,
    /// The most recently drawn source, so a fresh `rotation` shuffle can
    /// be nudged away from starting with the same one the last pass just
    /// ended on.
    last_source: Option<String>,
    /// Each window drawn by the most recent `sample_batch` call, in draw
    /// order — so the page can show what a step actually just trained
    /// on: not just which source, the text itself. Replaced wholesale at
    /// the start of every `sample_batch` call, not accumulated.
    last_batch_draws: Vec<BatchDraw>,
    /// See `DEFAULT_BOUNDARY_SAMPLE_RATE` and `set_boundary_sample_rate`.
    boundary_sample_rate: f32,
    /// Every distinct word the sources use, built once and thrown away
    /// whenever a source changes. Generated text is measured against it
    /// (see `eval::text_stats`), and rebuilding it per measurement would
    /// walk the whole corpus every twenty-five steps.
    word_vocab: Option<std::rc::Rc<std::collections::HashSet<String>>>,
    dirty: bool,
    /// The tokenizer every source is encoded with. Swapping it
    /// re-encodes everything (see `set_tokenizer`), because token ids
    /// from two different vocabularies cannot be mixed in one stream.
    tokenizer: Tokenizer,
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

/// How large a vocabulary a corpus of `chars` bytes can support.
///
/// Roughly one merge per 200 bytes of text, floored at 512 and with the
/// byte alphabet always included. The shape of the rule matters more than
/// the constant: a merge that appears twice in the whole corpus teaches
/// the model nothing and still costs a row of the embedding table, which
/// at these model sizes is real capacity taken from attention and the
/// MLP. Somebody pasting one scene gets a small vocabulary; somebody
/// loading a shelf of scripts gets the ceiling.
pub fn suggested_vocab_size(chars: usize) -> usize {
    (tokenizer::BASE_VOCAB_SIZE + chars / 200).max(512)
}

impl Corpus {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            cleaned_text: HashMap::new(),
            order: Vec::new(),
            flat_cache: Vec::new(),
            boundaries: Vec::new(),
            spans: Vec::new(),
            val_cache: Vec::new(),
            val_spans: Vec::new(),
            sample_counts: HashMap::new(),
            window_cursors: HashMap::new(),
            rotation: VecDeque::new(),
            last_source: None,
            last_batch_draws: Vec::new(),
            boundary_sample_rate: DEFAULT_BOUNDARY_SAMPLE_RATE,
            word_vocab: None,
            dirty: true,
            tokenizer: Tokenizer::byte_level(),
        }
    }

    /// A corpus that encodes its sources with a trained BPE vocabulary.
    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, ..Self::new() }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Replace the tokenizer and re-encode every source with it.
    ///
    /// Re-encoding is not optional: a token id only means anything
    /// relative to the vocabulary that produced it, so leaving old
    /// sources encoded with the old vocabulary would feed the model a
    /// stream where the same id means two different things.
    pub fn set_tokenizer(&mut self, tokenizer: Tokenizer) {
        self.tokenizer = tokenizer;
        let ids: Vec<String> = self.order.clone();
        for id in ids {
            if let Some(cleaned) = self.cleaned_text.get(&id).cloned() {
                let tokens = self.tokenizer.encode(&cleaned);
                self.sources.insert(id, tokenizer::wrap_with_boundaries(&tokens));
            }
        }
        self.dirty = true;
        self.word_vocab = None;
    }

    /// Learn a BPE vocabulary from the sources already loaded, and
    /// re-encode everything with it.
    ///
    /// Without this the tokenizer is byte-level: one token per byte, so a
    /// 900-word story costs ~4,000 tokens instead of ~900. That is four
    /// times the work per unit of text, in training and in generation
    /// alike, and four times less story inside the attention window. The
    /// merges have to be learned before a model exists, because the
    /// vocabulary size fixes the model's embedding table.
    ///
    /// `max_vocab_size` is a ceiling, not a target: the vocabulary
    /// actually learned scales with how much text there is
    /// (`suggested_vocab_size`), because a merge learned from two
    /// occurrences is noise, and every row costs `vocab * hidden`
    /// parameters whether it earns them or not.
    ///
    /// Returns the resulting vocabulary size, or `None` when there is no
    /// text to learn from.
    pub fn learn_vocabulary(&mut self, max_vocab_size: usize) -> Option<usize> {
        let texts: Vec<&str> = self.order.iter().filter_map(|id| self.cleaned_text.get(id).map(|s| s.as_str())).collect();
        if texts.is_empty() {
            return None;
        }
        let chars: usize = texts.iter().map(|t| t.len()).sum();
        let target = suggested_vocab_size(chars).min(max_vocab_size);
        let tokenizer = Tokenizer::train(&texts, target);
        let size = tokenizer.vocab_size();
        self.set_tokenizer(tokenizer);
        Some(size)
    }

    /// One flag per token id: does this corpus contain it?
    ///
    /// Generation uses this to stay inside the vocabulary the text
    /// actually used — see `SamplingConfig::allowed` for why an untrained
    /// token is worse than a merely unlikely one.
    pub fn seen_tokens(&self) -> std::rc::Rc<[bool]> {
        let mut seen = vec![false; self.tokenizer.vocab_size()];
        for tokens in self.sources.values() {
            for &id in tokens {
                if let Some(flag) = seen.get_mut(id as usize) {
                    *flag = true;
                }
            }
        }
        seen.into()
    }

    /// Clean, tokenize, and store (or replace) one source's text.
    pub fn upsert(&mut self, id: &str, raw_text: &str, is_html: bool) -> PreparedStats {
        let (cleaned, tokens, stats) = prep::prepare(&self.tokenizer, raw_text, is_html);
        let wrapped = tokenizer::wrap_with_boundaries(&tokens);
        if !self.sources.contains_key(id) {
            self.order.push(id.to_string());
            self.sample_counts.entry(id.to_string()).or_insert(0);
        }
        self.cleaned_text.insert(id.to_string(), cleaned);
        self.sources.insert(id.to_string(), wrapped);
        self.dirty = true;
        self.word_vocab = None;
        stats
    }

    /// Store one already-tokenized document.
    ///
    /// For the native trainer, which tokenizes a few tens of MB once
    /// ahead of time and then samples from it for hours. It skips the
    /// per-source bookkeeping `upsert` does for the browser's Sources
    /// panel, which would cost far more memory than the token stream
    /// itself over a whole pretraining corpus.
    ///
    /// `tokens` is stored as given, so the caller owns the framing:
    /// `tokenizer::wrap_with_boundaries` for a plain document, or
    /// `instruct::Request::to_training_tokens` for an instruction
    /// example.
    pub fn upsert_tokens(&mut self, id: &str, tokens: Vec<u32>) {
        if !self.sources.contains_key(id) {
            self.order.push(id.to_string());
        }
        self.sources.insert(id.to_string(), tokens);
        self.dirty = true;
        self.word_vocab = None;
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let removed = self.sources.remove(id).is_some();
        if removed {
            self.order.retain(|existing| existing != id);
            self.cleaned_text.remove(id);
            self.sample_counts.remove(id);
            self.window_cursors.remove(id);
            self.dirty = true;
            self.word_vocab = None;
        }
        removed
    }

    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Total tokens across all sources, including BOS/EOS boundary tokens.
    pub fn total_tokens(&self) -> usize {
        self.sources.values().map(|v| v.len()).sum()
    }

    pub fn source_token_count(&self, id: &str) -> Option<usize> {
        self.sources.get(id).map(|v| v.len())
    }

    /// The cleaned (HTML-stripped if applicable, whitespace-normalized)
    /// text for a source, as used for training.
    pub fn cleaned_text(&self, id: &str) -> Option<&str> {
        self.cleaned_text.get(id).map(String::as_str)
    }

    pub fn source_ids(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }

    fn rebuild_flat_if_needed(&mut self) {
        if !self.dirty {
            return;
        }
        self.flat_cache.clear();
        self.boundaries.clear();
        self.spans.clear();
        self.val_cache.clear();
        self.val_spans.clear();
        for id in &self.order {
            let Some(tokens) = self.sources.get(id) else { continue };
            // Each source contributes its own held-out slice, taken from
            // the end of that source, so every source is represented in
            // both streams and no training window can reach into one.
            let held = (tokens.len() as f32 * VALIDATION_FRACTION) as usize;
            let split = tokens.len().saturating_sub(held);
            // A source too small to split at all stays entirely in
            // training: half a scene is not a validation set. Still push a
            // (zero-length) val_span so `val_spans` stays index-aligned
            // with `order`/`spans` — per-source reporting zips the three
            // together and a skipped push here would misattribute every
            // source after this one.
            if held < 32 || split < 32 {
                self.boundaries.push(self.flat_cache.len());
                self.spans.push((self.flat_cache.len(), tokens.len()));
                self.flat_cache.extend_from_slice(tokens);
                self.val_spans.push((self.val_cache.len(), 0));
                continue;
            }
            self.boundaries.push(self.flat_cache.len());
            self.spans.push((self.flat_cache.len(), split));
            self.flat_cache.extend_from_slice(&tokens[..split]);
            self.val_spans.push((self.val_cache.len(), tokens.len() - split));
            self.val_cache.extend_from_slice(&tokens[split..]);
        }
        self.dirty = false;
    }

    /// Every distinct word the sources use, for measuring whether
    /// generated text is made of words this corpus contains.
    ///
    /// The user's own text is the right reference here, not a
    /// dictionary: a model trained on screenplays should be judged on
    /// whether it writes the words those screenplays use, and no word
    /// list has to ship with the page for that.
    pub fn word_vocabulary(&mut self) -> std::rc::Rc<std::collections::HashSet<String>> {
        if let Some(cached) = &self.word_vocab {
            return std::rc::Rc::clone(cached);
        }
        let built = std::rc::Rc::new(crate::eval::vocabulary(self.cleaned_text.values()));
        self.word_vocab = Some(std::rc::Rc::clone(&built));
        built
    }

    /// Characters of cleaned text across every source.
    ///
    /// The stable measure of "how much text is there". The token count
    /// is not: relearning the vocabulary re-encodes the same text into a
    /// different number of tokens, and a number that moves for reasons
    /// the user did not cause is a number they stop believing.
    pub fn total_chars(&self) -> usize {
        self.cleaned_text.values().map(|t| t.chars().count()).sum()
    }

    /// Bytes of cleaned text per token, over the whole corpus.
    ///
    /// This is what turns a loss per token into bits per byte, which is
    /// the only form of the number that can be compared between two
    /// vocabularies — or against gzip.
    pub fn bytes_per_token(&self) -> f32 {
        let bytes: usize = self.cleaned_text.values().map(|t| t.len()).sum();
        let tokens = self.total_tokens();
        if tokens == 0 { 0.0 } else { bytes as f32 / tokens as f32 }
    }

    /// How many tokens of each kind of writing the corpus holds.
    ///
    /// Counted in tokens rather than in sources, because one novel and
    /// twenty scenes are not a balanced corpus however the file count
    /// reads. Kinds with no text at all are left out.
    pub fn mix(&self) -> Vec<(crate::mix::SourceKind, usize)> {
        let mut totals: Vec<(crate::mix::SourceKind, usize)> = Vec::new();
        for id in &self.order {
            let (Some(text), Some(tokens)) = (self.cleaned_text.get(id), self.sources.get(id))
            else {
                continue;
            };
            let kind = crate::mix::classify(text);
            match totals.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, count)) => *count += tokens.len(),
                None => totals.push((kind, tokens.len())),
            }
        }
        totals.sort_by(|a, b| b.1.cmp(&a.1));
        totals
    }

    /// Per-source token counts and sample draws, for showing which
    /// sources training has actually drawn from.
    ///
    /// `order`, `spans` and `val_spans` are kept index-aligned by
    /// `rebuild_flat_if_needed` (a too-small-to-split source still gets a
    /// zero-length `val_spans` entry), so zipping the three by index is
    /// safe here.
    pub fn per_source_stats(&mut self) -> Vec<SourceStats> {
        self.rebuild_flat_if_needed();
        self.order
            .iter()
            .zip(self.spans.iter())
            .zip(self.val_spans.iter())
            .map(|((id, &(_, train_tokens)), &(_, held_out_tokens))| SourceStats {
                id: id.clone(),
                train_tokens,
                held_out_tokens,
                sampled: self.sample_counts.get(id).copied().unwrap_or(0),
            })
            .collect()
    }

    /// Restore a sample count after a fresh page load re-upserts sources
    /// into a new `Corpus` — the count belongs to the source, not to this
    /// in-memory instance, so it has to be handed back in from wherever it
    /// was persisted.
    pub fn set_sample_count(&mut self, id: &str, count: u64) {
        if self.sources.contains_key(id) {
            self.sample_counts.insert(id.to_string(), count);
        }
    }

    /// Tokens the training stream actually draws from, and the tokens
    /// held out of it. `total_tokens` counts both; a training plan has to
    /// separate them, because it is the first of the two that decides how
    /// long an epoch is.
    pub fn training_tokens(&mut self) -> usize {
        self.rebuild_flat_if_needed();
        self.flat_cache.len()
    }

    pub fn validation_tokens(&mut self) -> usize {
        self.rebuild_flat_if_needed();
        self.val_cache.len()
    }

    /// Whether there's enough data to sample at least one training window.
    pub fn can_sample(&mut self, context_len: usize) -> bool {
        self.rebuild_flat_if_needed();
        self.flat_cache.len() > context_len
    }

    /// Whether there is enough held-out text to measure anything.
    pub fn can_validate(&mut self, context_len: usize) -> bool {
        self.rebuild_flat_if_needed();
        self.val_cache.len() > context_len + 1
    }

    /// A batch drawn from the held-out stream — a slice of every source,
    /// never anything a training window can reach.
    /// A fixed set of windows drawn from one token stream, spread
    /// evenly across the sources in it and never crossing the seam
    /// between two. The same windows every call.
    ///
    /// Two things this has to get right, and the old held-out sampler
    /// got neither:
    ///
    /// **Fixed.** Fresh random windows every measurement means two
    /// consecutive numbers differ by which text was drawn, not by how
    /// the model changed, and at a few thousand tokens a measurement
    /// that term is the larger one. A fixed set removes it entirely.
    ///
    /// **Inside one source.** The stream is a concatenation, and a
    /// window spanning the seam asks the model to predict one script's
    /// title page from another script's last line. The training sampler
    /// has always refused to do that; the held-out sampler did it
    /// freely, which put a floor under held-out loss that had nothing to
    /// do with the model.
    fn fixed_windows(
        cache: &[u32],
        spans: &[(usize, usize)],
        count: usize,
        context_len: usize,
    ) -> Option<Batch> {
        if count == 0 {
            return None;
        }
        let usable: Vec<(usize, usize)> =
            spans.iter().copied().filter(|&(_, len)| len > context_len + 1).collect();
        if usable.is_empty() {
            return None;
        }
        // Total room for a window across every usable source. Walking a
        // position into this and then into the source that holds it
        // spreads the set in proportion to how much text each source
        // is, the same weighting the training sampler uses.
        let total: usize = usable.iter().map(|&(_, len)| len - context_len - 1).sum();
        let mut inputs = Vec::with_capacity(count * context_len);
        let mut targets = Vec::with_capacity(count * context_len);
        for i in 0..count {
            let mut offset = (i * total) / count;
            let mut start = usable[0].0;
            for &(base, len) in &usable {
                let room = len - context_len - 1;
                if offset < room {
                    start = base + offset;
                    break;
                }
                offset -= room;
            }
            if start + context_len + 1 > cache.len() {
                continue;
            }
            inputs.extend_from_slice(&cache[start..start + context_len]);
            targets.extend_from_slice(&cache[start + 1..start + 1 + context_len]);
        }
        let batch_size = inputs.len() / context_len.max(1);
        if batch_size == 0 {
            return None;
        }
        Some(Batch { inputs, targets, batch_size, context_len })
    }

    /// The held-out set: a fixed set of windows the model is never
    /// trained on.
    pub fn validation_batch(&mut self, count: usize, context_len: usize) -> Option<Batch> {
        self.rebuild_flat_if_needed();
        Self::fixed_windows(&self.val_cache, &self.val_spans, count, context_len)
    }

    /// The same thing drawn from the *training* text, and the reason it
    /// exists is the whole point of this pair.
    ///
    /// The training loss a run reports is measured on the batches it
    /// happens to be training on, and those are not a fair sample: some
    /// of them start at a source's opening (see `boundary_sample_rate`),
    /// which is the most predictable text in the corpus and which the
    /// model learns within a few hundred steps. Held-out windows never
    /// start at an opening, because openings live in the training
    /// portion of each source.
    ///
    /// So the two numbers measure different distributions, and their gap
    /// opens up the moment the model learns what an opening looks like —
    /// which reads exactly like overfitting, at a point in a run where
    /// the model has seen a tenth of its text and cannot possibly be
    /// memorizing it.
    ///
    /// This probe is drawn from training text the same way the held-out
    /// set is drawn from held-out text. The difference between the two
    /// is then a real answer to "does this model do worse on text it has
    /// not seen", because sampling is the one thing that is no longer
    /// different about them.
    pub fn training_probe_batch(&mut self, count: usize, context_len: usize) -> Option<Batch> {
        self.rebuild_flat_if_needed();
        Self::fixed_windows(&self.flat_cache, &self.spans, count, context_len)
    }

    pub fn sample_validation_batch(
        &mut self,
        batch_size: usize,
        context_len: usize,
        rng: &mut Rng,
    ) -> Option<Batch> {
        self.rebuild_flat_if_needed();
        if self.val_cache.len() <= context_len + 1 {
            return None;
        }
        let max_start = self.val_cache.len() - context_len - 1;
        let mut inputs = Vec::with_capacity(batch_size * context_len);
        let mut targets = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let start = rng.gen_range(max_start + 1);
            inputs.extend_from_slice(&self.val_cache[start..start + context_len]);
            targets.extend_from_slice(&self.val_cache[start + 1..start + 1 + context_len]);
        }
        Some(Batch { inputs, targets, batch_size, context_len })
    }

    /// Sample a batch of `(input, target)` windows for next-token
    /// prediction. `inputs`/`targets` are flattened row-major
    /// `[batch_size * context_len]` arrays; `targets[i]` is `inputs[i]`
    /// shifted one token to the right.
    ///
    /// Each window comes from a different source than the one before it
    /// (see `rotation`), and within one source every non-overlapping
    /// window is drawn exactly once before any of them repeat (see
    /// `window_cursors` and [`WindowCursor`]) — training never sits on
    /// one document for a run of consecutive steps, and never replays a
    /// window it has already shown the model while there are others in
    /// that source still untouched this pass. A fraction of draws
    /// (`boundary_sample_rate`) use a source's first window instead of
    /// its rotation's next one, so the model regularly sees openings —
    /// see `set_boundary_sample_rate` for turning it down on a corpus
    /// with a lot of front matter.
    pub fn sample_batch(
        &mut self,
        batch_size: usize,
        context_len: usize,
        rng: &mut Rng,
    ) -> Option<Batch> {
        self.rebuild_flat_if_needed();
        if context_len == 0 || self.flat_cache.len() <= context_len {
            return None;
        }
        // Every source long enough to hold at least one window, where it
        // starts in the flat stream, and how many non-overlapping windows
        // it holds. A window is drawn from inside one source: the flat
        // stream is a concatenation, and a window spanning the seam
        // teaches the model that one script's last line is followed by
        // another script's title page.
        let usable: HashMap<String, (usize, u32)> = self
            .order
            .iter()
            .zip(self.spans.iter())
            .filter(|&(_, &(_, len))| len > context_len)
            .map(|(id, &(base, len))| {
                let slots = (len - context_len - 1) / context_len + 1;
                (id.clone(), (base, slots as u32))
            })
            .collect();
        if usable.is_empty() {
            return None;
        }
        // Anything left over from before a source was removed or shrank
        // below one window would otherwise sit in the queue forever,
        // never drawn and never replaced.
        self.rotation.retain(|id| usable.contains_key(id));

        // How much of a window's own text to keep as an excerpt — enough
        // to actually read what it's training on, short enough that it's
        // cheap to decode and cheap to hand to the page every step.
        const EXCERPT_TOKENS: usize = 48;

        let mut inputs = Vec::with_capacity(batch_size * context_len);
        let mut targets = Vec::with_capacity(batch_size * context_len);
        self.last_batch_draws.clear();
        for _ in 0..batch_size {
            if self.rotation.is_empty() {
                self.refill_rotation(&usable, rng);
            }
            let Some(id) = self.rotation.pop_front() else { break };
            let &(base, slots) = usable.get(&id).expect("just filtered to usable");
            let boundary = rng.next_f32() < self.boundary_sample_rate;
            let slot = if boundary {
                0
            } else {
                self.window_cursors
                    .entry(id.clone())
                    .or_insert_with(|| WindowCursor::new(&id))
                    .next(slots)
                    .unwrap_or(0)
            };
            let start = base + slot as usize * context_len;
            if start + context_len + 1 > self.flat_cache.len() {
                // Should not happen — `slots` is derived from this same
                // source's own span — but a window that would read past
                // it is a window skipped, not a panic.
                self.last_source = Some(id);
                continue;
            }
            *self.sample_counts.entry(id.clone()).or_default() += 1;
            let excerpt_len = EXCERPT_TOKENS.min(context_len);
            let excerpt = self.tokenizer.decode(&self.flat_cache[start..start + excerpt_len]);
            self.last_batch_draws.push(BatchDraw { source_id: id.clone(), excerpt });
            inputs.extend_from_slice(&self.flat_cache[start..start + context_len]);
            targets.extend_from_slice(&self.flat_cache[start + 1..start + 1 + context_len]);
            self.last_source = Some(id);
        }
        let batch_size = inputs.len() / context_len;
        if batch_size == 0 {
            return None;
        }
        Some(Batch { inputs, targets, batch_size, context_len })
    }

    /// Refill `rotation` with one entry per usable source, shuffled.
    ///
    /// Nudged away from starting with whatever source the previous pass
    /// just ended on, so two sources are never adjacent across the seam
    /// between one pass through the rotation and the next.
    fn refill_rotation(&mut self, usable: &HashMap<String, (usize, u32)>, rng: &mut Rng) {
        let mut ids: Vec<String> = usable.keys().cloned().collect();
        ids.sort_unstable();
        for i in (1..ids.len()).rev() {
            let j = rng.gen_range(i + 1);
            ids.swap(i, j);
        }
        if ids.len() > 1 {
            if let Some(last) = &self.last_source {
                if ids[0] == *last {
                    ids.swap(0, 1);
                }
            }
        }
        self.rotation = ids.into();
    }

    /// Each window drawn by the most recent `sample_batch` call, in draw
    /// order — source id and a short text excerpt.
    pub fn last_batch_draws(&self) -> &[BatchDraw] {
        &self.last_batch_draws
    }

    /// This source's progress through its own shuffled pass over its
    /// non-overlapping training windows, as `(epoch, cursor)` — for
    /// persisting so a reload resumes that pass instead of restarting it
    /// (which is what let a training run replay the same handful of
    /// windows from the same source it happened to be on right before a
    /// reload). `None` if nothing has been drawn from this source yet.
    pub fn window_progress(&self, id: &str) -> Option<(u32, u32)> {
        self.window_cursors.get(id).map(|c| (c.epoch, c.cursor))
    }

    /// Every source's window-pass progress that exists yet, as
    /// `(id, epoch, cursor)` — the bulk form of `window_progress`, for
    /// writing it all back to storage in one pass rather than one round
    /// trip per source.
    pub fn all_window_progress(&self) -> Vec<(String, u32, u32)> {
        self.window_cursors.iter().map(|(id, c)| (id.clone(), c.epoch, c.cursor)).collect()
    }

    /// Restore a source's window-pass progress after a fresh page load
    /// re-upserts sources into a new `Corpus` — the progress belongs to
    /// the source, not to this in-memory instance, so it has to be handed
    /// back in from wherever it was persisted. A source that no longer
    /// exists is left alone rather than resurrecting a phantom entry.
    pub fn set_window_progress(&mut self, id: &str, epoch: u32, cursor: u32) {
        if self.sources.contains_key(id) {
            self.window_cursors.insert(id.to_string(), WindowCursor { epoch, cursor, ..WindowCursor::new(id) });
        }
    }

    /// How often a sampled window starts exactly at a source's beginning
    /// rather than at the next window due in its rotation.
    ///
    /// A training setting rather than a fixed rate because how much of a
    /// source's opening is actual prose, versus front matter (a title
    /// page, a table of contents, an epigraph), varies by corpus — a
    /// clean plain-text corpus can afford a higher rate than one built
    /// from scanned books.
    pub fn boundary_sample_rate(&self) -> f32 {
        self.boundary_sample_rate
    }

    pub fn set_boundary_sample_rate(&mut self, rate: f32) {
        self.boundary_sample_rate = rate.clamp(0.0, 1.0);
    }

    /// Source ids whose cleaned text is identical to an earlier source's.
    ///
    /// The same script added twice - a re-upload, the same file under two
    /// names - is trained on twice, which weights it double and inflates
    /// how well the model appears to do on it. Reported rather than
    /// removed: which copy to keep is the user's call.
    pub fn duplicate_sources(&self) -> Vec<String> {
        let mut seen: HashMap<u64, &str> = HashMap::new();
        let mut duplicates = Vec::new();
        for id in &self.order {
            let Some(text) = self.cleaned_text.get(id) else { continue };
            // FNV-1a over the cleaned text: cheap, and a collision here
            // would only mean one false report.
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in text.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            match seen.get(&hash) {
                Some(_) => duplicates.push(id.clone()),
                None => {
                    seen.insert(hash, id.as_str());
                }
            }
        }
        duplicates
    }
}

/// One source's progress through a shuffled pass ("epoch") over its own
/// non-overlapping training windows.
///
/// `order` — which window each position in the pass draws — is never
/// persisted, only `epoch` and `cursor` are (see
/// [`Corpus::window_progress`]/[`Corpus::set_window_progress`]): `order`
/// is rebuilt from the source's own id and `epoch` the moment it's
/// needed, deterministically, so there is nothing bigger than two small
/// integers to keep in a project file no matter how large the source is.
#[derive(Clone)]
struct WindowCursor {
    seed: u64,
    epoch: u32,
    cursor: u32,
    order: Vec<u32>,
}

impl WindowCursor {
    fn new(id: &str) -> Self {
        Self { seed: fnv1a(id.as_bytes()), epoch: 0, cursor: 0, order: Vec::new() }
    }

    /// The next window slot in this source's current pass — starting a
    /// fresh, freshly-shuffled pass when the current one runs out, or
    /// when `slots` no longer matches it (the source was edited out from
    /// under an in-progress pass, so its old indices no longer mean
    /// anything and this restarts the count rather than reading garbage).
    fn next(&mut self, slots: u32) -> Option<u32> {
        if slots == 0 {
            return None;
        }
        if self.order.len() as u32 != slots || self.cursor as usize >= self.order.len() {
            if self.cursor as usize >= self.order.len() && !self.order.is_empty() {
                self.epoch = self.epoch.wrapping_add(1);
            }
            self.order = shuffled_slots(self.seed ^ self.epoch as u64, slots);
            self.cursor = 0;
        }
        let slot = self.order[self.cursor as usize];
        self.cursor += 1;
        Some(slot)
    }
}

fn shuffled_slots(seed: u64, slots: u32) -> Vec<u32> {
    let mut order: Vec<u32> = (0..slots).collect();
    let mut rng = Rng::seed_from_u64(seed);
    for i in (1..order.len()).rev() {
        let j = rng.gen_range(i + 1);
        order.swap(i, j);
    }
    order
}

/// FNV-1a, for turning a source id into a stable per-source shuffle seed
/// without having to persist the seed itself.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub struct Batch {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
    pub batch_size: usize,
    pub context_len: usize,
}

/// One window `sample_batch` drew: which source it came from, and a
/// short decoded excerpt of its actual text. See
/// [`Corpus::last_batch_draws`].
pub struct BatchDraw {
    pub source_id: String,
    pub excerpt: String,
}

/// One source's share of the corpus, and how much of it training has
/// actually drawn from. See [`Corpus::per_source_stats`].
#[derive(Debug)]
pub struct SourceStats {
    pub id: String,
    pub train_tokens: usize,
    pub held_out_tokens: usize,
    pub sampled: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_cannot_sample() {
        let mut c = Corpus::new();
        assert!(!c.can_sample(8));
        let mut rng = Rng::seed_from_u64(0);
        assert!(c.sample_batch(2, 8, &mut rng).is_none());
    }

    #[test]
    fn upsert_and_remove_tracks_counts() {
        let mut c = Corpus::new();
        c.upsert("a", "hello world", false);
        assert_eq!(c.num_sources(), 1);
        assert!(c.total_tokens() > 0);
        c.upsert("b", "goodbye world", false);
        assert_eq!(c.num_sources(), 2);
        assert!(c.remove("a"));
        assert_eq!(c.num_sources(), 1);
        assert!(!c.remove("nonexistent"));
    }

    #[test]
    fn edit_replaces_not_duplicates() {
        let mut c = Corpus::new();
        c.upsert("a", "short", false);
        let first_len = c.total_tokens();
        c.upsert("a", "a much longer piece of text than before", false);
        assert_eq!(c.num_sources(), 1);
        assert!(c.total_tokens() > first_len);
    }

    #[test]
    fn sample_batch_has_expected_shape_and_shift() {
        let mut c = Corpus::new();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(20);
        c.upsert("a", &text, false);
        let mut rng = Rng::seed_from_u64(42);
        let batch = c.sample_batch(3, 16, &mut rng).expect("should sample");
        assert_eq!(batch.inputs.len(), 3 * 16);
        assert_eq!(batch.targets.len(), 3 * 16);
        // target[i] for a window is input shifted by one token: check the
        // first row's overlap explicitly.
        for t in 0..15 {
            assert_eq!(batch.targets[t], batch.inputs[t + 1]);
        }
    }

    #[test]
    fn removing_all_sources_makes_it_unsampleable_again() {
        let mut c = Corpus::new();
        c.upsert("a", "the quick brown fox jumps over the lazy dog", false);
        c.remove("a");
        assert!(!c.can_sample(4));
    }

    #[test]
    fn learning_a_vocabulary_shortens_the_token_stream() {
        let mut c = Corpus::new();
        let text = "INT. KITCHEN - DAY\n\nJANE\nWhere were you?\n\nJOHN\nOut.\n".repeat(40);
        c.upsert("a", &text, false);
        let before = c.total_tokens();
        let vocab = c.learn_vocabulary(600).expect("there is text to learn from");
        assert!(vocab > tokenizer::BASE_VOCAB_SIZE, "no merges were learned");
        let after = c.total_tokens();
        assert!(after < before, "expected fewer tokens after BPE, {before} -> {after}");
    }

    #[test]
    fn vocabulary_size_scales_with_the_corpus() {
        // A scene: a small vocabulary. A shelf of scripts: the ceiling.
        assert_eq!(suggested_vocab_size(0), 512);
        assert_eq!(suggested_vocab_size(2_000), 512);
        assert!(suggested_vocab_size(200_000) > 1_000);
        assert!(suggested_vocab_size(5_000_000) > 4_096);
    }

    #[test]
    fn every_source_contributes_to_the_held_out_stream() {
        // Two sources with disjoint vocabularies: if the held-out stream
        // came from the tail of the corpus it would contain only the
        // second one's tokens, and validation loss would be measuring a
        // different distribution rather than unseen text from the same
        // one.
        let mut c = Corpus::new();
        c.upsert("a", &"aaaa bbbb cccc dddd ".repeat(200), false);
        c.upsert("b", &"wwww xxxx yyyy zzzz ".repeat(200), false);
        assert!(c.can_validate(32));

        c.rebuild_flat_if_needed();
        let held: std::collections::HashSet<u32> = c.val_cache.iter().copied().collect();
        let from_a = held.contains(&(b'a' as u32));
        let from_b = held.contains(&(b'z' as u32));
        assert!(from_a && from_b, "both sources must appear in the held-out stream");

        // And no training window can reach the held-out tokens: the two
        // streams are separate buffers.
        let mut rng = Rng::seed_from_u64(4);
        for _ in 0..20 {
            assert!(c.sample_batch(2, 32, &mut rng).is_some());
            assert!(c.sample_validation_batch(2, 32, &mut rng).is_some());
        }
    }

    #[test]
    fn sample_counts_credit_the_right_source_and_sum_to_windows_drawn() {
        let mut c = Corpus::new();
        c.upsert("a", &"the quick brown fox jumps over the lazy dog ".repeat(20), false);
        c.upsert("b", &"a wizard's job is to vex chumps quickly in fog ".repeat(20), false);
        let mut rng = Rng::seed_from_u64(7);
        let mut drawn = 0;
        for _ in 0..25 {
            drawn += c.sample_batch(4, 16, &mut rng).expect("should sample").batch_size;
        }
        let stats = c.per_source_stats();
        assert_eq!(stats.len(), 2);
        let total_sampled: u64 = stats.iter().map(|s| s.sampled).sum();
        assert_eq!(total_sampled, drawn as u64);
        // With 100 windows drawn from two comparably-sized sources, both
        // should have been sampled at least once — a bug that always
        // credited the first source (e.g. an off-by-one in the order
        // index) would leave the other at zero.
        assert!(stats.iter().all(|s| s.sampled > 0), "{stats:?}");
    }

    #[test]
    fn set_sample_count_restores_a_persisted_value() {
        let mut c = Corpus::new();
        c.upsert("a", "hello world", false);
        c.set_sample_count("a", 4_200);
        assert_eq!(c.per_source_stats()[0].sampled, 4_200);
        // A count for a source that no longer exists is dropped rather
        // than resurrecting a phantom entry.
        c.set_sample_count("nonexistent", 99);
        assert!(c.per_source_stats().iter().all(|s| s.id != "nonexistent"));
    }

    #[test]
    fn per_source_stats_stay_aligned_when_a_source_is_too_small_to_hold_out() {
        // "a" is long enough to split into train/held-out; "b" is a few
        // words, well under the 32-token floor, so it stays entirely in
        // training. Before the val_spans fix this shifted every stat
        // after "b" onto the wrong source.
        let mut c = Corpus::new();
        c.upsert("a", &"the cave and the fire and the shadows on the wall. ".repeat(200), false);
        c.upsert("b", "too short to hold out", false);
        c.upsert("c", &"the cave and the fire and the shadows on the wall. ".repeat(200), false);
        let stats = c.per_source_stats();
        let by_id = |id: &str| stats.iter().find(|s| s.id == id).unwrap();
        assert_eq!(by_id("b").held_out_tokens, 0);
        assert!(by_id("a").held_out_tokens > 0);
        assert!(by_id("c").held_out_tokens > 0);
    }

    #[test]
    fn the_validation_set_is_the_same_every_time() {
        let mut c = Corpus::new();
        c.upsert("a", &"the cave and the fire and the shadows on the wall. ".repeat(400), false);
        let first = c.validation_batch(4, 32).expect("enough held-out text");
        for _ in 0..5 {
            let again = c.validation_batch(4, 32).expect("enough held-out text");
            assert_eq!(again.inputs, first.inputs, "the validation set must not move");
            assert_eq!(again.targets, first.targets);
        }
    }

    #[test]
    fn the_validation_set_spreads_across_the_held_out_stream() {
        let mut c = Corpus::new();
        for i in 0..6 {
            c.upsert(&format!("s{i}"), &format!("source {i} text. ").repeat(500), false);
        }
        let batch = c.validation_batch(4, 32).expect("enough held-out text");
        // Four windows drawn from four different places: if they were
        // the first four in a row they would share most of their tokens.
        let windows: Vec<&[u32]> = batch.inputs.chunks(32).collect();
        assert_eq!(windows.len(), 4);
        assert!(
            windows[0] != windows[3],
            "windows from opposite ends of the stream should differ"
        );
    }

    #[test]
    fn no_validation_set_without_enough_held_out_text() {
        let mut c = Corpus::new();
        c.upsert("a", "too short", false);
        assert!(c.validation_batch(4, 32).is_none());
    }

    #[test]
    fn a_held_out_window_never_spans_two_sources_either() {
        let mut c = Corpus::new();
        // Six sources with disjoint vocabularies, so a window that
        // crossed a seam would hold tokens from two of them.
        for letter in ["a", "b", "c", "d", "e", "f"] {
            c.upsert(letter, &format!("{letter} ").repeat(4000), false);
        }
        let batch = c.validation_batch(6, 16).expect("enough held-out text");
        for window in batch.inputs.chunks(16) {
            // Each source is one letter and a space, so a window inside
            // one has at most three distinct ids counting the boundary
            // tokens. One that crossed a seam would carry two letters.
            let mut letters: Vec<u32> = window
                .iter()
                .copied()
                .filter(|&id| (b'a' as u32..=b'f' as u32).contains(&id))
                .collect();
            letters.sort_unstable();
            letters.dedup();
            assert!(
                letters.len() <= 1,
                "a held-out window crossed a source seam: {letters:?}"
            );
        }
    }

    /// The probe and the held-out set are drawn the same way, from the
    /// two halves of the same split. That is what makes their difference
    /// mean something.
    #[test]
    fn the_training_probe_matches_the_held_out_set_in_shape() {
        let mut c = Corpus::new();
        for i in 0..4 {
            c.upsert(&format!("s{i}"), &format!("source {i} text here. ").repeat(600), false);
        }
        let probe = c.training_probe_batch(8, 32).expect("training text");
        let held = c.validation_batch(8, 32).expect("held-out text");
        assert_eq!(probe.batch_size, held.batch_size);
        assert_eq!(probe.context_len, held.context_len);
        // And both are fixed.
        assert_eq!(c.training_probe_batch(8, 32).unwrap().inputs, probe.inputs);
        assert_eq!(c.validation_batch(8, 32).unwrap().inputs, held.inputs);
        // Drawn from different halves, so they are not the same text.
        assert_ne!(probe.inputs, held.inputs);
    }

    #[test]
    fn a_training_window_never_spans_two_sources() {
        // Two sources with disjoint alphabets: a window containing both
        // would be a window that crossed the seam between scripts.
        let mut c = Corpus::new();
        c.upsert("a", &"aaaa ".repeat(500), false);
        c.upsert("b", &"zzzz ".repeat(500), false);
        let mut rng = Rng::seed_from_u64(9);
        for _ in 0..100 {
            let batch = c.sample_batch(4, 32, &mut rng).expect("batch");
            for window in batch.inputs.chunks(32) {
                let has_a = window.contains(&(b'a' as u32));
                let has_z = window.contains(&(b'z' as u32));
                assert!(!(has_a && has_z), "window spans two sources: {window:?}");
            }
        }
    }

    #[test]
    fn duplicate_sources_are_reported_not_removed() {
        let mut c = Corpus::new();
        c.upsert("a", "INT. KITCHEN - DAY\n\nJANE\nHi.", false);
        c.upsert("copy", "INT. KITCHEN - DAY\n\nJANE\nHi.", false);
        c.upsert("other", "EXT. GARDEN - NIGHT\n\nJOHN\nBye.", false);
        assert_eq!(c.duplicate_sources(), vec!["copy".to_string()]);
        assert_eq!(c.num_sources(), 3, "reporting a duplicate must not remove it");
    }

    #[test]
    fn a_tiny_corpus_has_no_validation_split() {
        let mut c = Corpus::new();
        c.upsert("a", "the quick brown fox jumps over the lazy dog", false);
        assert!(!c.can_validate(32), "too little text to hold any out");
        // ...and training still works.
        let mut rng = Rng::seed_from_u64(1);
        assert!(c.sample_batch(1, 4, &mut rng).is_some());
    }

    #[test]
    fn seen_tokens_marks_only_what_the_corpus_contains() {
        let mut c = Corpus::new();
        c.upsert("a", "abc abc abc", false);
        let seen = c.seen_tokens();
        assert!(seen[b'a' as usize], "'a' is in the text");
        assert!(!seen[b'Z' as usize], "'Z' is not");
        assert!(!seen[200], "no high byte is");
    }

    #[test]
    fn learning_a_vocabulary_needs_text() {
        assert_eq!(Corpus::new().learn_vocabulary(600), None);
    }

    #[test]
    fn cleaned_text_is_available_and_cleared_on_remove() {
        let mut c = Corpus::new();
        // Leading indentation is deliberately preserved (screenplay
        // formatting relies on it - see prep.rs); internal whitespace runs
        // and trailing whitespace still get cleaned up.
        c.upsert("a", "  Hello   world  ", false);
        assert_eq!(c.cleaned_text("a"), Some("  Hello world"));
        c.remove("a");
        assert_eq!(c.cleaned_text("a"), None);
    }

    #[test]
    fn sampling_sometimes_lands_exactly_on_a_source_boundary() {
        let mut c = Corpus::new();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(20);
        c.upsert("a", &text, false);
        c.upsert("b", &text, false);
        let mut rng = Rng::seed_from_u64(7);
        let mut boundary_hits = 0;
        let trials = 500;
        for _ in 0..trials {
            let batch = c.sample_batch(1, 16, &mut rng).unwrap();
            if batch.inputs[0] == tokenizer::BOS {
                boundary_hits += 1;
            }
        }
        // Should land on a boundary roughly DEFAULT_BOUNDARY_SAMPLE_RATE of
        // the time (40%) - well above what a uniformly random start over a
        // corpus this size would produce by chance (a fraction of a
        // percent), and comfortably below 100%.
        let rate = boundary_hits as f64 / trials as f64;
        assert!(rate > 0.2 && rate < 0.6, "boundary-aligned rate was {rate}");
    }

    #[test]
    fn boundary_sample_rate_is_a_training_setting() {
        let mut c = Corpus::new();
        let text = "the quick brown fox jumps over the lazy dog ".repeat(20);
        c.upsert("a", &text, false);
        c.set_boundary_sample_rate(1.0);
        let mut rng = Rng::seed_from_u64(3);
        for _ in 0..10 {
            let batch = c.sample_batch(1, 16, &mut rng).unwrap();
            assert_eq!(batch.inputs[0], tokenizer::BOS, "rate of 1.0 should always land on a boundary");
        }
        c.set_boundary_sample_rate(0.0);
        let mut boundary_hits = 0;
        for _ in 0..200 {
            let batch = c.sample_batch(1, 16, &mut rng).unwrap();
            if batch.inputs[0] == tokenizer::BOS {
                boundary_hits += 1;
            }
        }
        assert!(boundary_hits < 20, "rate of 0.0 should almost never land on a boundary, got {boundary_hits}/200");
        // Out-of-range values are clamped, not rejected.
        c.set_boundary_sample_rate(5.0);
        assert_eq!(c.boundary_sample_rate(), 1.0);
        c.set_boundary_sample_rate(-1.0);
        assert_eq!(c.boundary_sample_rate(), 0.0);
    }

    #[test]
    fn sample_batch_covers_every_window_once_before_repeating_any() {
        // 94 distinct printable characters, none repeated anywhere in the
        // source - so any two windows starting at different offsets are
        // guaranteed to differ, and a duplicate in `starts` below can only
        // mean the same window was drawn twice.
        let text: String = (0x21u8..=0x7E_u8).map(|b| b as char).collect();
        let mut c = Corpus::new();
        c.upsert("a", &text, false);
        c.set_boundary_sample_rate(0.0);
        let mut rng = Rng::seed_from_u64(11);
        let mut starts = std::collections::HashSet::new();
        // context_len 8 over this source holds 11 non-overlapping
        // windows; drawing 10 stays inside that one pass; a repeat among
        // them means a window was drawn twice before all 11 were seen.
        for _ in 0..10 {
            let batch = c.sample_batch(1, 8, &mut rng).unwrap();
            starts.insert(batch.inputs.clone());
        }
        assert_eq!(starts.len(), 10, "one pass should not repeat a window before covering the others");
    }

    #[test]
    fn consecutive_draws_favor_different_sources() {
        let mut c = Corpus::new();
        for letter in ["a", "b", "c", "d"] {
            c.upsert(letter, &format!("{letter} ").repeat(4000), false);
        }
        c.set_boundary_sample_rate(0.0);
        let mut rng = Rng::seed_from_u64(13);
        let mut last: Option<String> = None;
        let mut adjacent_repeats = 0;
        for _ in 0..200 {
            let before = c.per_source_stats();
            c.sample_batch(1, 16, &mut rng).unwrap();
            let after = c.per_source_stats();
            // The one source whose sample count just went up is the one
            // this draw came from — unambiguous, unlike inspecting the
            // window's own bytes (two different sources can share a byte
            // value at the offset a window happens to start on).
            let drawn = after
                .iter()
                .find(|a| {
                    let prior = before.iter().find(|b| b.id == a.id).map(|b| b.sampled).unwrap_or(0);
                    a.sampled > prior
                })
                .map(|s| s.id.clone())
                .expect("exactly one source's count should have moved");
            if last.as_deref() == Some(drawn.as_str()) {
                adjacent_repeats += 1;
            }
            last = Some(drawn);
        }
        assert_eq!(adjacent_repeats, 0, "the same source was drawn twice in a row");
    }

    #[test]
    fn window_progress_survives_being_read_back_in() {
        let mut c = Corpus::new();
        c.upsert("a", &"a".repeat(1000), false);
        c.set_boundary_sample_rate(0.0);
        let mut rng = Rng::seed_from_u64(5);
        c.sample_batch(1, 16, &mut rng).unwrap();
        c.sample_batch(1, 16, &mut rng).unwrap();
        let (epoch, cursor) = c.window_progress("a").expect("drawn from");
        assert_eq!(cursor, 2);

        // A fresh corpus, as a reload would build, with the progress
        // handed back in - the next draw should be the third window, not
        // a repeat of the first.
        let mut fresh = Corpus::new();
        fresh.upsert("a", &"a".repeat(1000), false);
        fresh.set_boundary_sample_rate(0.0);
        fresh.set_window_progress("a", epoch, cursor);
        let mut fresh_rng = Rng::seed_from_u64(999); // a different stream entirely
        let mut original = Corpus::new();
        original.upsert("a", &"a".repeat(1000), false);
        original.set_boundary_sample_rate(0.0);
        let mut original_rng = Rng::seed_from_u64(5);
        original.sample_batch(1, 16, &mut original_rng).unwrap();
        original.sample_batch(1, 16, &mut original_rng).unwrap();
        let expected_third = original.sample_batch(1, 16, &mut original_rng).unwrap();

        let restored_third = fresh.sample_batch(1, 16, &mut fresh_rng).unwrap();
        assert_eq!(
            restored_third.inputs, expected_third.inputs,
            "restoring progress should resume the same pass, not restart it"
        );
    }

    #[test]
    fn set_window_progress_ignores_a_source_that_no_longer_exists() {
        let mut c = Corpus::new();
        c.upsert("a", "hello world", false);
        c.set_window_progress("nonexistent", 1, 5);
        assert!(c.window_progress("nonexistent").is_none());
    }

    #[test]
    fn last_batch_draws_names_where_each_window_of_the_latest_batch_came_from() {
        let mut c = Corpus::new();
        for letter in ["a", "b", "c"] {
            c.upsert(letter, &format!("{letter} ").repeat(4000), false);
        }
        let mut rng = Rng::seed_from_u64(3);
        let batch = c.sample_batch(3, 16, &mut rng).expect("should sample");
        assert_eq!(c.last_batch_draws().len(), batch.batch_size);
        assert!(c.last_batch_draws().iter().all(|d| !d.excerpt.is_empty()));
        // Replaced wholesale, not accumulated across calls.
        let second = c.sample_batch(2, 16, &mut rng).expect("should sample");
        assert_eq!(c.last_batch_draws().len(), second.batch_size);
    }
}
