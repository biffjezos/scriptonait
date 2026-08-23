//! In-memory training corpus built from the user's sources.
//!
//! Source *content* (raw text, URL, timestamps, ...) is owned by the
//! frontend and persisted in IndexedDB — this module only needs the
//! prepared token stream for each source so it can assemble training
//! batches. Call [`Corpus::upsert`] whenever a source is added or edited in
//! the UI, and [`Corpus::remove`] when it's deleted; the flattened token
//! stream used for sampling is rebuilt lazily on the next batch request.

use std::collections::HashMap;

use crate::prep::{self, PreparedStats};
use crate::retrieval::{RetrievalIndex, RetrievedChunk};
use crate::rng::Rng;
use crate::screenplay::{self, StoryState};
use crate::tokenizer::{self, Tokenizer};

/// Fraction of sampled training windows that start exactly at a source's
/// beginning (its BOS token) instead of a uniformly random offset. A
/// control-tag preamble (see the frontend) or the story-state-derived
/// framing only means anything to the model if it's consistently seen at
/// the *start* of a window; pure uniform-random sampling would place it
/// there only by chance, roughly `context_len` times less often than
/// mid-window. The rest of the batch still samples uniformly so the model
/// keeps seeing ordinary mid-document continuations too.
const BOUNDARY_ALIGNED_SAMPLE_RATE: f32 = 0.4;

/// Fraction of the corpus held out of training, to measure loss on text
/// the model has never been shown. Five percent is enough to be a
/// meaningful sample of a corpus this size while costing almost nothing
/// in training data.
const VALIDATION_FRACTION: f32 = 0.05;

pub struct Corpus {
    sources: HashMap<String, Vec<u32>>,
    cleaned_text: HashMap<String, String>,
    per_source_state: HashMap<String, StoryState>,
    retrieval: RetrievalIndex,
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
            per_source_state: HashMap::new(),
            retrieval: RetrievalIndex::new(),
            order: Vec::new(),
            flat_cache: Vec::new(),
            boundaries: Vec::new(),
            spans: Vec::new(),
            val_cache: Vec::new(),
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
        }
        self.per_source_state.insert(id.to_string(), screenplay::extract_story_state(&cleaned));
        self.retrieval.upsert_document(id, &cleaned);
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
    /// retrieval index and the story-state scan that `upsert` builds —
    /// those exist for the browser's Sources panel, and building them
    /// over a whole pretraining corpus would cost far more memory than
    /// the token stream itself.
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
            self.retrieval.remove_document(id);
            self.cleaned_text.remove(id);
            self.per_source_state.remove(id);
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
    /// text for a source, as used for both training and heuristic
    /// screenplay-structure extraction.
    pub fn cleaned_text(&self, id: &str) -> Option<&str> {
        self.cleaned_text.get(id).map(String::as_str)
    }

    pub fn source_ids(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }

    /// Up to `k` scenes from the corpus most similar to `query` (TF-IDF
    /// cosine similarity over `screenplay::split_into_scenes` chunks) —
    /// useful as few-shot context prepended to a generation prompt.
    pub fn retrieve(&self, query: &str, k: usize) -> Vec<RetrievedChunk> {
        self.retrieval.top_k(query, k)
    }

    /// Characters/locations/scene-count tracked across every current
    /// source, in first-seen (insertion) order. See `screenplay.rs` for
    /// how this heuristic extraction works and its known limitations.
    pub fn story_state(&self) -> StoryState {
        let mut merged = StoryState::default();
        for id in &self.order {
            if let Some(state) = self.per_source_state.get(id) {
                merged.merge(state);
            }
        }
        merged
    }

    fn rebuild_flat_if_needed(&mut self) {
        if !self.dirty {
            return;
        }
        self.flat_cache.clear();
        self.boundaries.clear();
        self.spans.clear();
        self.val_cache.clear();
        for id in &self.order {
            let Some(tokens) = self.sources.get(id) else { continue };
            // Each source contributes its own held-out slice, taken from
            // the end of that source, so every source is represented in
            // both streams and no training window can reach into one.
            let held = (tokens.len() as f32 * VALIDATION_FRACTION) as usize;
            let split = tokens.len().saturating_sub(held);
            // A source too small to split at all stays entirely in
            // training: half a scene is not a validation set.
            if held < 32 || split < 32 {
                self.boundaries.push(self.flat_cache.len());
                self.spans.push((self.flat_cache.len(), tokens.len()));
                self.flat_cache.extend_from_slice(tokens);
                continue;
            }
            self.boundaries.push(self.flat_cache.len());
            self.spans.push((self.flat_cache.len(), split));
            self.flat_cache.extend_from_slice(&tokens[..split]);
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
    /// shifted one token to the right. A fraction of windows
    /// (`BOUNDARY_ALIGNED_SAMPLE_RATE`) start exactly at a source
    /// boundary rather than a uniformly random offset — see that
    /// constant's doc comment for why.
    pub fn sample_batch(
        &mut self,
        batch_size: usize,
        context_len: usize,
        rng: &mut Rng,
    ) -> Option<Batch> {
        self.rebuild_flat_if_needed();
        if self.flat_cache.len() <= context_len {
            return None;
        }
        // Sources long enough to hold a window, and where each one starts
        // in the flat stream. A window is drawn from inside one source:
        // the flat stream is a concatenation, and a window spanning the
        // seam teaches the model that one script's last line is followed
        // by another script's title page.
        let usable: Vec<(usize, usize)> = self
            .spans
            .iter()
            .copied()
            .filter(|&(_, len)| len > context_len + 1)
            .collect();
        if usable.is_empty() {
            return None;
        }
        // Weighted by length, so a corpus of one long script and one
        // short one samples in proportion to how much text each is.
        let total: usize = usable.iter().map(|&(_, len)| len - context_len - 1).sum();

        let mut inputs = Vec::with_capacity(batch_size * context_len);
        let mut targets = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let start = if rng.next_f32() < BOUNDARY_ALIGNED_SAMPLE_RATE {
                // A window that starts where a source starts, so the model
                // sees what an opening looks like - see the constant.
                usable[rng.gen_range(usable.len())].0
            } else {
                let mut pick = rng.gen_range(total.max(1));
                let mut chosen = usable[0].0;
                for &(base, len) in &usable {
                    let room = len - context_len - 1;
                    if pick < room {
                        chosen = base + pick;
                        break;
                    }
                    pick -= room;
                }
                chosen
            };
            inputs.extend_from_slice(&self.flat_cache[start..start + context_len]);
            targets.extend_from_slice(&self.flat_cache[start + 1..start + 1 + context_len]);
        }
        Some(Batch { inputs, targets, batch_size, context_len })
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

pub struct Batch {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
    pub batch_size: usize,
    pub context_len: usize,
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
    fn story_state_aggregates_across_sources_and_updates_on_remove() {
        let mut c = Corpus::new();
        c.upsert("a", "INT. KITCHEN - DAY\n\nJANE\nHi.\n\nJANE\nStill here.", false);
        c.upsert("b", "EXT. GARDEN - NIGHT\n\nJOHN\nBye.\n\nJOHN\nReally.", false);
        let state = c.story_state();
        assert_eq!(state.characters, vec!["JANE".to_string(), "JOHN".to_string()]);
        assert_eq!(state.scene_count, 2);

        c.remove("a");
        let state = c.story_state();
        assert_eq!(state.characters, vec!["JOHN".to_string()]);
        assert_eq!(state.scene_count, 1);
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
    fn retrieve_finds_similar_scenes_and_forgets_removed_sources() {
        let mut c = Corpus::new();
        c.upsert(
            "spy",
            "INT. SURVEILLANCE VAN - NIGHT\n\nAgents trace a wiretap through satellite relays.",
            false,
        );
        c.upsert("romance", "INT. RESTAURANT - EVENING\n\nTwo friends share a quiet dinner.", false);

        let results = c.retrieve("wiretap surveillance relays", 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].source_id, "spy");

        c.remove("spy");
        assert!(c.retrieve("wiretap surveillance relays", 2).is_empty());
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
        // Should land on a boundary roughly BOUNDARY_ALIGNED_SAMPLE_RATE of
        // the time (40%) - well above what a uniformly random start over a
        // corpus this size would produce by chance (a fraction of a
        // percent), and comfortably below 100%.
        let rate = boundary_hits as f64 / trials as f64;
        assert!(rate > 0.2 && rate < 0.6, "boundary-aligned rate was {rate}");
    }
}
