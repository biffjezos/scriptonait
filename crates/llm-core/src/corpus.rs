//! In-memory training corpus built from the user's sources.
//!
//! Source *content* (raw text, URL, timestamps, ...) is owned by the
//! frontend and persisted in IndexedDB — this module only needs the
//! prepared token stream for each source so it can assemble training
//! batches. Call [`Corpus::upsert`] whenever a source is added or edited in
//! the UI, and [`Corpus::remove`] when it's deleted; the flattened token
//! stream used for sampling is rebuilt lazily on the next batch request.
//!
//! Split by concern: this file keeps the struct itself, the tokenizer
//! lifecycle and source CRUD; [`store`] builds the flattened train/
//! held-out streams those live off; [`sampling`] draws batches from
//! them (both the random rotating-window strategy used for training and
//! the fixed deterministic one used for held-out/probe measurement);
//! [`stats`] answers "how much text, of what kind, sampled how much";
//! [`dedup`] finds sources whose text repeats.

mod dedup;
mod sampling;
mod stats;
mod store;

pub use sampling::{Batch, BatchDraw};
pub use stats::SourceStats;

use std::collections::{HashMap, VecDeque};

use crate::prep::{self, PreparedStats};
use crate::tokenizer::{self, Tokenizer};

use sampling::WindowCursor;

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
    /// reason `sample_counts` is. See [`sampling::WindowCursor`] and
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
    /// itself over a whole pretraining corpus. Feature-gated alongside
    /// `train::Trainer`: nothing outside that native trainer calls this.
    ///
    /// `tokens` is stored as given, so the caller owns the framing:
    /// `tokenizer::wrap_with_boundaries` for a plain document, or
    /// `instruct::Request::to_training_tokens` for an instruction
    /// example.
    #[cfg(feature = "native-trainer")]
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

    /// The cleaned (HTML-stripped if applicable, whitespace-normalized)
    /// text for a source, as used for training.
    pub fn cleaned_text(&self, id: &str) -> Option<&str> {
        self.cleaned_text.get(id).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
