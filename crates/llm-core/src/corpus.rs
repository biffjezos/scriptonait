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

pub struct Corpus {
    sources: HashMap<String, Vec<u32>>,
    cleaned_text: HashMap<String, String>,
    per_source_state: HashMap<String, StoryState>,
    retrieval: RetrievalIndex,
    order: Vec<String>,
    flat_cache: Vec<u32>,
    /// Index into `flat_cache` of each source's first token (its BOS).
    boundaries: Vec<usize>,
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
    /// Returns the resulting vocabulary size, or `None` when there is no
    /// text to learn from.
    pub fn learn_vocabulary(&mut self, target_vocab_size: usize) -> Option<usize> {
        let texts: Vec<&str> = self.order.iter().filter_map(|id| self.cleaned_text.get(id).map(|s| s.as_str())).collect();
        if texts.is_empty() {
            return None;
        }
        let tokenizer = Tokenizer::train(&texts, target_vocab_size);
        let size = tokenizer.vocab_size();
        self.set_tokenizer(tokenizer);
        Some(size)
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
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let removed = self.sources.remove(id).is_some();
        if removed {
            self.order.retain(|existing| existing != id);
            self.retrieval.remove_document(id);
            self.cleaned_text.remove(id);
            self.per_source_state.remove(id);
            self.dirty = true;
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
        for id in &self.order {
            if let Some(tokens) = self.sources.get(id) {
                self.boundaries.push(self.flat_cache.len());
                self.flat_cache.extend_from_slice(tokens);
            }
        }
        self.dirty = false;
    }

    /// Whether there's enough data to sample at least one training window.
    pub fn can_sample(&mut self, context_len: usize) -> bool {
        self.rebuild_flat_if_needed();
        self.flat_cache.len() > context_len
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
        let max_start = self.flat_cache.len() - context_len - 1;
        let boundary_starts: Vec<usize> = self.boundaries.iter().copied().filter(|&b| b <= max_start).collect();

        let mut inputs = Vec::with_capacity(batch_size * context_len);
        let mut targets = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let start = if !boundary_starts.is_empty() && rng.next_f32() < BOUNDARY_ALIGNED_SAMPLE_RATE {
                boundary_starts[rng.gen_range(boundary_starts.len())]
            } else {
                rng.gen_range(max_start + 1)
            };
            inputs.extend_from_slice(&self.flat_cache[start..start + context_len]);
            targets.extend_from_slice(&self.flat_cache[start + 1..start + 1 + context_len]);
        }
        Some(Batch { inputs, targets, batch_size, context_len })
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
