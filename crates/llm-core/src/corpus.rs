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
use crate::rng::Rng;
use crate::tokenizer;

pub struct Corpus {
    sources: HashMap<String, Vec<u32>>,
    order: Vec<String>,
    flat_cache: Vec<u32>,
    dirty: bool,
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
            order: Vec::new(),
            flat_cache: Vec::new(),
            dirty: true,
        }
    }

    /// Clean, tokenize, and store (or replace) one source's text.
    pub fn upsert(&mut self, id: &str, raw_text: &str, is_html: bool) -> PreparedStats {
        let (_, tokens, stats) = prep::prepare(raw_text, is_html);
        let wrapped = tokenizer::wrap_with_boundaries(&tokens);
        if !self.sources.contains_key(id) {
            self.order.push(id.to_string());
        }
        self.sources.insert(id.to_string(), wrapped);
        self.dirty = true;
        stats
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let removed = self.sources.remove(id).is_some();
        if removed {
            self.order.retain(|existing| existing != id);
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

    fn rebuild_flat_if_needed(&mut self) {
        if !self.dirty {
            return;
        }
        self.flat_cache.clear();
        for id in &self.order {
            if let Some(tokens) = self.sources.get(id) {
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
    /// shifted one token to the right.
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
        let mut inputs = Vec::with_capacity(batch_size * context_len);
        let mut targets = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let start = rng.gen_range(max_start + 1);
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
}
