//! Building the flattened, train/held-out-split token streams
//! [`super::sampling`] draws batches from.
//!
//! Each source contributes its own held-out slice, taken from the end
//! of that source, rather than the corpus as a whole being split into a
//! training head and a held-out tail: a tail split holds out whichever
//! source happens to be last, so its loss measures "text from a source
//! we never saw," a different distribution, not unseen text from the
//! ones training actually used.

use super::Corpus;

/// Fraction of the corpus held out of training, to measure loss on text
/// the model has never been shown. Five percent is enough to be a
/// meaningful sample of a corpus this size while costing almost nothing
/// in training data.
const VALIDATION_FRACTION: f32 = 0.05;

impl Corpus {
    pub(super) fn rebuild_flat_if_needed(&mut self) {
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
}
