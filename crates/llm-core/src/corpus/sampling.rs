//! Drawing training and held-out batches from a [`Corpus`]'s flattened
//! token streams (see `super::store`).
//!
//! Two independent strategies live here:
//!
//! - **Random rotating windows** ([`Corpus::sample_batch`]), used for
//!   training: every non-overlapping window in a source is drawn once
//!   before any of them repeat (see [`WindowCursor`]), and consecutive
//!   draws favor different sources (see `rotation`).
//! - **Fixed deterministic windows** ([`fixed_windows`]), used for the
//!   held-out set and the training probe: the same windows every call,
//!   spread evenly across the sources in a stream, so two consecutive
//!   measurements differ only because the model changed.

use std::collections::HashMap;

use super::Corpus;
use crate::rng::Rng;

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

impl Corpus {
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
pub(super) struct WindowCursor {
    seed: u64,
    epoch: u32,
    cursor: u32,
    order: Vec<u32>,
}

impl WindowCursor {
    pub(super) fn new(id: &str) -> Self {
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
/// without having to persist the seed itself. `pub(super)` rather than
/// private: `super::dedup::duplicate_sources` hashes the same way, and
/// shares this rather than reimplementing it.
pub(super) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer;

    #[test]
    fn empty_corpus_cannot_sample() {
        let mut c = Corpus::new();
        assert!(!c.can_sample(8));
        let mut rng = Rng::seed_from_u64(0);
        assert!(c.sample_batch(2, 8, &mut rng).is_none());
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
    fn a_tiny_corpus_has_no_validation_split() {
        let mut c = Corpus::new();
        c.upsert("a", "the quick brown fox jumps over the lazy dog", false);
        assert!(!c.can_validate(32), "too little text to hold any out");
        // ...and training still works.
        let mut rng = Rng::seed_from_u64(1);
        assert!(c.sample_batch(1, 4, &mut rng).is_some());
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
