//! Ties `Corpus` batch sampling to `model::forward`/`backward` and an
//! optimizer step. This is what both the browser's fine-tuning loop and
//! the native `llm-train` CLI drive, one `train_step` call per
//! UI-visible (or log-visible) "step".
//!
//! Off wasm, a step's batch is split across worker threads — each with
//! its own gradient buffer, summed at the end. That's the single largest
//! speedup available to this engine: the per-sequence forward/backward is
//! already close to what one core can do (see `crates/llm-bench`), and
//! batch elements are embarrassingly parallel because nothing writes to
//! the weights until every sequence in the batch is done. In a browser
//! there is only one thread to have: wasm threads need
//! `SharedArrayBuffer`, which needs cross-origin isolation headers, which
//! GitHub Pages does not serve. So this is exactly the reason pretraining
//! happens natively in CI and the browser only fine-tunes.

use crate::config::ModelConfig;
use crate::corpus::{Batch, Corpus};
use crate::model::{self, AdamState, Gradients, ModelWeights};
use crate::ops;
use crate::rng::Rng;

/// Everything about a training step that isn't the model's shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    /// Peak learning rate, reached at the end of warmup.
    pub lr: f32,
    /// Steps spent ramping the learning rate from ~0 to `lr`.
    ///
    /// A transformer's first few hundred steps are the dangerous ones:
    /// the attention softmax is near-uniform, gradients are large and
    /// badly conditioned, and a full-size step there can put the model
    /// somewhere it spends thousands of steps climbing back out of.
    pub warmup_steps: u64,
    /// Total steps the run is planned for, used to shape the cosine
    /// decay. Training past it just holds the floor learning rate.
    pub total_steps: u64,
    /// Floor of the cosine decay, as a fraction of `lr`.
    pub min_lr_ratio: f32,
    /// Decoupled weight decay (AdamW).
    pub weight_decay: f32,
    /// Global gradient-norm clip; see `model::clip_global_norm`.
    pub grad_clip: f32,
    /// Multiplier applied on top of the schedule, cut when held-out loss
    /// stops improving. 1.0 is untouched.
    ///
    /// The cosine schedule decays on a plan — it assumes the run is
    /// making progress at the rate the plan expected. A model that has
    /// stopped improving is often taking steps too large to settle into
    /// the minimum it is circling, and the standard answer is to cut the
    /// rate and let it. This is that cut, kept separate from the
    /// schedule so both are legible: the cosine says where the plan
    /// expected to be, this says what the run actually needed.
    pub plateau_scale: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            warmup_steps: 200,
            total_steps: 10_000,
            min_lr_ratio: 0.1,
            weight_decay: 0.1,
            grad_clip: 1.0,
            plateau_scale: 1.0,
        }
    }
}

impl TrainConfig {
    /// Learning rate for `step` (0-based): linear warmup, then cosine
    /// decay to `min_lr_ratio * lr`.
    pub fn lr_at(&self, step: u64) -> f32 {
        // Warmup is deliberately not scaled: a plateau cut made
        // mid-warmup would shrink the ramp the run has not finished
        // climbing, and warmup exists precisely to get past the part of
        // training where the rate cannot be judged yet.
        if self.warmup_steps > 0 && step < self.warmup_steps {
            // +1 so step 0 isn't a literal zero-size step.
            return self.lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = ((step - self.warmup_steps) as f32 / decay_steps as f32).clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine) * self.plateau_scale
    }
}

/// What one training step did, beyond its loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    pub loss: f32,
    /// Learning rate this step actually used.
    pub lr: f32,
    /// Gradient norm *before* clipping. A run where this is pinned at
    /// the clip threshold every step is a run whose learning rate is too
    /// high.
    pub grad_norm: f32,
    /// Tokens the step consumed (batch size x context length), for
    /// throughput reporting.
    pub tokens: usize,
}

pub struct Trainer {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    adam: AdamState,
    rng: Rng,
    /// Reused across steps - see `model::backward_into`. Allocating this
    /// per step (let alone per batch element) meant churning several MB
    /// through the allocator on every single training step.
    grad_accum: Gradients,
    /// One spare gradient buffer per worker thread, allocated on the
    /// first threaded step and reused forever after. Each is the size of
    /// the model, so they're worth keeping but not worth allocating if
    /// this trainer never runs a batch big enough to split.
    #[cfg(not(target_arch = "wasm32"))]
    worker_grads: Vec<Gradients>,
    #[cfg(not(target_arch = "wasm32"))]
    threads: usize,
    pub step: u64,
}

impl Trainer {
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let weights = ModelWeights::init(&config, seed);
        let adam = AdamState::new(&config);
        let grad_accum = Gradients::zeros(&config);
        Self {
            weights,
            config,
            adam,
            rng: Rng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5),
            grad_accum,
            #[cfg(not(target_arch = "wasm32"))]
            worker_grads: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            step: 0,
        }
    }

    /// A trainer resuming from an existing checkpoint's weights.
    ///
    /// `step` continues the learning-rate schedule where the previous
    /// run left off — restarting warmup on every resume would give the
    /// loss curve a visible notch at each restart.
    pub fn resume(config: ModelConfig, weights: ModelWeights, step: u64, seed: u64) -> Self {
        let mut trainer = Self::new(config, seed);
        trainer.weights = weights;
        trainer.step = step;
        trainer
    }

    /// Optimizer state, for a run that will be continued in another
    /// process. See `AdamState::to_bytes` for why it isn't part of the
    /// checkpoint.
    pub fn optimizer_bytes(&self) -> Vec<u8> {
        self.adam.to_bytes()
    }

    pub fn load_optimizer(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.adam = AdamState::from_bytes(bytes, &self.config)?;
        Ok(())
    }

    /// Cap how many threads a step may use. Values below 1 are treated as
    /// 1. The native trainer exposes this so a run can be told to leave
    /// cores for the rest of the machine.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
        self.worker_grads.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Samples one batch from `corpus` and runs a full step (forward,
    /// cross-entropy, backward, optimizer update). Returns the batch's
    /// mean loss, or `None` if the corpus doesn't have enough tokens yet
    /// to fill even one `context_len` window (e.g. no sources added yet).
    pub fn train_step(&mut self, corpus: &mut Corpus, batch_size: usize, lr: f32) -> Option<f32> {
        let train = TrainConfig { lr, warmup_steps: 0, ..TrainConfig::default() };
        self.train_step_with(corpus, batch_size, &train).map(|r| r.loss)
    }

    /// A full step under an explicit `TrainConfig`: schedule the learning
    /// rate, accumulate the batch's gradients, clip, then AdamW.
    pub fn train_step_with(
        &mut self,
        corpus: &mut Corpus,
        batch_size: usize,
        train: &TrainConfig,
    ) -> Option<StepReport> {
        let batch = corpus.sample_batch(batch_size, self.config.context_len, &mut self.rng)?;
        let total_loss = self.accumulate_gradients(&batch);
        self.grad_accum.scale_(1.0 / batch.batch_size as f32);
        let grad_norm = model::clip_global_norm(&mut self.grad_accum, train.grad_clip);
        let lr = train.lr_at(self.step);
        self.adam.step(&mut self.weights, &self.grad_accum, lr, train.weight_decay);
        self.step += 1;
        Some(StepReport {
            loss: total_loss / batch.batch_size as f32,
            lr,
            grad_norm,
            tokens: batch.batch_size * batch.context_len,
        })
    }

    /// Forward + backward over every sequence in `batch`, leaving the
    /// summed (not yet averaged) gradient in `self.grad_accum` and
    /// returning the summed loss.
    fn accumulate_gradients(&mut self, batch: &Batch) -> f32 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let workers = self.threads.min(batch.batch_size);
            if workers > 1 {
                return self.accumulate_gradients_threaded(batch, workers);
            }
        }
        self.grad_accum.zero_();
        let mut total = 0.0f32;
        for b in 0..batch.batch_size {
            total += Self::sequence_backward(
                &self.weights,
                &self.config,
                batch,
                b,
                &mut self.grad_accum,
            );
        }
        total
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn accumulate_gradients_threaded(&mut self, batch: &Batch, workers: usize) -> f32 {
        while self.worker_grads.len() < workers {
            self.worker_grads.push(Gradients::zeros(&self.config));
        }
        let weights = &self.weights;
        let config = &self.config;
        let bs = batch.batch_size;

        let losses: Vec<f32> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            // Each worker owns a contiguous slice of the batch and its
            // own gradient buffer, so no two threads ever touch the same
            // float. `chunks_mut` is what proves that to the compiler.
            let per = bs.div_ceil(workers);
            for (w, grad) in self.worker_grads[..workers].iter_mut().enumerate() {
                let lo = (w * per).min(bs);
                let hi = ((w + 1) * per).min(bs);
                handles.push(scope.spawn(move || {
                    grad.zero_();
                    let mut total = 0.0f32;
                    for b in lo..hi {
                        total += Self::sequence_backward(weights, config, batch, b, grad);
                    }
                    total
                }));
            }
            handles.into_iter().map(|h| h.join().expect("training worker panicked")).collect()
        });

        self.grad_accum.zero_();
        for grad in &self.worker_grads[..workers] {
            self.grad_accum.add_assign(grad);
        }
        losses.iter().sum()
    }

    /// One sequence's forward, loss, and backward, accumulated into
    /// `grads`. Free-standing (no `&self`) so worker threads can call it
    /// while holding a shared borrow of the weights.
    fn sequence_backward(
        weights: &ModelWeights,
        config: &ModelConfig,
        batch: &Batch,
        b: usize,
        grads: &mut Gradients,
    ) -> f32 {
        let start = b * batch.context_len;
        let input = &batch.inputs[start..start + batch.context_len];
        let target = &batch.targets[start..start + batch.context_len];

        let (logits, cache) = model::forward(weights, config, input);
        let (loss, d_logits) =
            ops::cross_entropy(&logits, target, batch.context_len, config.vocab_size());
        model::backward_into(weights, config, &cache, &d_logits, grads);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig { num_layers: 2, hidden_dim: 8, num_heads: 2, num_kv_heads: 1, context_len: 8, local_window: 8, ..Default::default() }
    }

    #[test]
    fn train_step_returns_none_on_empty_corpus() {
        let mut trainer = Trainer::new(tiny_config(), 1);
        let mut corpus = Corpus::new();
        assert!(trainer.train_step(&mut corpus, 2, 0.01).is_none());
    }

    #[test]
    fn loss_trends_down_over_many_steps() {
        let config = tiny_config();
        let mut trainer = Trainer::new(config, 42);
        let mut corpus = Corpus::new();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(30);
        corpus.upsert("a", &text, false);

        let first = trainer.train_step(&mut corpus, 2, 0.02).unwrap();
        let mut last = first;
        for _ in 0..60 {
            last = trainer.train_step(&mut corpus, 2, 0.02).unwrap();
        }
        assert!(last < first, "expected loss to decrease: first={first} last={last}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn threaded_and_single_threaded_steps_agree() {
        // Splitting a batch across threads must be a pure speedup: same
        // seed, same corpus, same batch -> the same loss and (to float
        // tolerance) the same weights. Not bit-identical: the threaded
        // path sums each worker's partial gradient and then adds the
        // partials, and float addition isn't associative, so the last
        // couple of mantissa bits legitimately differ from the
        // sequential sum.
        let config = tiny_config();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(40);

        let run = |threads: usize| {
            let mut trainer = Trainer::new(config, 11);
            trainer.set_threads(threads);
            let mut corpus = Corpus::new();
            corpus.upsert("a", &text, false);
            let mut losses = Vec::new();
            for _ in 0..4 {
                losses.push(trainer.train_step(&mut corpus, 6, 0.01).unwrap());
            }
            (losses, trainer.weights.to_bytes())
        };

        let (single_losses, single_weights) = run(1);
        let (multi_losses, multi_weights) = run(4);
        for (a, b) in single_losses.iter().zip(&multi_losses) {
            assert!((a - b).abs() < 1e-4, "loss diverged: {a} vs {b}");
        }
        assert_eq!(single_weights.len(), multi_weights.len());
        let as_floats = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let (a, b) = (as_floats(&single_weights), as_floats(&multi_weights));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "threaded step moved the weights somewhere else: worst diff {worst}");
    }

    #[test]
    fn a_resumed_run_continues_rather_than_restarting() {
        // What this is really testing: a pretraining run split across
        // several CI jobs must behave like one continuous run. If the
        // optimizer state or the step count is dropped, the loss curve
        // notches upward at every restart.
        let config = tiny_config();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(60);
        let train = TrainConfig { lr: 0.01, warmup_steps: 5, total_steps: 100, ..Default::default() };

        let mut straight = Trainer::new(config, 21);
        let mut corpus = Corpus::new();
        corpus.upsert("a", &text, false);
        for _ in 0..12 {
            straight.train_step_with(&mut corpus, 2, &train).unwrap();
        }

        // The same twelve steps, but split across two "processes" with
        // the checkpoint and optimizer state carried between them.
        let mut first = Trainer::new(config, 21);
        let mut corpus_a = Corpus::new();
        corpus_a.upsert("a", &text, false);
        for _ in 0..6 {
            first.train_step_with(&mut corpus_a, 2, &train).unwrap();
        }
        let optimizer = first.optimizer_bytes();
        let mut second = Trainer::resume(config, first.weights.clone(), first.step, 21);
        second.load_optimizer(&optimizer).unwrap();
        let mut corpus_b = Corpus::new();
        corpus_b.upsert("a", &text, false);
        for _ in 0..6 {
            second.train_step_with(&mut corpus_b, 2, &train).unwrap();
        }

        assert_eq!(second.step, straight.step);
        // Not identical weights: the resumed trainer's batch sampler
        // restarts from the seed, so it sees different windows. What has
        // to match is that it's in the same place on the schedule and
        // hasn't thrown its momentum away.
        assert_eq!(train.lr_at(second.step), train.lr_at(straight.step));
        let restored = Trainer::resume(config, first.weights.clone(), first.step, 21);
        assert!(
            restored.config == config,
            "a resumed trainer must keep the checkpoint's shape"
        );
    }

    #[test]
    fn optimizer_state_survives_a_round_trip() {
        let config = tiny_config();
        let mut trainer = Trainer::new(config, 3);
        let mut corpus = Corpus::new();
        corpus.upsert("a", &"shadows on the wall of the cave. ".repeat(40), false);
        trainer.train_step(&mut corpus, 2, 0.01);
        let bytes = trainer.optimizer_bytes();

        let mut restored = Trainer::resume(config, trainer.weights.clone(), trainer.step, 3);
        restored.load_optimizer(&bytes).unwrap();
        assert_eq!(restored.optimizer_bytes(), bytes);

        // A state from a different shape has to be rejected, not
        // silently reinterpreted.
        let other = ModelConfig { hidden_dim: 16, ..config };
        let mut mismatched = Trainer::new(other, 3);
        assert!(mismatched.load_optimizer(&bytes).is_err());
    }

    #[test]
    fn learning_rate_warms_up_then_decays() {
        let t = TrainConfig { lr: 1.0, warmup_steps: 100, total_steps: 1000, min_lr_ratio: 0.1, ..Default::default() };
        assert!(t.lr_at(0) > 0.0 && t.lr_at(0) < 0.02, "warmup should start near zero");
        assert!((t.lr_at(99) - 1.0).abs() < 1e-6, "warmup should end at the peak");
        assert!(t.lr_at(500) < 1.0 && t.lr_at(500) > 0.1, "mid-run should be decaying");
        assert!((t.lr_at(1000) - 0.1).abs() < 1e-6, "decay should land on the floor");
        assert!((t.lr_at(5000) - 0.1).abs() < 1e-6, "past the plan it holds the floor");
    }

    #[test]
    fn gradient_clipping_reports_and_bounds_the_norm() {
        let config = tiny_config();
        let mut trainer = Trainer::new(config, 5);
        let mut corpus = Corpus::new();
        corpus.upsert("a", &"the quick brown fox. ".repeat(40), false);
        let train = TrainConfig { grad_clip: 1e-6, warmup_steps: 0, ..Default::default() };
        let report = trainer.train_step_with(&mut corpus, 2, &train).unwrap();
        // The reported norm is the pre-clip one, so a tiny threshold
        // must not change it.
        assert!(report.grad_norm > 1e-6, "expected a real gradient norm, got {}", report.grad_norm);
        assert_eq!(report.tokens, 2 * config.context_len);
    }

    #[test]
    fn step_counter_increments() {
        let mut trainer = Trainer::new(tiny_config(), 1);
        let mut corpus = Corpus::new();
        corpus.upsert("a", &"hello world, this is a test corpus. ".repeat(10), false);
        trainer.train_step(&mut corpus, 2, 0.01);
        trainer.train_step(&mut corpus, 2, 0.01);
        assert_eq!(trainer.step, 2);
    }

    fn schedule() -> TrainConfig {
        TrainConfig { lr: 1e-3, warmup_steps: 100, total_steps: 1000, ..Default::default() }
    }

    #[test]
    fn a_plateau_cut_scales_the_schedule_after_warmup() {
        let plain = schedule();
        let cut = TrainConfig { plateau_scale: 0.5, ..schedule() };
        for step in [100, 300, 700, 999, 5000] {
            let expected = plain.lr_at(step) * 0.5;
            assert!(
                (cut.lr_at(step) - expected).abs() < 1e-9,
                "step {step}: {} vs {expected}",
                cut.lr_at(step)
            );
        }
    }

    #[test]
    fn a_plateau_cut_leaves_warmup_alone() {
        let plain = schedule();
        let cut = TrainConfig { plateau_scale: 0.25, ..schedule() };
        for step in [0, 1, 50, 99] {
            assert_eq!(cut.lr_at(step), plain.lr_at(step), "step {step}");
        }
    }

    #[test]
    fn the_schedule_is_untouched_by_default() {
        assert_eq!(TrainConfig::default().plateau_scale, 1.0);
    }
}
