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
        let batch = corpus.sample_batch(batch_size, self.config.context_len, &mut self.rng)?;
        let total_loss = self.accumulate_gradients(&batch);
        self.grad_accum.scale_(1.0 / batch.batch_size as f32);
        self.adam.step(&mut self.weights, &self.grad_accum, lr);
        self.step += 1;
        Some(total_loss / batch.batch_size as f32)
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
        ModelConfig { num_layers: 2, hidden_dim: 8, num_heads: 2, context_len: 8, local_window: 8 }
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
    fn step_counter_increments() {
        let mut trainer = Trainer::new(tiny_config(), 1);
        let mut corpus = Corpus::new();
        corpus.upsert("a", &"hello world, this is a test corpus. ".repeat(10), false);
        trainer.train_step(&mut corpus, 2, 0.01);
        trainer.train_step(&mut corpus, 2, 0.01);
        assert_eq!(trainer.step, 2);
    }
}
