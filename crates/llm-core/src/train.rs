//! Ties `Corpus` batch sampling to `model::forward`/`backward` and an Adam
//! step. This is what the wasm bindings drive from the browser's training
//! loop (one `train_step` call per UI-visible "step").

use crate::config::ModelConfig;
use crate::corpus::Corpus;
use crate::model::{self, AdamState, Gradients, ModelWeights};
use crate::ops;
use crate::rng::Rng;

pub struct Trainer {
    pub weights: ModelWeights,
    pub config: ModelConfig,
    adam: AdamState,
    rng: Rng,
    pub step: u64,
}

impl Trainer {
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let weights = ModelWeights::init(&config, seed);
        let adam = AdamState::new(&config);
        Self { weights, config, adam, rng: Rng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5), step: 0 }
    }

    /// Samples one batch from `corpus` and runs a full step (forward,
    /// cross-entropy, backward, Adam update). Returns the batch's mean
    /// loss, or `None` if the corpus doesn't have enough tokens yet to
    /// fill even one `context_len` window (e.g. no sources added yet).
    pub fn train_step(&mut self, corpus: &mut Corpus, batch_size: usize, lr: f32) -> Option<f32> {
        let batch = corpus.sample_batch(batch_size, self.config.context_len, &mut self.rng)?;
        let mut total_loss = 0.0f32;
        let mut grad_accum = Gradients::zeros(&self.config);

        for b in 0..batch.batch_size {
            let start = b * batch.context_len;
            let input = &batch.inputs[start..start + batch.context_len];
            let target = &batch.targets[start..start + batch.context_len];

            let (logits, cache) = model::forward(&self.weights, &self.config, input);
            let (loss, d_logits) =
                ops::cross_entropy(&logits, target, batch.context_len, self.config.vocab_size());
            total_loss += loss;

            let grads = model::backward(&self.weights, &self.config, &cache, &d_logits);
            grad_accum.add_assign(&grads);
        }
        grad_accum.scale_(1.0 / batch.batch_size as f32);
        self.adam.step(&mut self.weights, &grad_accum, lr);
        self.step += 1;
        Some(total_loss / batch.batch_size as f32)
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
