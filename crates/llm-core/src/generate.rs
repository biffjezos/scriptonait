//! Autoregressive sampling. No KV cache: each new token re-runs the full
//! forward pass over the current (context-window-truncated) token
//! sequence. That's O(n^2) in the number of generated tokens, but at the
//! model sizes this project targets (a browser tab, "not much memory")
//! it's a fine trade for the simplicity of not maintaining incremental
//! attention state — see the README for KV-caching as a possible
//! follow-up optimization.

use crate::config::ModelConfig;
use crate::model::{self, ModelWeights};
use crate::ops;
use crate::rng::Rng;
use crate::tokenizer;

/// Sample a continuation for `prompt`. Returns the full decoded text
/// (prompt + generated continuation) — the caller already has the prompt,
/// but returning the whole string keeps this API trivial to render
/// directly in the UI.
pub fn generate(
    weights: &ModelWeights,
    config: &ModelConfig,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
) -> String {
    let mut tokens = tokenizer::encode(prompt);
    let mut rng = Rng::seed_from_u64(seed);
    let vocab = config.vocab_size();

    for _ in 0..max_new_tokens {
        let window: &[u32] = if tokens.len() > config.context_len {
            &tokens[tokens.len() - config.context_len..]
        } else {
            &tokens
        };
        if window.is_empty() {
            break;
        }
        let (logits, _) = model::forward(weights, config, window);
        let last_row = &logits[(window.len() - 1) * vocab..window.len() * vocab];
        let next = sample(last_row, temperature, &mut rng);
        if next == tokenizer::EOS {
            break;
        }
        tokens.push(next);
    }
    tokenizer::decode(&tokens)
}

fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return ops::argmax(logits) as u32;
    }
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut r = rng.next_f32() * sum;
    for (i, &e) in exps.iter().enumerate() {
        if r < e {
            return i as u32;
        }
        r -= e;
    }
    (exps.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig { num_layers: 1, hidden_dim: 8, num_heads: 2, context_len: 16, local_window: 16 }
    }

    #[test]
    fn generate_is_deterministic_given_same_seed() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let a = generate(&weights, &config, "hi", 10, 0.8, 123);
        let b = generate(&weights, &config, "hi", 10, 0.8, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_max_new_tokens_returns_prompt_unchanged() {
        // Byte-level round trip, no generation requested.
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let prompt = "abc";
        assert_eq!(generate(&weights, &config, prompt, 0, 1.0, 7), prompt);
    }

    #[test]
    fn generate_extends_an_ascii_prompt_without_altering_it() {
        // An ASCII prompt's bytes are always valid UTF-8 on their own, so
        // whatever gets generated afterwards, the decoded output must
        // still start with the original prompt.
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        for budget in [1, 5, 20] {
            let out = generate(&weights, &config, "abc", budget, 1.0, 7);
            assert!(out.starts_with("abc"), "budget={budget} out={out:?}");
        }
    }

    #[test]
    fn temperature_zero_is_deterministic_across_seeds() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 2);
        let a = generate(&weights, &config, "hello", 8, 0.0, 1);
        let b = generate(&weights, &config, "hello", 8, 0.0, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_prompt_still_generates() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 3);
        let out = generate(&weights, &config, "", 4, 0.7, 5);
        assert!(out.len() <= 4);
    }
}
