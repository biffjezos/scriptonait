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
use crate::tokenizer::{self, Tokenizer};

/// Sample a continuation for `prompt`. Returns the full decoded text
/// (prompt + generated continuation) — the caller already has the prompt,
/// but returning the whole string keeps this API trivial to render
/// directly in the UI.
pub fn generate(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    // An empty prompt would otherwise leave `tokens` empty, so the very
    // first window is empty too and the loop below exits before generating
    // anything (see `window.is_empty()`). BOS is what every training
    // window actually starts with (`tokenizer::wrap_with_boundaries`), so
    // it's a real, trained "start of document" signal to condition on
    // instead of an arbitrary filler token - and `decode` drops it from
    // the output, so it never shows up in the generated text.
    if tokens.is_empty() {
        tokens.push(tokenizer::BOS);
    }
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
    tokenizer.decode(&tokens)
}

/// Samples a token id from one row of logits (temperature `<= 0.0` means
/// greedy argmax). Public so callers driving their own token-by-token loop
/// — e.g. wasm-app's WebGPU-accelerated generation path, which needs to
/// interleave GPU forward passes with sampling — can reuse the exact same,
/// already-tested sampling logic instead of re-implementing it.
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
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
        ModelConfig { num_layers: 1, hidden_dim: 8, num_heads: 2, context_len: 16, local_window: 16, ..Default::default() }
    }

    #[test]
    fn generate_is_deterministic_given_same_seed() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let a = generate(&weights, &config, &Tokenizer::byte_level(), "hi", 10, 0.8, 123);
        let b = generate(&weights, &config, &Tokenizer::byte_level(), "hi", 10, 0.8, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_max_new_tokens_returns_prompt_unchanged() {
        // Byte-level round trip, no generation requested.
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let prompt = "abc";
        assert_eq!(generate(&weights, &config, &Tokenizer::byte_level(), prompt, 0, 1.0, 7), prompt);
    }

    #[test]
    fn generate_extends_an_ascii_prompt_without_altering_it() {
        // An ASCII prompt's bytes are always valid UTF-8 on their own, so
        // whatever gets generated afterwards, the decoded output must
        // still start with the original prompt.
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        for budget in [1, 5, 20] {
            let out = generate(&weights, &config, &Tokenizer::byte_level(), "abc", budget, 1.0, 7);
            assert!(out.starts_with("abc"), "budget={budget} out={out:?}");
        }
    }

    #[test]
    fn temperature_zero_is_deterministic_across_seeds() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 2);
        let a = generate(&weights, &config, &Tokenizer::byte_level(), "hello", 8, 0.0, 1);
        let b = generate(&weights, &config, &Tokenizer::byte_level(), "hello", 8, 0.0, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_prompt_still_generates() {
        // Regression test: an empty prompt used to leave `tokens` empty,
        // so the generation loop's first window was empty too and it
        // exited immediately - silently returning "" instead of actually
        // generating anything (see the BOS-seeding fix above).
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 3);
        let out = generate(&weights, &config, &Tokenizer::byte_level(), "", 4, 0.0, 5);
        assert!(!out.is_empty(), "empty prompt should still produce output");
    }
}
