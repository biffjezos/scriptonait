//! Autoregressive sampling, KV-cached.
//!
//! The prompt is run through the model once (`model::prefill`), and every
//! token after that costs one row of work plus attention against the
//! cached keys and values (`model::decode_step`). The previous
//! implementation re-ran the whole forward pass over the whole context
//! for every single token; at a 512-token context that is roughly 300
//! times more arithmetic to produce the same story.
//!
//! The one thing the cache can't absorb is running past `context_len`.
//! RoPE encodes absolute position, and a cached key carries the rotation
//! of the position it was written at, so positions can't simply be
//! renumbered when the window slides. Instead, when the cache fills, it
//! is rebuilt from the most recent half of the generated text — one extra
//! prefill per half context, which is cheap next to what it buys, and it
//! keeps every position inside the range the model was trained on.

use crate::config::ModelConfig;
use crate::model::{self, GenCache, ModelWeights};
use crate::ops;
use crate::rng::Rng;
use crate::tokenizer::{self, Tokenizer};

/// How a token is picked from a row of logits.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// `<= 0.0` means greedy (deterministic) decoding.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. 0 disables it.
    ///
    /// A model this size puts a long, noisy tail of probability on
    /// tokens that make no sense in context; sampling proportionally
    /// from the full distribution means hitting that tail regularly, and
    /// one wrong token derails the sentence after it. Truncating the
    /// tail is the single biggest quality difference in generated text
    /// at this scale.
    pub top_k: usize,
    /// Nucleus sampling: keep the most likely tokens whose probabilities
    /// sum to `top_p`. 1.0 disables it. Applied after `top_k`.
    pub top_p: f32,
    /// Minimum-probability sampling: keep only tokens at least `min_p`
    /// times as likely as the most likely one. 0.0 disables it. Applied
    /// after `top_k` and before `top_p`.
    ///
    /// This is the truncation that adapts to how sure the model is,
    /// which top-p does not. Top-p keeps a fixed 95% of the mass whether
    /// the model has one obvious continuation or fifty plausible ones:
    /// in the first case it drags in a tail of tokens hundreds of times
    /// less likely than the leader, and in the second it cuts off
    /// candidates that were nearly as good as the one it kept. A
    /// threshold relative to the leader does the opposite of both — a
    /// confident step stays sharp, an uncertain one stays broad — and at
    /// this model size, where the distribution is often barely peaked at
    /// all, that difference is visible in the output.
    pub min_p: f32,
    /// Divides the logit of any token seen in the last
    /// `repetition_window` positions (multiplies it, if it's negative).
    /// 1.0 disables it.
    ///
    /// Small models fall into repetition loops — the same line, then the
    /// same line again — because the most likely continuation of a
    /// phrase they've just produced is that phrase. This is the cheap
    /// structural defence.
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub seed: u64,
    /// Tokens the model is allowed to emit, one flag per id. `None`
    /// allows everything.
    ///
    /// A vocabulary contains every byte value so that any input can be
    /// encoded, but a corpus of English screenplays contains almost none
    /// of the high ones. Their embedding rows are never a target, so a
    /// model early in training has no reason to have pushed them down,
    /// and sampling from the top of an barely-trained distribution emits
    /// them - which is what turns an early sample into a wall of
    /// replacement characters rather than readable nonsense. Restricting
    /// generation to the tokens the training text actually contains
    /// costs nothing and makes early samples legible.
    pub allowed: Option<std::rc::Rc<[bool]>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.1,
            repetition_window: 128,
            seed: 0,
            allowed: None,
        }
    }
}

/// What ended a generation. Surfaced so the UI can say "the model
/// stopped" rather than silently returning something short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted the end-of-document token.
    EndOfText,
    /// The token budget ran out.
    Budget,
    /// The caller's callback asked to stop.
    Caller,
}

/// A generation in progress. Owns the KV cache and the token history.
pub struct Generator<'a> {
    weights: &'a ModelWeights,
    config: &'a ModelConfig,
    cache: GenCache,
    /// Logits for the next token, always current.
    next_logits: Vec<f32>,
    rng: Rng,
    /// Tokens generated so far (not including the prompt).
    generated: Vec<u32>,
}

impl<'a> Generator<'a> {
    /// Start a generation from `prompt_tokens`, which are truncated to
    /// the last `context_len` if longer.
    pub fn new(
        weights: &'a ModelWeights,
        config: &'a ModelConfig,
        prompt_tokens: &[u32],
        seed: u64,
    ) -> Self {
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        // An empty prompt would leave nothing to condition on. BOS is
        // what every training document starts with, so it's a real,
        // trained "start of document" signal rather than filler — and
        // `decode` drops it, so it never appears in the output.
        if tokens.is_empty() {
            tokens.push(tokenizer::BOS);
        }
        if tokens.len() > config.context_len {
            tokens = tokens[tokens.len() - config.context_len..].to_vec();
        }
        let (next_logits, cache) = model::prefill(weights, config, &tokens);
        Self { weights, config, cache, next_logits, rng: Rng::seed_from_u64(seed), generated: Vec::new() }
    }

    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    /// Logits the next token will be sampled from.
    pub fn next_logits(&self) -> &[f32] {
        &self.next_logits
    }

    /// Sample one token, feed it back in, and return it. Returns `None`
    /// once the model emits EOS — the caller decides whether that ends
    /// the generation.
    pub fn step(&mut self, sampling: &SamplingConfig) -> Option<u32> {
        let recent = self.recent_tokens(sampling.repetition_window);
        let next = sample_with(&self.next_logits, sampling, &recent, &mut self.rng);
        if next == tokenizer::EOS {
            return None;
        }
        self.advance(next);
        Some(next)
    }

    /// Force a specific token into the generation (used to seed a
    /// continuation, or by callers doing their own sampling — the WebGPU
    /// path, for one).
    pub fn advance(&mut self, token: u32) {
        self.maybe_reset_cache();
        self.next_logits = model::decode_step(self.weights, self.config, &mut self.cache, token);
        self.generated.push(token);
    }

    /// The last `n` tokens of context (prompt included) — what the
    /// repetition penalty looks at.
    fn recent_tokens(&self, n: usize) -> Vec<u32> {
        let all = self.cache.tokens();
        let lo = all.len().saturating_sub(n);
        all[lo..].to_vec()
    }

    /// When the cache reaches `context_len`, rebuild it from the most
    /// recent half of the text. See this module's header for why the
    /// window can't just slide.
    fn maybe_reset_cache(&mut self) {
        if self.cache.position() < self.config.context_len {
            return;
        }
        let keep = (self.config.context_len / 2).max(1);
        let all = self.cache.tokens();
        let tail: Vec<u32> = all[all.len() - keep..].to_vec();
        let (logits, cache) = model::prefill(self.weights, self.config, &tail);
        self.next_logits = logits;
        self.cache = cache;
    }
}

/// Generate a continuation, calling `on_token` with each newly decoded
/// piece of text. Returning `false` from the callback stops generation
/// (that's how a UI implements a Stop button, and how a word-count
/// target is enforced).
///
/// Returns the generated continuation (not including the prompt) and why
/// it stopped.
pub fn generate_stream(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    sampling: &SamplingConfig,
    on_token: &mut dyn FnMut(&str, usize) -> bool,
) -> (String, StopReason) {
    let mut generator = Generator::new(weights, config, prompt_tokens, sampling.seed);
    let mut out = Vec::with_capacity(max_new_tokens);
    // A BPE token is a byte string, and a character can span several of
    // them, so tokens are decoded through a byte buffer that only
    // releases complete characters. Handing a caller a half-character
    // would show up as a stray replacement character in the UI.
    let mut pending: Vec<u8> = Vec::new();
    let mut reason = StopReason::Budget;
    for i in 0..max_new_tokens {
        let Some(token) = generator.step(sampling) else {
            reason = StopReason::EndOfText;
            break;
        };
        out.push(token);
        pending.extend_from_slice(tokenizer.piece(token));
        let piece = take_complete_chars(&mut pending);
        if !on_token(&piece, i + 1) {
            reason = StopReason::Caller;
            break;
        }
    }
    // Whatever is left is genuinely incomplete (the generation ended
    // mid-character); render it the same lossy way `decode` will, so the
    // streamed text and the returned text agree exactly.
    if !pending.is_empty() {
        let tail = String::from_utf8_lossy(&pending).into_owned();
        pending.clear();
        on_token(&tail, out.len());
    }
    (tokenizer.decode(&out), reason)
}

/// Drain `buf` up to the last complete character, returning that text and
/// leaving any trailing partial character behind for the next token.
///
/// Genuinely invalid bytes (not merely incomplete ones) are replaced with
/// U+FFFD and consumed, matching what `String::from_utf8_lossy` does to
/// the same bytes — which is what keeps the streamed text identical to
/// the decoded whole.
pub fn take_complete_chars(buf: &mut Vec<u8>) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(buf) {
            Ok(s) => {
                out.push_str(s);
                buf.clear();
                return out;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // SAFETY-free: `valid_up_to` is by definition a valid
                // UTF-8 boundary, so this can't fail.
                out.push_str(std::str::from_utf8(&buf[..valid]).unwrap_or(""));
                match e.error_len() {
                    // An invalid sequence: consume it as one replacement
                    // character and keep going.
                    Some(bad) => {
                        out.push('\u{fffd}');
                        buf.drain(..valid + bad);
                    }
                    // Merely incomplete: keep it for the next token.
                    None => {
                        buf.drain(..valid);
                        return out;
                    }
                }
            }
        }
    }
}

/// Sample a continuation for `prompt`, returning the full decoded text
/// (prompt + continuation).
pub fn generate(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    seed: u64,
) -> String {
    let sampling = SamplingConfig {
        temperature,
        // Greedy decoding has to stay exactly greedy — the truncation
        // and penalty defaults would make `temperature: 0.0` mean
        // something other than argmax.
        top_k: if temperature <= 0.0 { 0 } else { SamplingConfig::default().top_k },
        top_p: if temperature <= 0.0 { 1.0 } else { SamplingConfig::default().top_p },
        repetition_penalty: if temperature <= 0.0 { 1.0 } else { SamplingConfig::default().repetition_penalty },
        seed,
        ..SamplingConfig::default()
    };
    let prompt_tokens = tokenizer.encode(prompt);
    let (continuation, _) = generate_stream(
        weights,
        config,
        tokenizer,
        &prompt_tokens,
        max_new_tokens,
        &sampling,
        &mut |_, _| true,
    );
    format!("{prompt}{continuation}")
}

/// Samples a token id from one row of logits under `sampling`.
/// `recent` is the recent token history the repetition penalty applies
/// to; pass an empty slice to skip it.
pub fn sample_with(
    logits: &[f32],
    sampling: &SamplingConfig,
    recent: &[u32],
    rng: &mut Rng,
) -> u32 {
    let mut work: Vec<f32> = logits.to_vec();

    if let Some(allowed) = sampling.allowed.as_deref() {
        for (id, l) in work.iter_mut().enumerate() {
            // End-of-text is always allowed: it is how a generation
            // stops, and it is a special token rather than text.
            if id as u32 != tokenizer::EOS && !allowed.get(id).copied().unwrap_or(true) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    if sampling.repetition_penalty != 1.0 {
        for &id in recent {
            if let Some(l) = work.get_mut(id as usize) {
                // Dividing a positive logit and multiplying a negative
                // one both move it *down*, which is the point; a plain
                // divide would push a negative logit up.
                *l = if *l > 0.0 { *l / sampling.repetition_penalty } else { *l * sampling.repetition_penalty };
            }
        }
    }

    if sampling.temperature <= 0.0 {
        return ops::argmax(&work) as u32;
    }
    for l in work.iter_mut() {
        *l /= sampling.temperature;
    }

    // Candidate ids, most likely first. Only the top slice can matter
    // once top-k/top-p are applied, but the full sort keeps this simple
    // and it's a rounding error next to a forward pass.
    let mut order: Vec<usize> = (0..work.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        work[b].partial_cmp(&work[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    if sampling.top_k > 0 {
        order.truncate(sampling.top_k);
    }

    let max = order.iter().map(|&i| work[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = order.iter().map(|&i| (work[i] - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return order.first().copied().unwrap_or(0) as u32;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Relative to the leader, so this has to come after normalization
    // and before top-p narrows the field.
    if sampling.min_p > 0.0 {
        let threshold = sampling.min_p * probs.first().copied().unwrap_or(0.0);
        // At least one candidate always survives: the leader is by
        // definition `min_p` times itself or better only when `min_p <=
        // 1`, and a caller who passes more than that still gets a token
        // rather than a panic.
        let keep = probs.iter().take_while(|&&p| p >= threshold).count().max(1);
        probs.truncate(keep);
        order.truncate(keep);
        // Truncation changed the mass; the draw below divides by the
        // total it actually has, so nothing else needs rescaling here.
    }

    if sampling.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= sampling.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        order.truncate(keep);
    }

    let total: f32 = probs.iter().sum();
    let mut r = rng.next_f32() * total;
    for (i, &p) in probs.iter().enumerate() {
        if r < p {
            return order[i] as u32;
        }
        r -= p;
    }
    order.last().copied().unwrap_or(0) as u32
}

/// Backwards-compatible plain sampler: temperature only, no truncation.
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    let cfg = SamplingConfig {
        temperature,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        ..SamplingConfig::default()
    };
    sample_with(logits, &cfg, &[], rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig { num_layers: 1, hidden_dim: 8, num_heads: 2, num_kv_heads: 2, context_len: 16, local_window: 16, ..Default::default() }
    }

    #[test]
    fn generate_is_deterministic_given_same_seed() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let t = Tokenizer::byte_level();
        let a = generate(&weights, &config, &t, "hi", 10, 0.8, 123);
        let b = generate(&weights, &config, &t, "hi", 10, 0.8, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_max_new_tokens_returns_prompt_unchanged() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let prompt = "abc";
        assert_eq!(generate(&weights, &config, &Tokenizer::byte_level(), prompt, 0, 1.0, 7), prompt);
    }

    #[test]
    fn generate_extends_an_ascii_prompt_without_altering_it() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 1);
        let t = Tokenizer::byte_level();
        for budget in [1, 5, 20] {
            let out = generate(&weights, &config, &t, "abc", budget, 1.0, 7);
            assert!(out.starts_with("abc"), "budget={budget} out={out:?}");
        }
    }

    #[test]
    fn temperature_zero_is_deterministic_across_seeds() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 2);
        let t = Tokenizer::byte_level();
        let a = generate(&weights, &config, &t, "hello", 8, 0.0, 1);
        let b = generate(&weights, &config, &t, "hello", 8, 0.0, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_prompt_still_generates() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 3);
        let out = generate(&weights, &config, &Tokenizer::byte_level(), "", 4, 0.0, 5);
        assert!(!out.is_empty(), "empty prompt should still produce output");
    }

    /// The load-bearing test for the KV cache: decoding incrementally
    /// must produce exactly what re-running the whole forward pass
    /// produces. If this drifts, generation is quietly running a
    /// different model than training trained.
    #[test]
    fn kv_cached_logits_match_a_full_forward_pass() {
        let config = ModelConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 4,
            num_kv_heads: 2,
            // Long enough that this test never triggers a cache rebuild
            // (that has its own test), but with a narrower attention
            // window than context so the cache's trimming is exercised
            // against the banded attention the forward pass uses.
            context_len: 64,
            local_window: 16,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 9);
        let prompt: Vec<u32> = "the shadows on the wall".bytes().map(u32::from).collect();
        let continuation: Vec<u32> = " are all they know".bytes().map(u32::from).collect();

        let mut generator = Generator::new(&weights, &config, &prompt, 0);
        let vocab = config.vocab_size();
        let mut all = prompt.clone();
        for &token in &continuation {
            // What the cache says the logits are after `all`...
            let cached = generator.next_logits().to_vec();
            let (full, _) = model::forward(&weights, &config, &all);
            let expected = &full[(all.len() - 1) * vocab..all.len() * vocab];
            let worst =
                cached.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 2e-3, "cached logits drifted by {worst} at length {}", all.len());
            generator.advance(token);
            all.push(token);
        }
    }

    /// Same check across the point where the cache has to be rebuilt.
    #[test]
    fn generation_survives_running_past_the_context_length() {
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 8,
            local_window: 8,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 4);
        let t = Tokenizer::byte_level();
        // 40 new tokens against an 8-token context forces several
        // rebuilds.
        let out = generate(&weights, &config, &t, "abc", 40, 0.9, 11);
        assert!(out.starts_with("abc"));
        assert!(out.len() > 3, "generation stalled after the context filled");
    }

    #[test]
    fn top_k_restricts_the_choice_to_the_k_best() {
        // Logit 3 is best, 1 second; with top_k = 2 nothing else can
        // ever be sampled, whatever the seed.
        let logits = vec![0.0, 5.0, 0.0, 9.0, 0.0];
        let cfg = SamplingConfig { temperature: 5.0, top_k: 2, top_p: 1.0, repetition_penalty: 1.0, ..Default::default() };
        for seed in 0..50 {
            let mut rng = Rng::seed_from_u64(seed);
            let picked = sample_with(&logits, &cfg, &[], &mut rng);
            assert!(picked == 3 || picked == 1, "top_k leaked token {picked}");
        }
    }

    #[test]
    fn top_p_keeps_the_nucleus() {
        // One token holds ~all the mass, so any top_p below that must
        // collapse to it.
        let logits = vec![20.0, 0.0, 0.0, 0.0];
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.9, repetition_penalty: 1.0, ..Default::default() };
        for seed in 0..20 {
            let mut rng = Rng::seed_from_u64(seed);
            assert_eq!(sample_with(&logits, &cfg, &[], &mut rng), 0);
        }
    }

    #[test]
    fn repetition_penalty_pushes_recent_tokens_down() {
        // Token 0 wins outright until it's penalized below token 1.
        let logits = vec![2.0, 1.9, 0.0];
        let greedy = SamplingConfig { temperature: 0.0, repetition_penalty: 1.5, ..Default::default() };
        let mut rng = Rng::seed_from_u64(1);
        assert_eq!(sample_with(&logits, &greedy, &[], &mut rng), 0);
        assert_eq!(sample_with(&logits, &greedy, &[0], &mut rng), 1);
    }

    #[test]
    fn streaming_pieces_reassemble_into_the_returned_text() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 6);
        let t = Tokenizer::byte_level();
        let mut streamed = String::new();
        let (text, reason) = generate_stream(
            &weights,
            &config,
            &t,
            &t.encode("INT. "),
            24,
            &SamplingConfig { seed: 3, ..Default::default() },
            &mut |piece, _| {
                streamed.push_str(piece);
                true
            },
        );
        assert_eq!(streamed, text, "streamed pieces must reassemble into the whole output");
        assert!(matches!(reason, StopReason::Budget | StopReason::EndOfText));
    }

    #[test]
    fn a_callback_can_stop_generation_early() {
        let config = tiny_config();
        let weights = ModelWeights::init(&config, 6);
        let t = Tokenizer::byte_level();
        let mut count = 0;
        let (_, reason) = generate_stream(
            &weights,
            &config,
            &t,
            &t.encode("x"),
            100,
            &SamplingConfig::default(),
            &mut |_, n| {
                count = n;
                n < 5
            },
        );
        assert_eq!(reason, StopReason::Caller);
        assert_eq!(count, 5);
    }

    /// A peaked distribution: min-p should cut everything that is not
    /// close to the leader, whatever the tail's total mass is.
    #[test]
    fn min_p_cuts_the_tail_of_a_confident_step() {
        // exp(6) is ~400x exp(0), so with min_p = 0.1 only the leader
        // clears the bar.
        let logits = vec![6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sampling = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.1,
            repetition_penalty: 1.0,
            ..SamplingConfig::default()
        };
        let mut rng = Rng::seed_from_u64(1);
        for _ in 0..64 {
            assert_eq!(sample_with(&logits, &sampling, &[], &mut rng), 0);
        }
    }

    /// The same threshold on a flat distribution keeps everything —
    /// which is the whole point of making it relative to the leader.
    #[test]
    fn min_p_keeps_the_field_when_the_model_is_unsure() {
        let logits = vec![0.10, 0.05, 0.0, -0.05, -0.10];
        let sampling = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.1,
            repetition_penalty: 1.0,
            ..SamplingConfig::default()
        };
        let mut rng = Rng::seed_from_u64(2);
        let mut seen = [false; 5];
        for _ in 0..500 {
            seen[sample_with(&logits, &sampling, &[], &mut rng) as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every token should still be reachable: {seen:?}");
    }

    /// Nothing is ever cut down to nothing, however the threshold is set.
    #[test]
    fn min_p_always_leaves_a_candidate() {
        let logits = vec![1.0, 0.9, 0.8];
        let sampling = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 5.0,
            repetition_penalty: 1.0,
            ..SamplingConfig::default()
        };
        let mut rng = Rng::seed_from_u64(3);
        assert_eq!(sample_with(&logits, &sampling, &[], &mut rng), 0);
    }

    /// Off by default, so adding it changed nothing for anyone who does
    /// not ask for it.
    #[test]
    fn min_p_is_off_unless_asked_for() {
        assert_eq!(SamplingConfig::default().min_p, 0.0);
    }
}
