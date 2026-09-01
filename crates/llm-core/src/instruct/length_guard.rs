//! Tracking how much has been written, deciding when to stop, and the two
//! ways to drive generation against that rule: one blocking call
//! (`generate_response`), and one step-at-a-time session for a caller
//! that has to yield control between tokens (`ResponseSession`).

use crate::config::ModelConfig;
use crate::generate::{self, SamplingConfig, StopReason};
use crate::model::ModelWeights;
use crate::tokenizer::Tokenizer;

use super::Request;

/// Words per token, used to turn a word target into a token budget.
///
/// Measured against the trained vocabulary this ships with; it's an
/// estimate, and the generation loop counts actual words rather than
/// trusting it. It only has to be close enough that the budget isn't the
/// thing that ends a generation.
const TOKENS_PER_WORD: f32 = 1.6;

/// Tracks how much has been written and decides when to stop.
///
/// Extracted so the CPU and WebGPU generation paths share one copy of
/// the rule rather than two that can drift. The rule: run to the target
/// word count, then keep going only until the next sentence or paragraph
/// boundary so the piece ends on a finished sentence, with a hard
/// ceiling 40% over in case the model never produces one.
pub struct LengthGuard {
    target: Option<usize>,
    words: usize,
    /// Whether the previous piece ended mid-word, so the next one
    /// continues it rather than starting a new word.
    carry: bool,
    stopped_by_length: bool,
}

impl LengthGuard {
    pub fn new(target: Option<usize>) -> Self {
        Self { target, words: 0, carry: false, stopped_by_length: false }
    }

    /// Token budget to give the generator: enough to reach the target
    /// with room to find a boundary, and never unbounded.
    pub fn token_budget(&self) -> usize {
        let words = self.target.unwrap_or(600);
        ((words as f32 * TOKENS_PER_WORD * 1.6) as usize).clamp(32, 8192)
    }

    /// Account for a newly generated piece. Returns false when the
    /// generation should stop.
    pub fn observe(&mut self, piece: &str) -> bool {
        let (counted, ends_mid_word) = count_words(piece, self.carry);
        self.words += counted;
        self.carry = ends_mid_word;
        let Some(target) = self.target else { return true };
        if self.words >= target && ends_a_sentence(piece) {
            self.stopped_by_length = true;
            return false;
        }
        if self.words >= target * 7 / 5 {
            self.stopped_by_length = true;
            return false;
        }
        true
    }

    pub fn words(&self) -> usize {
        self.words
    }

    pub fn stopped_by_length(&self) -> bool {
        self.stopped_by_length
    }
}

/// How a generated answer turned out.
#[derive(Debug, Clone)]
pub struct Response {
    pub text: String,
    pub word_count: usize,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
}

/// Generate the answer to `request`, stopping near the requested length.
///
/// Length is enforced by the caller, not by the model: once the target
/// word count is reached the generation runs on only until the next
/// sentence or paragraph boundary, so it ends on a finished sentence
/// rather than mid-clause. A hard ceiling above the target catches the
/// case where the model never produces a boundary.
///
/// `max_tokens_override`, when given, replaces the word-target-derived
/// budget with a hard token ceiling — the caller asked for exactly this
/// many tokens and no sentence-boundary courtesy is owed past it. The
/// word-target's own early stop (finish the current sentence once the
/// target is reached) still applies underneath it when both are set, so
/// whichever limit is reached first wins.
///
/// `on_progress` is called with each new piece of text and the running
/// word count; returning `false` stops early (a Stop button).
pub fn generate_response(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    request: &Request,
    sampling: &SamplingConfig,
    max_tokens_override: Option<usize>,
    on_progress: &mut dyn FnMut(&str, usize) -> bool,
) -> Response {
    let prompt_tokens = request.to_prompt_tokens(tokenizer);
    let mut guard = LengthGuard::new(request.target_words);
    let max_new_tokens = max_tokens_override.unwrap_or_else(|| guard.token_budget());

    let (text, reason) = generate::generate_stream(
        weights,
        config,
        tokenizer,
        &prompt_tokens,
        max_new_tokens,
        sampling,
        &mut |piece, _| {
            let keep_going = guard.observe(piece);
            if !on_progress(piece, guard.words()) {
                return false;
            }
            keep_going
        },
    );

    Response {
        word_count: text.split_whitespace().count(),
        tokens_generated: tokenizer.encode(&text).len(),
        // generate_stream only knows "the callback returned false," not
        // why — both a real Stop button (on_progress) and the length
        // guard reaching its target end up as StopReason::Caller from
        // its point of view. This is the one place that can tell them
        // apart, from the guard's own flag, and correct it to Budget
        // (labeled "length" in dto.rs) so the length-target UI message
        // is reachable at all, instead of every guard-triggered stop
        // reading identically to a manual Stop.
        stop_reason: if guard.stopped_by_length() { StopReason::Budget } else { reason },
        text,
    }
}

/// `generate_response`, driven one token at a time instead of in a
/// single call.
///
/// `generate_response` cannot be interrupted between tokens — a caller
/// on a single-threaded host (wasm in a browser tab) that needs to yield
/// control back to its event loop mid-generation has no way to do that
/// inside one blocking call. `ResponseSession` exposes the same
/// generator/length-guard machinery through repeated `step()` calls, so
/// a host can await a yield between them, and produces the exact same
/// `Response` `generate_response` would have, via `finish()`.
pub struct ResponseSession<'a> {
    generator: generate::Generator<'a>,
    tokenizer: &'a Tokenizer,
    sampling: SamplingConfig,
    guard: LengthGuard,
    max_new_tokens: usize,
    /// Bytes decoded so far that don't yet form a complete UTF-8
    /// character — see `generate::take_complete_chars`.
    pending: Vec<u8>,
    out: Vec<u32>,
    tokens_this_far: usize,
    reason: StopReason,
    done: bool,
}

impl<'a> ResponseSession<'a> {
    pub fn new(
        weights: &'a ModelWeights,
        config: &'a ModelConfig,
        tokenizer: &'a Tokenizer,
        request: &Request,
        sampling: SamplingConfig,
        max_tokens_override: Option<usize>,
    ) -> Self {
        let prompt_tokens = request.to_prompt_tokens(tokenizer);
        let guard = LengthGuard::new(request.target_words);
        let max_new_tokens = max_tokens_override.unwrap_or_else(|| guard.token_budget());
        let generator = generate::Generator::new(weights, config, &prompt_tokens, sampling.seed);
        Self {
            generator,
            tokenizer,
            sampling,
            guard,
            max_new_tokens,
            pending: Vec::new(),
            out: Vec::new(),
            tokens_this_far: 0,
            reason: StopReason::Budget,
            done: false,
        }
    }

    /// Bytes left over from the last token that don't yet complete a
    /// character, rendered the same lossy way `Tokenizer::decode` would —
    /// so text streamed piece-by-piece agrees with `finish()`'s `text`.
    fn flush_pending(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        tail
    }

    /// Advance by one token. Returns any newly-complete text and, once
    /// the generation has finished (length target, EOS, or `cancel()`),
    /// the reason why — after which further calls keep returning that
    /// same reason with no new text.
    pub fn step(&mut self) -> (String, Option<StopReason>) {
        if self.done {
            return (String::new(), Some(self.reason));
        }
        if self.tokens_this_far >= self.max_new_tokens {
            self.done = true;
            self.reason = StopReason::Budget;
            let flushed = self.flush_pending();
            self.guard.observe(&flushed);
            return (flushed, Some(self.reason));
        }
        let Some(token) = self.generator.step(&self.sampling) else {
            self.done = true;
            self.reason = StopReason::EndOfText;
            let flushed = self.flush_pending();
            self.guard.observe(&flushed);
            return (flushed, Some(self.reason));
        };
        self.out.push(token);
        self.pending.extend_from_slice(self.tokenizer.piece(token));
        let piece = generate::take_complete_chars(&mut self.pending);
        self.tokens_this_far += 1;
        if !self.guard.observe(&piece) {
            self.done = true;
            // Not Caller: the length guard decided this, not `cancel()`
            // (a real Stop button) — Budget is what dto.rs labels
            // "length", so reaching the target reads as "reached the
            // length you asked for" instead of an indistinguishable
            // "stopped".
            self.reason = StopReason::Budget;
            let flushed = self.flush_pending();
            self.guard.observe(&flushed);
            let mut combined = piece;
            combined.push_str(&flushed);
            return (combined, Some(self.reason));
        }
        (piece, None)
    }

    /// Stop early — the CPU-side equivalent of `generate_response`'s
    /// `on_progress` callback returning `false` (a Stop button).
    pub fn cancel(&mut self) -> String {
        if self.done {
            return String::new();
        }
        self.done = true;
        self.reason = StopReason::Caller;
        self.flush_pending()
    }

    pub fn words(&self) -> usize {
        self.guard.words()
    }

    /// The same `Response` `generate_response` would have returned, built
    /// from every token `step()` produced. Call once `step()` has
    /// reported a stop reason, or after `cancel()`.
    pub fn finish(self) -> Response {
        let text = self.tokenizer.decode(&self.out);
        Response {
            word_count: text.split_whitespace().count(),
            tokens_generated: self.tokenizer.encode(&text).len(),
            // Unlike generate_response, step() above already has direct
            // control and sets the correct reason itself — no need to
            // recover it after the fact from the guard's own flag.
            stop_reason: self.reason,
            text,
        }
    }
}

/// Words starting in `piece`, given whether the previous piece ended
/// mid-word. Returns `(new words, ends mid-word)`.
fn count_words(piece: &str, continues_a_word: bool) -> (usize, bool) {
    if piece.is_empty() {
        return (0, continues_a_word);
    }
    let mut count = piece.split_whitespace().count();
    if continues_a_word && !piece.starts_with(char::is_whitespace) && count > 0 {
        count -= 1; // the first run finishes the previous word
    }
    let ends_mid_word = !piece.ends_with(char::is_whitespace);
    (count, ends_mid_word)
}

fn ends_a_sentence(piece: &str) -> bool {
    let trimmed = piece.trim_end();
    trimmed.ends_with(['.', '!', '?', '"']) || piece.contains("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruct::{parse_prompt, Form};

    #[test]
    fn word_counting_handles_pieces_that_split_a_word() {
        // "hel" + "lo world" is two words, not three.
        let (a, mid_a) = count_words("hel", false);
        let (b, _) = count_words("lo world", mid_a);
        assert_eq!(a + b, 2);
    }

    #[test]
    fn generation_stops_near_the_requested_length() {
        // A real length check needs a model that emits words, so this
        // uses the byte-level tokenizer and a tiny model and only
        // asserts the ceiling — the exact stopping point depends on
        // whether the model ever produces a sentence boundary.
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 16,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 64,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 3);
        let t = Tokenizer::byte_level();
        let request = Request {
            form: Form::Novel,
            target_words: Some(20),
            subject: "space".to_string(),
            reference: None,
        };
        let response = generate_response(
            &weights,
            &config,
            &t,
            &request,
            &SamplingConfig { seed: 1, ..Default::default() },
            None,
            &mut |_, _| true,
        );
        assert!(
            response.word_count <= 20 * 7 / 5 + 2,
            "overshot the hard ceiling: {} words",
            response.word_count
        );
    }

    #[test]
    fn max_tokens_override_is_a_hard_ceiling_regardless_of_the_word_target() {
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 16,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 64,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 3);
        let t = Tokenizer::byte_level();
        // A word target of 500 would normally budget hundreds of tokens;
        // the override must win regardless.
        let request = Request {
            form: Form::Novel,
            target_words: Some(500),
            subject: "space".to_string(),
            reference: None,
        };
        // Count callback invocations directly rather than trusting
        // `tokens_generated` (re-derived by re-encoding the decoded
        // text): an untrained, randomly-initialized model's raw byte
        // output is mostly invalid UTF-8, and the lossy decode/re-encode
        // round trip can inflate the count via replacement characters —
        // a measurement artifact, not evidence the loop ran too long.
        let mut pieces = 0;
        generate_response(
            &weights,
            &config,
            &t,
            &request,
            &SamplingConfig { seed: 1, ..Default::default() },
            Some(5),
            &mut |_, _| {
                pieces += 1;
                true
            },
        );
        // <= 6, not 5: generate_stream's loop is bounded to exactly 5
        // iterations, but a final callback flushing any incomplete
        // trailing UTF-8 bytes can fire once more after it.
        assert!(pieces <= 6, "override should cap generation at 5 tokens, got {pieces} callbacks");
    }

    #[test]
    fn a_progress_callback_can_stop_generation() {
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            context_len: 32,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 2);
        let request = parse_prompt("a 500 word story about rain");
        let mut pieces = 0;
        let response = generate_response(
            &weights,
            &config,
            &Tokenizer::byte_level(),
            &request,
            &SamplingConfig::default(),
            None,
            &mut |_, _| {
                pieces += 1;
                pieces < 3
            },
        );
        assert!(pieces <= 3);
        assert!(response.tokens_generated < 100);
    }

    #[test]
    fn response_session_matches_generate_response_step_for_step() {
        // ResponseSession exists so a caller can yield between tokens;
        // it must still produce exactly what one uninterrupted
        // `generate_response` call would, both in the pieces streamed
        // and in the final `Response`.
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 16,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 64,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 3);
        let t = Tokenizer::byte_level();
        let request = Request {
            form: Form::Novel,
            target_words: Some(20),
            subject: "space".to_string(),
            reference: None,
        };
        let sampling = SamplingConfig { seed: 1, ..Default::default() };

        let mut expected_pieces = Vec::new();
        let expected = generate_response(&weights, &config, &t, &request, &sampling, None, &mut |piece, _| {
            expected_pieces.push(piece.to_string());
            true
        });

        let mut session = ResponseSession::new(&weights, &config, &t, &request, sampling, None);
        let mut actual_pieces = Vec::new();
        loop {
            let (piece, reason) = session.step();
            actual_pieces.push(piece);
            if reason.is_some() {
                break;
            }
        }
        let actual = session.finish();

        assert_eq!(actual_pieces, expected_pieces);
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.word_count, expected.word_count);
        assert_eq!(actual.tokens_generated, expected.tokens_generated);
        assert_eq!(actual.stop_reason, expected.stop_reason);
        // The actual bug this test was extended to catch: reaching the
        // word target has to report Budget (dto.rs labels it "length"),
        // not Caller (labeled "stopped") — which is indistinguishable
        // from a real Stop button pressed mid-generation.
        assert_eq!(expected.stop_reason, StopReason::Budget);
    }

    #[test]
    fn response_session_cancel_matches_a_progress_callback_stopping_early() {
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            context_len: 32,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 2);
        let request = parse_prompt("a 500 word story about rain");
        let t = Tokenizer::byte_level();
        let mut session =
            ResponseSession::new(&weights, &config, &t, &request, SamplingConfig::default(), None);
        let mut pieces = 0;
        loop {
            let (_, reason) = session.step();
            pieces += 1;
            if reason.is_some() {
                break;
            }
            if pieces >= 3 {
                session.cancel();
                break;
            }
        }
        let response = session.finish();
        assert!(pieces <= 4, "3 steps plus at most one cancel-flush piece");
        assert_eq!(response.stop_reason, StopReason::Caller);
    }
}
