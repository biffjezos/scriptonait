//! JS-bridge data types (every `#[wasm_bindgen] pub struct` the page
//! reads results through) and the JSON-building helpers shared across
//! the API surface split over `gpu_session.rs`, `training_api.rs`,
//! `inference.rs`, `model_state.rs` and `corpus_api.rs`.

use wasm_bindgen::prelude::*;

use llm_core::config::LayerSharing;
use llm_core::generate::StopReason;
use llm_core::instruct;

/// `LayerSharing` at the JS boundary: a `mode` tag (0 = Off, 1 =
/// UniformGroups, 2 = RecurrentCore) plus every variant's own numbers,
/// unused ones ignored — the same tag scheme `checkpoint.rs` uses on
/// disk, kept consistent rather than inventing a second one here.
pub(crate) fn layer_sharing_from_raw(
    mode: u32,
    unique_layers: u32,
    prelude_layers: u32,
    coda_layers: u32,
    core_loop_min: u32,
    core_loop_max: u32,
) -> LayerSharing {
    match mode {
        1 => LayerSharing::UniformGroups { unique_layers: unique_layers.max(1) as usize },
        2 => LayerSharing::RecurrentCore {
            prelude_layers: prelude_layers as usize,
            coda_layers: coda_layers as usize,
            core_loop_min: core_loop_min.max(1) as usize,
            core_loop_max: core_loop_max.max(1) as usize,
        },
        _ => LayerSharing::Off,
    }
}

/// The inverse of `layer_sharing_from_raw`, for reporting a model's own
/// shape back to JS via `ModelInfo`.
pub(crate) fn layer_sharing_to_raw(sharing: LayerSharing) -> (u32, u32, u32, u32, u32, u32) {
    match sharing {
        LayerSharing::Off => (0, 0, 0, 0, 0, 0),
        LayerSharing::UniformGroups { unique_layers } => (1, unique_layers as u32, 0, 0, 0, 0),
        LayerSharing::RecurrentCore { prelude_layers, coda_layers, core_loop_min, core_loop_max } => (
            2,
            0,
            prelude_layers as u32,
            coda_layers as u32,
            core_loop_min as u32,
            core_loop_max as u32,
        ),
    }
}

/// Encode `s` as a JSON string literal, quotes included.
///
/// Not `{:?}` (Rust's `Debug` format for `str`), which every hand-rolled
/// JSON `format!` in this file used to reach for because it happens to
/// look like a JSON string for ordinary text. It isn't one: `Debug`
/// renders some control characters as `\u{1}` — variable-width, curly
/// braces — which is not valid JSON syntax (JSON requires exactly four
/// hex digits, no braces). A source's pasted text needs
/// nothing more exotic than a stray vertical-tab or form-feed byte (not
/// uncommon in text copied out of a PDF) to produce one, and
/// `JSON.parse` on the frontend throws "Bad Unicode escape in JSON" the
/// moment it does — which doesn't just fail that one read, it throws
/// out of whatever call in `worker.js` triggered it, including the
/// training loop's own progress reporting.
pub(crate) fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The most recent batch's draws as a JSON array of `{"id":..,"excerpt":..}`
/// objects.
pub(crate) fn json_batch_draws(draws: &[llm_core::corpus::BatchDraw]) -> String {
    let rows = draws
        .iter()
        .map(|d| format!("{{\"id\":{},\"excerpt\":{}}}", json_string(&d.source_id), json_string(&d.excerpt)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rows}]")
}

#[wasm_bindgen]
pub struct SourceStats {
    pub char_count: u32,
    pub byte_count: u32,
    pub token_count: u32,
}

/// The model's shape and provenance, for display.
#[wasm_bindgen]
pub struct ModelInfo {
    pub layers: u32,
    /// `LayerSharing` as a JS-friendly tag + fields — see
    /// `layer_sharing_to_raw`. `layer_sharing_mode`: 0 = Off, 1 =
    /// UniformGroups (ALBERT-style static grouping), 2 = RecurrentCore
    /// (Geiping et al. 2025's variable loop count). `unique_layers` is
    /// UniformGroups' own field; `prelude_layers`/`coda_layers`/
    /// `core_loop_min`/`core_loop_max` are RecurrentCore's.
    pub layer_sharing_mode: u32,
    pub unique_layers: u32,
    pub prelude_layers: u32,
    pub coda_layers: u32,
    pub core_loop_min: u32,
    pub core_loop_max: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub context_len: u32,
    pub window: u32,
    pub vocab_size: u32,
    pub params: f64,
    pub step: f64,
    pub pretrained: bool,
}

/// What a prompt was understood to be asking for. The UI shows this back
/// to the user, because a prompt that was misread should be visibly
/// misread rather than quietly producing the wrong thing.
#[wasm_bindgen]
pub struct ParsedPrompt {
    form: String,
    /// 0 means the prompt didn't ask for a length.
    pub target_words: u32,
    subject: String,
    reference: String,
    instruction: String,
}

#[wasm_bindgen]
impl ParsedPrompt {
    #[wasm_bindgen(getter)]
    pub fn form(&self) -> String {
        self.form.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn subject(&self) -> String {
        self.subject.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn reference(&self) -> String {
        self.reference.clone()
    }
    /// The exact instruction line the model is conditioned on.
    #[wasm_bindgen(getter)]
    pub fn instruction(&self) -> String {
        self.instruction.clone()
    }
}

pub(crate) fn describe(request: &instruct::Request) -> ParsedPrompt {
    ParsedPrompt {
        form: request.form.as_str().to_string(),
        target_words: request.target_words.unwrap_or(0) as u32,
        subject: request.subject.clone(),
        reference: request.reference.clone().unwrap_or_default(),
        instruction: request.instruction(),
    }
}

#[wasm_bindgen]
pub struct GenerationResult {
    pub(crate) text: String,
    pub word_count: u32,
    pub tokens_generated: u32,
    pub(crate) stop_reason: String,
}

#[wasm_bindgen]
impl GenerationResult {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
    /// One of `end-of-text`, `length`, or `stopped` from a real
    /// generation, or `busy`/`no-data` when generation never started.
    #[wasm_bindgen(getter)]
    pub fn stop_reason(&self) -> String {
        self.stop_reason.clone()
    }
}

pub(crate) fn stop_reason_label(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfText => "end-of-text",
        StopReason::Budget => "length",
        StopReason::Caller => "stopped",
    }
}

impl GenerationResult {
    pub(crate) fn from_response(response: instruct::Response) -> Self {
        Self {
            text: response.text,
            word_count: response.word_count as u32,
            tokens_generated: response.tokens_generated as u32,
            stop_reason: stop_reason_label(response.stop_reason).to_string(),
        }
    }
}

/// One training step's numbers, including what it cost to run.
#[wasm_bindgen]
pub struct StepReport {
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens: u32,
    pub step: f64,
    /// Compute dispatches and command-buffer submissions this step made.
    pub dispatches: u32,
    pub submits: u32,
    /// This step's batch, in draw order, as a JSON array of
    /// `{"id":..,"excerpt":..}` — not a `pub` field, since wasm-bindgen
    /// only auto-generates a JS property for `Copy` fields; see
    /// `sources()` below.
    pub(crate) sources_json: String,
}

#[wasm_bindgen]
impl StepReport {
    /// What this step actually trained on: a JSON array of
    /// `{"id":..,"excerpt":..}`, one per window drawn, in draw order —
    /// which source, and a short excerpt of that window's own text.
    #[wasm_bindgen(getter)]
    pub fn sources(&self) -> String {
        self.sources_json.clone()
    }
}
