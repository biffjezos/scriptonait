//! Wire shapes shared between this server and `frontend/backend/
//! remote-backend.js`. Deliberately mirrors what the page already sends
//! `worker.js` for the same actions (`upsert-source`'s `{id, text,
//! isHtml}`, `train`'s settings, `train-progress`/`train-stopped`'s
//! event shape) so the client's existing `onStream` handlers need no
//! new parsing logic to accept them from a remote source instead of a
//! local Worker.

use serde::{Deserialize, Serialize};

/// One corpus source, exactly the shape the page already sends
/// `upsert-source` (see `frontend/worker.js`).
#[derive(Deserialize)]
pub struct SourceUpload {
    pub id: String,
    pub text: String,
    #[serde(rename = "isHtml", default)]
    pub is_html: bool,
}

#[derive(Deserialize)]
pub struct ModelConfigUpload {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub context_len: usize,
    pub local_window: usize,
    pub seed: u64,
    /// How many of `num_layers` depth positions are distinct weight sets
    /// (ALBERT-style static layer sharing — see
    /// `llm_core::config::ModelConfig::layer_group`). Absent from a client
    /// that predates sharing, or a client that just isn't using it —
    /// defaults to `num_layers`, today's behavior.
    pub unique_layers: Option<usize>,
}

/// Exactly one of `config`/`checkpoint_base64` must be set: a fresh
/// model shape, or an existing checkpoint (the same bf16/f32 wire format
/// `llm_core::checkpoint` already reads and writes locally) to resume
/// training on. `sources` is the corpus snapshot uploaded once at
/// session creation — see the "snapshot at start, edits wait" design
/// note: this server never sees a source added or changed after this.
#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub config: Option<ModelConfigUpload>,
    pub checkpoint_base64: Option<String>,
    #[serde(default)]
    pub sources: Vec<SourceUpload>,
}

#[derive(Serialize)]
pub struct CreateSessionResponse {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub vocab_size: u32,
    pub params: f64,
}

/// Mirrors `readTrainingSettings()`'s own fields (`frontend/app.js`) as
/// closely as this server's smaller v1 loop needs them — no held-out
/// loss, plateau detection, or advice yet (see this crate's own README
/// note in `main.rs`); just the numbers a step and a schedule need.
#[derive(Deserialize)]
pub struct TrainStartRequest {
    pub batch_size: u32,
    pub peak_learning_rate: f32,
    pub max_steps: u32,
    #[serde(default)]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip")]
    pub grad_clip: f32,
    /// How often to emit a `train-progress` event, in milliseconds —
    /// same idea as `worker.js`'s `PROGRESS_INTERVAL_MS`.
    #[serde(default = "default_progress_interval_ms")]
    pub progress_interval_ms: u64,
}

fn default_grad_clip() -> f32 {
    1.0
}

fn default_progress_interval_ms() -> u64 {
    250
}

/// One event on the training-progress stream (`GET /session/:id/train/
/// events`, Server-Sent Events). Field names match `worker.js`'s own
/// `post('train-progress', {...})`/`post('train-stopped', {...})`
/// payloads so `remote-backend.js` can hand these to the exact same
/// `onStream` handlers `app.js` already has wired up for local training.
#[derive(Serialize, Clone)]
#[serde(tag = "type")]
pub enum TrainEvent {
    #[serde(rename = "train-progress")]
    Progress {
        step: u64,
        loss: f32,
        #[serde(rename = "gradNorm")]
        grad_norm: f32,
        lr: f32,
        #[serde(rename = "tokensPerSecond")]
        tokens_per_second: f64,
        #[serde(rename = "elapsedSeconds")]
        elapsed_seconds: f64,
        #[serde(rename = "fractionDone")]
        fraction_done: f64,
    },
    #[serde(rename = "train-stopped")]
    Stopped { step: u64, reason: String },
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub adapter: String,
    pub backend: String,
    #[serde(rename = "deviceType")]
    pub device_type: String,
    #[serde(rename = "isSoftware")]
    pub is_software: bool,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
