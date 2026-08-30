//! The browser's view of the model: one `WasmLLM` class over `llm-core`
//! and `llm-gpu`.
//!
//! **Training runs on the GPU of the machine that opened the page, and
//! nowhere else.** `train_step` samples a batch of token ids out of the
//! user's own sources and hands it to `llm_gpu::GpuTrainer`, which does
//! the forward pass, the backward pass and the AdamW update in WGSL,
//! with the weights, gradients and optimizer moments resident in GPU
//! memory. There is no CPU training path to fall back to: without
//! WebGPU the page says so and does not train.
//!
//! Generation runs on the GPU too, once `init_gpu` has uploaded the
//! weights; the prompt prefill is the one piece still on the CPU,
//! because that forward pass is the gradient-checked reference the GPU
//! kernels are written against.
//!
//! The trained weights live on the GPU between steps. They come back
//! across the bus only when something actually needs them on this side —
//! exporting a checkpoint, or generating — which `sync_from_gpu` does
//! once, not per step.
//!
//! `WasmLLM`'s own methods are split by concern across sibling modules,
//! since wasm-bindgen only needs the `#[wasm_bindgen]` attribute on
//! whichever `impl` block a given exported method sits in — it does not
//! require all of a type's exported methods to live in one block, or one
//! file. This file keeps the shared state (`Inner`, `GpuBackend`,
//! `WasmLLM` itself) and the `busy`-guard primitive every other module
//! calls; `dto.rs` holds the JS-bridge types and JSON helpers; the rest
//! are one file per concern: `model_state.rs` (construction, shape/plan
//! info, checkpoint I/O), `gpu_session.rs` (opening a device, syncing
//! weights back, the debug forward-pass comparison), `inference.rs`
//! (generation, CPU and GPU), `corpus_api.rs` (source/vocabulary
//! management), `training_api.rs` (the training loop, its schedule
//! controls, and the Adam optimizer state that rides beside a
//! checkpoint).

mod corpus_api;
mod dto;
mod gpu_session;
mod inference;
mod model_state;
mod training_api;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::instruct;
use llm_core::model::ModelWeights;
use llm_core::rng::Rng;
use llm_core::train::TrainConfig;

use dto::{describe, json_string, ParsedPrompt};

/// A ready WebGPU device, this model's weights uploaded to it for
/// generation, and — once training starts — the resident training state.
struct GpuBackend {
    ctx: Rc<llm_gpu::GpuContext>,
    /// `None` only while `generate_on_gpu` has it checked out for the
    /// duration of a generation — see that function's comment for why.
    model: Option<llm_gpu::GpuModel>,
    /// Allocated on the first training step, not at init: it holds the
    /// gradients, both Adam moments and every layer's activations, which
    /// is several times the model's own size and pointless to reserve
    /// for a session that only generates.
    trainer: Option<llm_gpu::GpuTrainer>,
    /// Training step the uploaded generation weights came from, so
    /// training invalidates them instead of generating from stale ones.
    uploaded_at_step: u64,
    summary: String,
    is_software: bool,
}

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

/// Command-buffer size used until this machine has been measured. The
/// benchmark replaces it with whatever this adapter is actually fastest
/// at; it is a starting point, not a tuning.
const DEFAULT_DISPATCHES_PER_SUBMIT: u32 = 32;

/// Ceiling a corpus-derived BPE vocabulary is capped at — used both by
/// `learn_vocabulary` (via `max_vocab_size()` below, which the frontend
/// calls instead of hardcoding this number itself) and `describe_shape`'s
/// pre-model cost estimate, so the two agree on what vocabulary size a
/// given corpus would actually get.
const MAX_VOCAB: usize = 8192;

/// The ceiling `learn_vocabulary` and `describe_shape`'s estimate cap a
/// corpus-derived vocabulary at. No UI control owns this value; it's a
/// model-shape constraint, so it's defined once here and exported rather
/// than re-declared as a literal in the frontend.
#[wasm_bindgen]
pub fn max_vocab_size() -> u32 {
    MAX_VOCAB as u32
}

struct Inner {
    /// True while an async GPU operation owns the training state.
    ///
    /// Every one of them takes the resident `GpuTrainer` out of `gpu`,
    /// awaits, and puts it back. Two at once - a training run and a
    /// profile, say - would each find it missing, build a second one
    /// (four copies of every parameter, again), and race on this
    /// `RefCell` until one of them panicked the whole wasm instance.
    /// Whoever asks second is told no.
    busy: std::cell::Cell<bool>,
    config: ModelConfig,
    /// The host-side copy. Authoritative until the first training step;
    /// after that the GPU's copy is, and this one is refreshed by
    /// `sync_from_gpu` when something needs it here.
    weights: ModelWeights,
    step: u64,
    /// Tokens this model has been trained on, cumulative over every run.
    ///
    /// Not derivable from `step`: a step is one batch, and the batch
    /// size changes between runs and between machines. This is counted
    /// as it happens and carried in the checkpoint, because it — not the
    /// step count — is what says how trained a model is.
    tokens_seen: u64,
    /// The seed the model was built from, so a rebuild at a new
    /// vocabulary size starts from the same place.
    seed: u64,
    /// Batch sampling only — which sequences to train on, not any part
    /// of the arithmetic.
    rng: Rng,
    corpus: Corpus,
    gpu: Option<GpuBackend>,
    train: TrainConfig,
    /// How many GPU operations share one command buffer, measured on
    /// this machine by `bench_step` rather than fixed here: too few
    /// pays submission cost on every dispatch, too many hands the
    /// driver a command buffer long enough to trip its watchdog, and
    /// where the best sits is a property of the adapter.
    dispatches_per_submit: u32,
    /// True when the weights came from a checkpoint rather than from
    /// random initialization — the difference between "this model can
    /// write" and "this model needs training first", which the UI has to
    /// say out loud.
    pretrained: bool,
    /// Which formula `set_project_plan` uses for `warmup_steps`: the
    /// existing 2%-of-plan heuristic (`false`, `TrainConfig::warmup_for`)
    /// or RAdam's own derivation from `beta2` (`true`,
    /// `TrainConfig::warmup_for_variance`). A session-only preference,
    /// not part of `TrainConfig` itself — it picks which formula runs,
    /// it isn't a number the schedule's own math needs at every step.
    warmup_variance: bool,
}

#[wasm_bindgen]
pub struct WasmLLM(Rc<RefCell<Inner>>);

/// The guarded entry points.
///
/// Each of these takes the resident GPU training state out, awaits, and
/// puts it back, so exactly one may be in flight at a time. Anything else
/// asking meanwhile is told no, rather than building a second copy of the
/// training state and racing on the `RefCell` until the wasm instance
/// panics - which is what "RefCell already borrowed" was.
impl WasmLLM {
    fn acquire(&self) -> Result<(), JsValue> {
        let inner = self.0.borrow();
        if inner.busy.get() {
            return Err(js_err(
                "the GPU is already busy with another operation — wait for it to finish, or \
                 press Stop",
            ));
        }
        inner.busy.set(true);
        Ok(())
    }

    fn release(&self) {
        self.0.borrow().busy.set(false);
    }
}

/// Price a model shape before anything is built from it.
///
/// Deliberately a free function rather than a method: the whole point is
/// to answer "what would this cost" while no model exists, which is
/// exactly when somebody is choosing the numbers. Returns JSON.
///
/// `corpus_chars` is how much text is loaded, which decides the
/// vocabulary the model would be built with — and the vocabulary sets
/// the embedding table, which at these sizes is a quarter of the
/// parameters. Passing 0 falls back to the ceiling the page uses.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn describe_shape(
    layers: u32,
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    context_len: u32,
    window: u32,
    corpus_chars: f64,
) -> String {
    let vocab = if corpus_chars > 0.0 {
        llm_core::corpus::suggested_vocab_size(corpus_chars as usize).min(MAX_VOCAB)
    } else {
        MAX_VOCAB
    };
    let config = ModelConfig {
        num_layers: layers as usize,
        hidden_dim: hidden as usize,
        num_heads: heads.max(1) as usize,
        num_kv_heads: kv_heads.max(1) as usize,
        context_len: context_len as usize,
        local_window: window as usize,
        vocab_size: vocab,
        ..ModelConfig::default()
    };
    // Everything below has to survive an invalid shape: the fields are
    // being typed into, so most keystrokes produce one, and an estimator
    // that throws on the way to a valid number is an estimator nobody
    // can watch while they type.
    let problem = match config.validate() {
        Ok(()) => String::new(),
        Err(err) => err.to_string(),
    };
    let divides = config.num_heads > 0
        && config.hidden_dim % config.num_heads == 0
        && config.num_kv_heads > 0
        && config.num_heads % config.num_kv_heads == 0;
    let (params, training_bytes, inference_bytes, efficiency, head_dim, kv_dim, ffn_dim) =
        if divides && config.hidden_dim > 0 && config.context_len > 0 {
            (
                config.param_count(),
                config.memory_bytes(true),
                config.memory_bytes(false),
                config.tile_efficiency(),
                config.head_dim(),
                config.kv_dim(),
                config.ffn_dim(),
            )
        } else {
            (0, 0, 0, 1.0, 0, 0, 0)
        };
    format!(
        "{{\"valid\":{},\"problem\":{},\"params\":{},\"vocabSize\":{},\"headDim\":{},\
         \"kvDim\":{},\"ffnDim\":{},\"trainingBytes\":{},\"inferenceBytes\":{},\
         \"memoryLimitBytes\":{},\"tileEfficiency\":{:.4},\"band\":{}}}",
        problem.is_empty(),
        json_string(&problem),
        params,
        vocab,
        head_dim,
        kv_dim,
        ffn_dim,
        training_bytes,
        inference_bytes,
        llm_core::config::MAX_TRAINING_BYTES,
        efficiency,
        if config.context_len > 0 {
            llm_core::ops::band_width(config.context_len, config.effective_window())
        } else {
            0
        },
    )
}

/// Parse a prompt without a model, so the UI can echo back what it
/// understood while the checkpoint is still downloading.
#[wasm_bindgen]
pub fn parse_prompt_standalone(prompt: String) -> ParsedPrompt {
    describe(&instruct::parse_prompt(&prompt))
}
