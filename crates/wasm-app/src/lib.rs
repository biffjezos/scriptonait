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

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::checkpoint::{Checkpoint, WeightDtype};
use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::generate::{SamplingConfig, StopReason};
use llm_core::instruct;
use llm_core::tokenizer::Tokenizer;
use llm_core::model::{AdamState, ModelWeights};
use llm_core::rng::Rng;
use llm_core::train::TrainConfig;

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

/// The most recent batch's draws as a JSON array of `{"id":..,"excerpt":..}`
/// objects, `{:?}` handling the quoting and escaping of each string.
fn json_batch_draws(draws: &[llm_core::corpus::BatchDraw]) -> String {
    let rows = draws
        .iter()
        .map(|d| format!("{{\"id\":{:?},\"excerpt\":{:?}}}", d.source_id, d.excerpt))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rows}]")
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

/// Give the host's event loop one turn before continuing.
///
/// Scheduled via `setTimeout(0)` rather than a microtask (a bare
/// resolved `Promise`): a microtask queue drains completely before the
/// event loop is allowed to process anything else, including the
/// callback a GPU buffer-mapping readback resolves through, so a
/// microtask-only yield would not actually let a concurrently in-flight
/// training step's readback make progress. `setTimeout` is a macrotask,
/// scheduled the same way, so it does.
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let global: JsValue = js_sys::global().into();
        let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
        match set_timeout {
            Some(set_timeout) => {
                // Errors here would mean the host has no `setTimeout`
                // (true of every wasm host this project runs in), so
                // there is nothing useful to do with one but drop it —
                // the `Promise` just never resolves, and neither
                // `resolve` nor `reject` were called.
                let resolve: JsValue = resolve.into();
                let _ = set_timeout.call2(&global, &resolve, &JsValue::from_f64(0.0));
            }
            None => {
                // No `setTimeout` on this host: resolve immediately
                // rather than hang forever on a yield nothing can ever
                // deliver.
                let _ = resolve.call0(&JsValue::UNDEFINED);
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
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

fn describe(request: &instruct::Request) -> ParsedPrompt {
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
    text: String,
    pub word_count: u32,
    pub tokens_generated: u32,
    stop_reason: String,
}

#[wasm_bindgen]
impl GenerationResult {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
    /// One of `end-of-text`, `length`, or `stopped`.
    #[wasm_bindgen(getter)]
    pub fn stop_reason(&self) -> String {
        self.stop_reason.clone()
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
    sources_json: String,
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

fn stop_reason_label(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfText => "end-of-text",
        StopReason::Budget => "length",
        StopReason::Caller => "stopped",
    }
}

/// Hand one piece of text to the page. A callback that returns nothing
/// means "keep going" — only an explicit `false` stops the generation —
/// and one that throws stops it, because a page in that state has
/// nowhere to put the rest.
fn report(on_token: &js_sys::Function, piece: &str, words: usize) -> bool {
    let piece = JsValue::from_str(piece);
    let words = JsValue::from_f64(words as f64);
    match on_token.call2(&JsValue::UNDEFINED, &piece, &words) {
        Ok(value) => value.as_bool().unwrap_or(true),
        Err(_) => false,
    }
}

impl GenerationResult {
    fn from_response(response: instruct::Response) -> Self {
        Self {
            text: response.text,
            word_count: response.word_count as u32,
            tokens_generated: response.tokens_generated as u32,
            stop_reason: stop_reason_label(response.stop_reason).to_string(),
        }
    }
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

#[wasm_bindgen]
impl WasmLLM {
    /// One training step over the current sources, on the GPU.
    pub async fn train_step(&self, batch_size: u32) -> Result<Option<StepReport>, JsValue> {
        self.acquire()?;
        let result = self.train_step_inner(batch_size).await;
        self.release();
        result
    }

    /// One step, timed per phase; see `profile_step_inner`.
    pub async fn profile_step(
        &self,
        batch_size: u32,
        dispatches_per_submit: u32,
    ) -> Result<String, JsValue> {
        self.acquire()?;
        let result = self.profile_step_inner(batch_size, dispatches_per_submit).await;
        self.release();
        result
    }

    /// Time one step at a given batch size and command-buffer size,
    /// changing nothing: no weight update, no step counter. This is the
    /// measurement the machine profile is built from.
    pub async fn bench_step(
        &self,
        batch_size: u32,
        dispatches_per_submit: u32,
    ) -> Result<f64, JsValue> {
        self.acquire()?;
        let result = self.bench_step_inner(batch_size, dispatches_per_submit).await;
        self.release();
        result
    }

    /// How many GPU operations currently share a command buffer.
    pub fn dispatches_per_submit(&self) -> u32 {
        self.0.borrow().dispatches_per_submit
    }

    /// Set it, from a stored machine profile or from a fresh benchmark.
    pub fn set_dispatches_per_submit(&self, n: u32) {
        self.0.borrow_mut().dispatches_per_submit = n.clamp(1, 1024);
    }

    /// Bring the trained weights back from the GPU; see
    /// `sync_from_gpu_inner`.
    pub async fn sync_from_gpu(&self) -> Result<(), JsValue> {
        self.acquire()?;
        let result = self.sync_from_gpu_inner().await;
        self.release();
        result
    }

    /// Loss on a fixed set of held-out windows — the number that
    /// separates learning from memorizing. Returns -1 when there is not
    /// enough text to hold any out, or no training state on the GPU yet.
    ///
    /// `batch_size` is how many windows the set holds, not a sample
    /// size: the same windows come back every call, so two measurements
    /// differ only because the weights differ.
    pub async fn validation_loss(&self, batch_size: u32) -> Result<f32, JsValue> {
        self.acquire()?;
        let result = self.validation_loss_inner(batch_size).await;
        self.release();
        result
    }

    /// Loss on a fixed set of *training* windows, drawn exactly as the
    /// held-out set is drawn. The number to compare held-out against.
    pub async fn training_probe_loss(&self, batch_size: u32) -> Result<f32, JsValue> {
        self.acquire()?;
        let result = self.fixed_set_loss(batch_size, false).await;
        self.release();
        result
    }

    /// Time each kernel at this model's shapes, as JSON. What the phase
    /// profile is to a step, this is to a phase.
    pub async fn profile_kernels(&self, reps: u32) -> Result<String, JsValue> {
        self.acquire()?;
        let result = self.profile_kernels_inner(reps).await;
        self.release();
        result
    }

    /// The Adam moment buffers, serialized. Saved beside the checkpoint
    /// so a restored model resumes with its momentum instead of jolting.
    pub async fn export_optimizer(&self) -> Result<Vec<u8>, JsValue> {
        self.acquire()?;
        let result = self.export_optimizer_inner().await;
        self.release();
        result
    }

    /// Restore moment buffers saved by `export_optimizer`. A mismatch
    /// with the current model's shape is an error, not a silent reset.
    pub async fn import_optimizer(&self, bytes: &[u8]) -> Result<(), JsValue> {
        self.acquire()?;
        let result = self.import_optimizer_inner(bytes).await;
        self.release();
        result
    }
}

#[wasm_bindgen]
impl WasmLLM {
    /// Load the model the site ships with.
    ///
    /// The checkpoint carries its shape, its weights, and the tokenizer
    /// its token ids belong to, so there is nothing for the caller to
    /// get wrong and nothing to configure.
    pub fn from_checkpoint(bytes: &[u8]) -> Result<WasmLLM, JsValue> {
        let checkpoint = Checkpoint::from_bytes(bytes).map_err(js_err)?;
        Ok(WasmLLM(Rc::new(RefCell::new(Inner {
            busy: std::cell::Cell::new(false),
            dispatches_per_submit: DEFAULT_DISPATCHES_PER_SUBMIT,
            config: checkpoint.config,
            weights: checkpoint.weights,
            step: checkpoint.step,
            tokens_seen: checkpoint.tokens_seen,
            seed: 1,
            rng: Rng::seed_from_u64(1),
            corpus: Corpus::with_tokenizer(checkpoint.tokenizer),
            gpu: None,
            // Fine-tuning a trained model wants a small, flat learning
            // rate: the point is to bend it toward your text, not to
            // re-run a pretraining schedule over it. The plan and the
            // plateau cut, though, are the checkpoint's own — restoring
            // them (rather than the flat defaults below) is what makes a
            // reload continue the same absolute schedule instead of
            // starting a fresh 2,000-step one every time the page loads.
            train: if checkpoint.planned_steps > 0 {
                TrainConfig {
                    lr: 5e-5,
                    total_steps: checkpoint.planned_steps as u64,
                    warmup_steps: TrainConfig::warmup_for(checkpoint.planned_steps as u64),
                    min_lr_ratio: 1.0,
                    plateau_scale: checkpoint.plateau_scale,
                    ..TrainConfig::default()
                }
            } else {
                TrainConfig {
                    lr: 5e-5,
                    warmup_steps: 20,
                    total_steps: 2000,
                    min_lr_ratio: 1.0,
                    ..TrainConfig::default()
                }
            },
            pretrained: true,
        }))))
    }

    /// A fresh, untrained model — the "train one from scratch on my own
    /// text" path, kept for people who want it.
    #[wasm_bindgen(constructor)]
    pub fn new(
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        context_len: u32,
        local_window: u32,
        seed: f64,
    ) -> Result<WasmLLM, JsValue> {
        let tokenizer = Tokenizer::byte_level();
        let config = ModelConfig {
            num_layers: num_layers as usize,
            hidden_dim: hidden_dim as usize,
            num_heads: num_heads as usize,
            num_kv_heads: num_kv_heads as usize,
            context_len: context_len as usize,
            local_window: local_window as usize,
            vocab_size: tokenizer.vocab_size(),
            ..Default::default()
        };
        config.validate().map_err(js_err)?;
        Ok(WasmLLM(Rc::new(RefCell::new(Inner {
            busy: std::cell::Cell::new(false),
            dispatches_per_submit: DEFAULT_DISPATCHES_PER_SUBMIT,
            config,
            weights: ModelWeights::init(&config, seed as u64),
            step: 0,
            tokens_seen: 0,
            seed: seed as u64,
            rng: Rng::seed_from_u64(seed as u64),
            corpus: Corpus::with_tokenizer(tokenizer),
            gpu: None,
            train: TrainConfig::default(),
            pretrained: false,
        }))))
    }

    // --- Model info ------------------------------------------------------

    pub fn info(&self) -> ModelInfo {
        let inner = self.0.borrow();
        let config = inner.config;
        ModelInfo {
            layers: config.num_layers as u32,
            hidden: config.hidden_dim as u32,
            heads: config.num_heads as u32,
            kv_heads: config.num_kv_heads as u32,
            context_len: config.context_len as u32,
            window: config.effective_window() as u32,
            vocab_size: config.vocab_size as u32,
            params: config.param_count() as f64,
            step: inner.step as f64,
            pretrained: inner.pretrained,
        }
    }

    /// Measure a piece of generated text against the corpus it was
    /// trained on, and convert a loss into bits per byte. JSON.
    ///
    /// Loss alone cannot say whether the output is English. These can:
    /// what fraction of the words appear anywhere in the user's own
    /// sources, how much of it is a four-word run it already wrote, and
    /// how many bits it takes this model to encode a byte of their text
    /// — which, unlike loss, is comparable between two vocabularies and
    /// against gzip.
    ///
    /// `loss` is a per-token cross-entropy in nats; pass a negative
    /// number when there isn't one to convert.
    pub fn evaluate(&self, text: String, loss: f32) -> String {
        let inner = &mut *self.0.borrow_mut();
        let bytes_per_token = inner.corpus.bytes_per_token();
        let known = inner.corpus.word_vocabulary();
        let stats = llm_core::eval::text_stats(&text, &known);
        let bits = if loss >= 0.0 {
            llm_core::eval::bits_per_byte_from_ratio(loss, bytes_per_token)
        } else {
            0.0
        };
        let unknown = stats
            .unknown_examples
            .iter()
            .map(|w| format!("{w:?}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"words\":{},\"knownWordRate\":{:.4},\"repeated4gramRate\":{:.4},\
             \"distinctWordRate\":{:.4},\"bitsPerByte\":{:.4},\"bytesPerToken\":{:.3},\
             \"unknownExamples\":[{}]}}",
            stats.words,
            stats.known_word_rate,
            stats.repeated_4gram_rate,
            stats.distinct_word_rate,
            bits,
            bytes_per_token,
            unknown,
        )
    }

    /// Everything a training plan is computed from, as JSON.
    ///
    /// The schedule's own numbers (peak rate, warmup length, planned
    /// length) and the corpus's own numbers (how many tokens train, how
    /// many are held out) live on this side, so the page asks rather than
    /// keeping a second copy that drifts. What the numbers *mean* — which
    /// phase the run is in, what to do about it — is worked out where it
    /// can be changed quickly, in the worker.
    pub fn training_plan(&self) -> String {
        let inner = &mut *self.0.borrow_mut();
        let config = inner.config;
        let train = inner.train;
        let step = inner.step;
        let context_len = config.context_len;
        // Characters as well as tokens, because the token count moves
        // when the vocabulary is relearned and the character count does
        // not. The same corpus reads as 6.5M tokens at a 4k vocabulary
        // and 4.5M at an 8k one, and somebody watching that number
        // change with no explanation is right to distrust all of them.
        let corpus_chars = inner.corpus.total_chars();
        let training_tokens = inner.corpus.training_tokens();
        let validation_tokens = inner.corpus.validation_tokens();
        let mix = inner
            .corpus
            .mix()
            .iter()
            .map(|(kind, tokens)| {
                format!("{{\"kind\":{:?},\"label\":{:?},\"tokens\":{}}}", kind.key(), kind.label(), tokens)
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"step\":{},\"plannedSteps\":{},\"warmupSteps\":{},\"peakLr\":{},\
             \"minLrRatio\":{},\"lrNow\":{},\"weightDecay\":{},\"gradClip\":{},\
             \"params\":{},\"layers\":{},\"hidden\":{},\"vocabSize\":{},\
             \"contextLen\":{},\"sources\":{},\"corpusTokens\":{},\
             \"trainingTokens\":{},\"validationTokens\":{},\"pretrained\":{},\
             \"plateauScale\":{},\"tokensSeen\":{},\"corpusChars\":{},\"startStep\":{},\
             \"mix\":[{}]}}",
            step,
            train.total_steps,
            train.warmup_steps,
            train.lr,
            train.min_lr_ratio,
            train.lr_at(step),
            train.weight_decay,
            train.grad_clip,
            config.param_count(),
            config.num_layers,
            config.hidden_dim,
            config.vocab_size,
            context_len,
            inner.corpus.num_sources(),
            inner.corpus.total_tokens(),
            training_tokens,
            validation_tokens,
            inner.pretrained,
            train.plateau_scale,
            inner.tokens_seen,
            corpus_chars,
            train.start_step,
            mix,
        )
    }

    /// Rough memory estimate in bytes; see `ModelConfig::memory_bytes`.
    pub fn memory_bytes(&self, training: bool) -> f64 {
        self.0.borrow().config.memory_bytes(training) as f64
    }

    // --- Prompt understanding -------------------------------------------

    /// Read a prompt as an instruction, without generating anything.
    pub fn parse_prompt(&self, prompt: String) -> ParsedPrompt {
        describe(&instruct::parse_prompt(&prompt))
    }

    /// Ask the browser for a WebGPU device and upload the weights to it.
    ///
    /// Call once after loading a model. Returns a short description of
    /// what the browser actually gave us — a device name, or why there
    /// isn't one. Generation survives a failure here (it finishes on the
    /// CPU); training does not, because there is no CPU training path.
    pub async fn init_gpu(&self) -> Result<String, JsValue> {
        self.acquire()?;
        let result = self.init_gpu_inner().await;
        self.release();
        result
    }

    async fn init_gpu_inner(&self) -> Result<String, JsValue> {
        let (config, weights) = {
            let inner = self.0.borrow();
            (inner.config, inner.weights.clone())
        };
        if !llm_gpu::supports(&config) {
            return Err(js_err("this model's shape is past what the GPU kernels handle"));
        }
        let ctx = Rc::new(llm_gpu::GpuContext::new().await.map_err(js_err)?);
        let model = llm_gpu::GpuModel::upload(&ctx, &weights, &config).map_err(js_err)?;
        let summary = ctx.adapter_summary.clone();
        let is_software = ctx.is_software;
        let step = self.0.borrow().step;
        self.0.borrow_mut().gpu = Some(GpuBackend {
            ctx,
            model: Some(model),
            trainer: None,
            uploaded_at_step: step,
            summary: summary.clone(),
            is_software,
        });
        Ok(summary)
    }

    /// Everything known about the device, as JSON, for the page to log.
    /// This is the answer to "is it actually using my GPU?" — including
    /// the case that matters most, a software adapter that reports itself
    /// as WebGPU and runs at CPU speed.
    pub fn gpu_report(&self) -> String {
        let inner = self.0.borrow();
        let Some(gpu) = inner.gpu.as_ref() else {
            return "{\"available\":false}".to_string();
        };
        let ctx = &gpu.ctx;
        let training_bytes = gpu.trainer.as_ref().map(|t| t.allocated_bytes()).unwrap_or(0);
        format!(
            "{{\"available\":true,\"adapter\":{:?},\"backend\":{:?},\"deviceType\":{:?},\
             \"isSoftware\":{},\"f16\":{},\"maxWorkgroupsPerDimension\":{},\
             \"maxStorageBufferBindingSize\":{},\"maxBufferSize\":{},\
             \"trainingStateBytes\":{},\"trainerReady\":{}}}",
            ctx.adapter_name,
            ctx.backend,
            ctx.device_type,
            ctx.is_software,
            ctx.has_f16,
            ctx.max_workgroups_per_dimension,
            ctx.max_storage_buffer_binding_size,
            ctx.max_buffer_size,
            training_bytes,
            gpu.trainer.is_some(),
        )
    }

    /// True when the browser handed us a software rasterizer rather than
    /// real hardware — which trains at CPU speed and is worth saying out
    /// loud rather than leaving someone to wonder.
    pub fn gpu_is_software(&self) -> bool {
        self.0.borrow().gpu.as_ref().is_some_and(|gpu| gpu.is_software)
    }

    /// What generation will actually run on: a device description, or
    /// "CPU". The page reports this rather than offering it as a choice.
    pub fn device_summary(&self) -> String {
        match &self.0.borrow().gpu {
            Some(gpu) if gpu.is_software => format!("{} (software renderer)", gpu.summary),
            Some(gpu) => gpu.summary.clone(),
            None => "CPU".to_string(),
        }
    }

    pub fn using_gpu(&self) -> bool {
        self.0.borrow().gpu.is_some()
    }

    /// Generate an answer to `prompt`.
    ///
    /// `on_token` is called with `(piece: string, words: number)` as the
    /// text arrives; returning `false` from it stops the generation,
    /// which is how the UI's Stop button works.
    ///
    /// `extra_context`, if non-empty, is folded into the instruction's
    /// subject — that's how retrieved scenes and the story-state
    /// preamble get in without inventing a second prompt format.
    /// `prefer_gpu` chooses the device for this call, independent of
    /// whether training is using the GPU right now.
    ///
    /// When `false`, this deliberately takes none of the machinery below:
    /// no `acquire()`, no `busy` check, no `sync_from_gpu_inner()`. CPU
    /// inference has to work *while training holds the GPU*, using
    /// whatever weights are already resident on this side — the current
    /// state if a sync has already happened, the previous state if
    /// training hasn't synced back yet. Waiting for that sync, or being
    /// refused because the GPU is busy, is exactly the failure this
    /// avoids: there is nothing here for a concurrent training step to
    /// race, because this path never touches anything a training step
    /// touches.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        prompt: String,
        extra_context: String,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        seed: f64,
        prefer_gpu: bool,
        // A hard ceiling on tokens generated, overriding the length the
        // prompt itself asked for ("write a 700 word novel..."). 0 means
        // no override — length stays whatever the prompt implies (or the
        // default budget, if it implies nothing), exactly as before this
        // parameter existed.
        max_tokens: u32,
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let max_tokens_override = (max_tokens > 0).then_some(max_tokens as usize);
        if !prefer_gpu {
            let request = self.build_request(&prompt, &extra_context);
            let sampling =
                self.sampling_config(temperature, top_k, top_p, min_p, repetition_penalty, seed);
            return self.generate_on_cpu(&request, &sampling, max_tokens_override, on_token).await;
        }

        // Generation owns the GPU for as long as it runs, and it pulls
        // the trained weights back across the bus first. Two of those at
        // once — or one alongside a save, which is what happened —
        // put two futures through `sync_from_gpu_inner` together and
        // exhausted the wasm heap between them. `busy` exists to stop
        // exactly that.
        //
        // A single attempt, not a retry: if the GPU is busy, this
        // returns "busy" once and stops. Nothing here polls or loops
        // waiting for training to let go of the device.
        if self.acquire().is_err() {
            return GenerationResult {
                text: String::new(),
                word_count: 0,
                tokens_generated: 0,
                stop_reason: "busy".to_string(),
            };
        }
        let result = self.generate_inner(prompt, extra_context, temperature, top_k, top_p, min_p,
            repetition_penalty, seed, max_tokens_override, on_token).await;
        self.release();
        result
    }

    fn sampling_config(
        &self,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        seed: f64,
    ) -> SamplingConfig {
        SamplingConfig {
            temperature,
            top_k: top_k as usize,
            top_p,
            min_p,
            repetition_penalty,
            seed: seed as u64,
            // Never emit a token the training text does not contain. A
            // vocabulary holds every byte value so any input can be
            // encoded, but a model early in training has had no reason to
            // push the unused ones down, and they are what fills an early
            // sample with replacement characters.
            allowed: Some(self.0.borrow().corpus.seen_tokens()),
            ..SamplingConfig::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_inner(
        &self,
        prompt: String,
        extra_context: String,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        seed: f64,
        max_tokens_override: Option<usize>,
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let request = self.build_request(&prompt, &extra_context);
        let sampling = self.sampling_config(temperature, top_k, top_p, min_p, repetition_penalty, seed);

        // Training since the last generation left the current weights on
        // the GPU's training buffers; bring them across and re-upload
        // them for decoding. Doing it here, once, is why a training step
        // itself never pays for a weight transfer. A failure is not
        // fatal: generation falls back to the CPU below.
        let _ = self.sync_from_gpu_inner().await;

        let gpu_is_current = {
            let inner = self.0.borrow();
            inner.gpu.as_ref().is_some_and(|g| g.uploaded_at_step == inner.step)
        };
        if gpu_is_current {
            match self.generate_on_gpu(&request, &sampling, max_tokens_override, on_token).await {
                Ok(result) => return result,
                Err(_) => {
                    // A device that was lost or a kernel that failed:
                    // drop it and finish on the CPU rather than handing
                    // the user an error where a story should be.
                    self.0.borrow_mut().gpu = None;
                }
            }
        }
        self.generate_on_cpu(&request, &sampling, max_tokens_override, on_token).await
    }

    fn build_request(&self, prompt: &str, extra_context: &str) -> instruct::Request {
        let mut request = instruct::parse_prompt(prompt);
        if !extra_context.trim().is_empty() {
            request.subject = if request.subject.is_empty() {
                extra_context.trim().to_string()
            } else {
                format!("{}; {}", request.subject, extra_context.trim())
            };
        }
        request
    }

    /// Generation runs one token at a time via `ResponseSession`, with a
    /// yield back to the browser's event loop between tokens.
    ///
    /// It used to be one uninterrupted call into `generate_response`.
    /// wasm in a browser tab is single-threaded, so that blocked the
    /// event loop for the whole generation — including the callback a
    /// concurrently in-flight GPU training step's buffer readback was
    /// waiting on, stalling training for as long as a long CPU
    /// generation ran. Yielding between tokens gives that callback (and
    /// anything else queued) a turn.
    ///
    /// Deliberately clones the weights/config/tokenizer out of `inner`
    /// up front rather than holding a `Ref` across the loop: a `Ref`
    /// held across a yield point would panic the first `borrow_mut()`
    /// a concurrent caller made while this was suspended (see
    /// `acquire`/`release` above for why nothing here can risk that).
    async fn generate_on_cpu(
        &self,
        request: &instruct::Request,
        sampling: &SamplingConfig,
        max_tokens_override: Option<usize>,
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let (weights, config, tokenizer) = {
            let inner = self.0.borrow();
            (inner.weights.clone(), inner.config, inner.corpus.tokenizer().clone())
        };
        let mut session = instruct::ResponseSession::new(
            &weights,
            &config,
            &tokenizer,
            request,
            sampling.clone(),
            max_tokens_override,
        );
        loop {
            let (piece, reason) = session.step();
            let keep_going = report(on_token, &piece, session.words());
            if reason.is_some() {
                break;
            }
            if !keep_going {
                session.cancel();
                break;
            }
            yield_to_event_loop().await;
        }
        GenerationResult::from_response(session.finish())
    }

    /// The same generation, decoded on the GPU.
    ///
    /// The prompt is prefilled on the CPU and its keys and values handed
    /// to the GPU; from there each token is one dispatch batch and one
    /// readback of the logits, with sampling and the stopping rule
    /// staying on the CPU where they're tested.
    async fn generate_on_gpu(
        &self,
        request: &instruct::Request,
        sampling: &SamplingConfig,
        max_tokens_override: Option<usize>,
        on_token: &js_sys::Function,
    ) -> Result<GenerationResult, String> {
        let (weights, config, tokenizer, prompt_tokens) = {
            let inner = self.0.borrow();
            let tokenizer = inner.corpus.tokenizer().clone();
            let prompt_tokens = request.to_prompt_tokens(&tokenizer);
            (inner.weights.clone(), inner.config, tokenizer, prompt_tokens)
        };

        let (mut logits, cache) = llm_core::model::prefill(&weights, &config, &prompt_tokens);
        // The model comes out of `Inner` for the rest of this call, the
        // same pattern `train_step_inner` uses for the trainer: holding a
        // `RefCell` borrow across `decode_step`'s `.await` below would
        // panic the moment CPU inference — which never checks `busy`, by
        // design, so it does not wait for this to finish — ran while a
        // token was mid-flight here. `gpu.model` is restored once the
        // loop ends; on an error mid-loop it stays `None`, which is fine
        // because the caller (`generate_inner`) drops the whole `gpu`
        // backend on any `Err` from here.
        let (ctx, mut model) = {
            let mut inner = self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().ok_or("no GPU")?;
            let mut model = gpu.model.take().ok_or("no GPU")?;
            model.seed_from_cpu_cache(&gpu.ctx, &cache);
            (gpu.ctx.clone(), model)
        };

        let mut guard = instruct::LengthGuard::new(request.target_words);
        let max_new_tokens = max_tokens_override.unwrap_or_else(|| guard.token_budget());
        let mut rng = llm_core::rng::Rng::seed_from_u64(sampling.seed);
        let mut produced: Vec<u32> = Vec::new();
        let mut recent: Vec<u32> = prompt_tokens.clone();
        let mut pending: Vec<u8> = Vec::new();
        let mut stop_reason = StopReason::Budget;

        for _ in 0..max_new_tokens {
            let window = recent.len().saturating_sub(sampling.repetition_window);
            let next = llm_core::generate::sample_with(&logits, sampling, &recent[window..], &mut rng);
            if next == llm_core::tokenizer::EOS {
                stop_reason = StopReason::EndOfText;
                break;
            }
            produced.push(next);
            recent.push(next);

            pending.extend_from_slice(tokenizer.piece(next));
            let piece = llm_core::generate::take_complete_chars(&mut pending);
            let keep_going = guard.observe(&piece);
            if !report(on_token, &piece, guard.words()) {
                stop_reason = StopReason::Caller;
                break;
            }
            if !keep_going {
                stop_reason = StopReason::Caller;
                break;
            }

            logits = model.decode_step(&ctx, next).await?;
        }

        {
            let mut inner = self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.model = Some(model);
            }
        }

        if !pending.is_empty() {
            let tail = String::from_utf8_lossy(&pending).into_owned();
            report(on_token, &tail, guard.words());
        }

        let text = tokenizer.decode(&produced);
        Ok(GenerationResult {
            word_count: text.split_whitespace().count() as u32,
            tokens_generated: produced.len() as u32,
            stop_reason: stop_reason_label(if guard.stopped_by_length() {
                StopReason::Caller
            } else {
                stop_reason
            })
            .to_string(),
            text,
        })
    }

    /// Suggested token budget for a prompt, so the UI can estimate how
    /// long a generation will take before starting it.
    pub fn estimated_tokens(&self, prompt: String) -> u32 {
        instruct::LengthGuard::new(instruct::parse_prompt(&prompt).target_words).token_budget() as u32
    }

    // --- Sources -----------------------------------------------------

    /// Cleans and tokenizes `raw_text`, storing (or replacing, if `id`
    /// already exists) it as a source. `is_html` should be true for text
    /// fetched from a URL, false for pasted or uploaded plain text.
    pub fn upsert_source(&self, id: String, raw_text: String, is_html: bool) -> SourceStats {
        let stats = self.0.borrow_mut().corpus.upsert(&id, &raw_text, is_html);
        SourceStats {
            char_count: stats.char_count as u32,
            byte_count: stats.byte_count as u32,
            token_count: stats.token_count as u32,
        }
    }

    pub fn remove_source(&self, id: String) -> bool {
        self.0.borrow_mut().corpus.remove(&id)
    }

    pub fn num_sources(&self) -> u32 {
        self.0.borrow().corpus.num_sources() as u32
    }

    pub fn total_tokens(&self) -> f64 {
        self.0.borrow().corpus.total_tokens() as f64
    }

    /// Per-source token counts and how many training windows have been
    /// drawn from each, as JSON — for showing which sources training has
    /// actually used, not just which sources exist. Read-only and never
    /// touches the GPU, so it needs no `busy` guard.
    pub fn corpus_source_stats(&self) -> String {
        let stats = self.0.borrow_mut().corpus.per_source_stats();
        let rows = stats
            .iter()
            .map(|s| {
                format!(
                    "{{\"id\":{:?},\"trainTokens\":{},\"heldOutTokens\":{},\"sampled\":{}}}",
                    s.id, s.train_tokens, s.held_out_tokens, s.sampled,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{rows}]")
    }

    /// Restore a source's persisted sample count after a fresh page load
    /// re-upserts it into a new corpus — see `Corpus::set_sample_count`.
    pub fn set_source_sample_count(&self, id: String, count: f64) {
        self.0.borrow_mut().corpus.set_sample_count(&id, count as u64);
    }

    /// How often a sampled training window starts exactly at a source's
    /// beginning rather than at the next window due in its rotation —
    /// see `Corpus::boundary_sample_rate`. A training setting, not a
    /// fixed constant, because how much of a source's opening is front
    /// matter (a title page, a table of contents) rather than prose
    /// varies by corpus.
    pub fn boundary_sample_rate(&self) -> f32 {
        self.0.borrow().corpus.boundary_sample_rate()
    }

    pub fn set_boundary_sample_rate(&self, rate: f32) {
        self.0.borrow_mut().corpus.set_boundary_sample_rate(rate);
    }

    /// One source's progress through its own shuffled pass over its
    /// training windows, as `{"epoch":n,"cursor":n}`, or `null` if
    /// nothing has been drawn from it yet — for persisting so a reload
    /// resumes that pass instead of restarting it (see
    /// `Corpus::window_progress`).
    pub fn window_progress(&self, id: String) -> String {
        match self.0.borrow().corpus.window_progress(&id) {
            Some((epoch, cursor)) => format!("{{\"epoch\":{epoch},\"cursor\":{cursor}}}"),
            None => "null".to_string(),
        }
    }

    /// Every source's window-pass progress that exists yet, as JSON —
    /// for writing it all back to storage in one pass rather than one
    /// round trip per source. See `Corpus::all_window_progress`.
    pub fn corpus_window_progress(&self) -> String {
        let entries = self.0.borrow().corpus.all_window_progress();
        let rows = entries
            .iter()
            .map(|(id, epoch, cursor)| format!("{{\"id\":{id:?},\"epoch\":{epoch},\"cursor\":{cursor}}}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{rows}]")
    }

    /// Restore a source's window-pass progress after a fresh page load
    /// re-upserts it into a new corpus — see `Corpus::set_window_progress`.
    pub fn set_window_progress(&self, id: String, epoch: u32, cursor: u32) {
        self.0.borrow_mut().corpus.set_window_progress(&id, epoch, cursor);
    }

    /// Learn a BPE vocabulary from the loaded sources and re-encode them
    /// with it. Returns the new vocabulary size, or 0 when there is no
    /// text yet.
    ///
    /// `max_vocab_size` is a ceiling. What is actually learned scales with
    /// how much text this visitor loaded — their vocabulary, their model,
    /// their machine; nothing here is shared with anyone else's session.
    ///
    /// Must happen before a model is created: the vocabulary size fixes
    /// the embedding table. Without it every token is one byte, which
    /// costs about four times the tokens - and therefore four times the
    /// training time - for the same text.
    pub fn learn_vocabulary(&self, max_vocab_size: u32) -> Result<u32, JsValue> {
        self.acquire()?;
        let result = self.learn_vocabulary_inner(max_vocab_size);
        self.release();
        Ok(result)
    }

    fn learn_vocabulary_inner(&self, max_vocab_size: u32) -> u32 {
        let inner = &mut *self.0.borrow_mut();
        let current = inner.corpus.tokenizer().vocab_size() as u32;
        // A trained model's weights are indexed by the vocabulary that
        // trained them: changing it would make every token id mean
        // something else.
        if inner.step > 0 || inner.pretrained {
            return current;
        }
        let Some(size) = inner.corpus.learn_vocabulary(max_vocab_size as usize) else {
            return current;
        };
        // The embedding table is one row per token, so a new vocabulary
        // is a new model. It has not been trained yet, so nothing is lost.
        inner.config.vocab_size = size;
        inner.weights = ModelWeights::init(&inner.config, inner.seed);
        // Both the uploaded generation weights and the resident training
        // state belong to the old shape.
        inner.gpu = None;
        size as u32
    }

    /// Titles of sources whose text duplicates an earlier one. Training
    /// on the same script twice weights it double.
    pub fn duplicate_sources(&self) -> Vec<String> {
        self.0.borrow().corpus.duplicate_sources()
    }

    pub fn vocab_size(&self) -> u32 {
        self.0.borrow().corpus.tokenizer().vocab_size() as u32
    }

    /// Whether a training step can run: enough source text to fill a
    /// context window, and a GPU to run it on.
    pub fn can_train(&self) -> bool {
        let inner = &mut *self.0.borrow_mut();
        if inner.gpu.is_none() {
            return false;
        }
        let context_len = inner.config.context_len;
        inner.corpus.can_sample(context_len)
    }

    /// Whether this browser gave us a device to train on at all. The
    /// page uses this to explain itself when `can_train` is false.
    pub fn has_gpu(&self) -> bool {
        self.0.borrow().gpu.is_some()
    }

    /// Runs `llm_core::qa::check_generated` against `text`, returning
    /// each note as `"[INFO] ..."`/`"[WARNING] ..."`.
    /// `target_word_count = 0` means "no target" (skips the length
    /// check).
    pub fn qa_check(&self, text: String, target_word_count: u32) -> Vec<String> {
        let target = if target_word_count == 0 { None } else { Some(target_word_count as usize) };
        llm_core::qa::check_generated(&text, target)
            .into_iter()
            .map(|note| {
                let prefix = match note.severity {
                    llm_core::qa::Severity::Info => "INFO",
                    llm_core::qa::Severity::Warning => "WARNING",
                };
                format!("[{prefix}] {}", note.message)
            })
            .collect()
    }

    // --- Training --------------------------------------------------------

    /// One training step over the current sources, on the GPU.
    ///
    /// Errors when there is no WebGPU device: this is the whole training
    /// path, not an accelerated version of another one. Returns
    /// `undefined` when there isn't enough text yet to fill a single
    /// context window.
    ///
    /// The batch is sampled here — which windows of the user's text to
    /// train on — and handed over as token ids. Everything after that
    /// (forward, loss, backward, AdamW) happens in WGSL, and the weights
    /// stay in GPU memory between steps.
    async fn train_step_inner(&self, batch_size: u32) -> Result<Option<StepReport>, JsValue> {
        let (config, train, step, batch, sources_json) = {
            let inner = &mut *self.0.borrow_mut();
            if inner.gpu.is_none() {
                return Err(js_err(
                    "training needs WebGPU, and this browser did not give us a device",
                ));
            }
            let context_len = inner.config.context_len;
            let Some(batch) =
                inner.corpus.sample_batch(batch_size as usize, context_len, &mut inner.rng)
            else {
                return Ok(None);
            };
            let sources_json = json_batch_draws(inner.corpus.last_batch_draws());
            (inner.config, inner.train, inner.step, batch, sources_json)
        };
        let lr = train.lr_at(step);

        // The resident trainer is moved out of `self` for the duration of
        // the step. Holding a `RefCell` borrow across an `await` would
        // panic the moment the page called any other method while the GPU
        // was still working — and the page does exactly that, because the
        // Stop button has to be answered mid-step.
        let (ctx, mut trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().expect("checked above");
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        if trainer.is_none() {
            let weights = self.0.borrow().weights.clone();
            match llm_gpu::GpuTrainer::new(&ctx, &config, &weights, batch.context_len) {
                Ok(fresh) => trainer = Some(fresh),
                Err(err) => return Err(js_err(err)),
            }
        }
        let mut trainer = trainer.expect("created above");
        trainer.set_dispatches_per_submit(self.0.borrow().dispatches_per_submit);
        let result = trainer
            .train_step(&ctx, &batch.inputs, &batch.targets, lr, train.weight_decay, train.grad_clip)
            .await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        let report = result.map_err(js_err)?;

        let inner = &mut *self.0.borrow_mut();
        inner.step += 1;
        // The tokens this step actually consumed, not an estimate from
        // the batch size the page happens to be set to.
        inner.tokens_seen += report.tokens as u64;
        Ok(Some(StepReport {
            loss: report.loss,
            lr: report.lr,
            grad_norm: report.grad_norm,
            tokens: report.tokens as u32,
            step: inner.step as f64,
            dispatches: report.dispatches,
            submits: report.submits,
            sources_json,
        }))
    }

    /// Run one step with a device sync after each phase and report where
    /// the milliseconds went, as JSON.
    ///
    /// `dispatches_per_submit` also sets how much work goes into each
    /// command buffer for this step, which is the direct test of whether
    /// a step is bound by per-submission cost or by arithmetic: if the
    /// total falls as this rises, it was the submissions.
    async fn profile_step_inner(&self, batch_size: u32, dispatches_per_submit: u32) -> Result<String, JsValue> {
        let (config, train, step, batch) = {
            let inner = &mut *self.0.borrow_mut();
            if inner.gpu.is_none() {
                return Err(js_err("profiling needs a GPU device"));
            }
            let context_len = inner.config.context_len;
            let Some(batch) =
                inner.corpus.sample_batch(batch_size as usize, context_len, &mut inner.rng)
            else {
                return Err(js_err("not enough text to sample a batch"));
            };
            (inner.config, inner.train, inner.step, batch)
        };
        let lr = train.lr_at(step);

        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().expect("checked above");
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let mut trainer = match trainer {
            Some(existing) => existing,
            None => {
                let weights = self.0.borrow().weights.clone();
                llm_gpu::GpuTrainer::new(&ctx, &config, &weights, batch.context_len).map_err(js_err)?
            }
        };
        let previous = dispatches_per_submit.max(1);
        trainer.set_dispatches_per_submit(previous);
        let result = trainer
            .profile_step(
                &ctx,
                &batch.inputs,
                &batch.targets,
                lr,
                train.weight_decay,
                train.grad_clip,
            )
            .await;
        {
            let inner = &mut *self.0.borrow_mut();
            inner.step += 1;
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        let report = result.map_err(js_err)?;
        let p = report.phase_ms.unwrap_or_default();
        Ok(format!(
            "{{\"dispatchesPerSubmit\":{},\"dispatches\":{},\"submits\":{},\"totalMs\":{:.1},\
             \"zeroMs\":{:.1},\"forwardMs\":{:.1},\"lossMs\":{:.1},\"backwardMs\":{:.1},\
             \"reduceMs\":{:.1},\"readbackMs\":{:.1},\"adamMs\":{:.1},\"tokens\":{}}}",
            previous,
            report.dispatches,
            report.submits,
            p.total,
            p.zero,
            p.forward,
            p.loss,
            p.backward,
            p.reduce,
            p.readback,
            p.adam,
            report.tokens,
        ))
    }

    /// One timed step that leaves no trace: `GpuTrainer::bench_step`
    /// runs at learning rate zero and restores the step counter, so a
    /// benchmark sweep costs time and nothing else.
    async fn bench_step_inner(
        &self,
        batch_size: u32,
        dispatches_per_submit: u32,
    ) -> Result<f64, JsValue> {
        let (config, batch) = {
            let inner = &mut *self.0.borrow_mut();
            if inner.gpu.is_none() {
                return Err(js_err("benchmarking needs a GPU device"));
            }
            let context_len = inner.config.context_len;
            let Some(batch) =
                inner.corpus.sample_batch(batch_size as usize, context_len, &mut inner.rng)
            else {
                return Err(js_err("not enough text to sample a batch"));
            };
            (inner.config, batch)
        };

        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().expect("checked above");
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let mut trainer = match trainer {
            Some(existing) => existing,
            None => {
                let weights = self.0.borrow().weights.clone();
                llm_gpu::GpuTrainer::new(&ctx, &config, &weights, batch.context_len)
                    .map_err(js_err)?
            }
        };
        trainer.set_dispatches_per_submit(dispatches_per_submit.max(1));
        let result = trainer.bench_step(&ctx, &batch.inputs, &batch.targets).await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        result.map_err(js_err)
    }

    async fn profile_kernels_inner(&self, reps: u32) -> Result<String, JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let Some(gpu) = inner.gpu.as_mut() else {
                return Err(js_err("profiling needs a GPU device"));
            };
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let Some(mut trainer) = trainer else {
            return Err(js_err("no training state yet — train a few steps first"));
        };
        let result = trainer.profile_kernels(&ctx, reps).await;
        let inner = &mut *self.0.borrow_mut();
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.trainer = Some(trainer);
        }
        result.map_err(js_err)
    }

    async fn validation_loss_inner(&self, batch_size: u32) -> Result<f32, JsValue> {
        self.fixed_set_loss(batch_size, true).await
    }

    /// Loss on one of the two fixed sets: held-out text, or training
    /// text drawn the same way.
    ///
    /// Both exist so their difference means something. The per-step
    /// training loss cannot be compared with held-out loss, because 40%
    /// of training windows start at a source's opening and no held-out
    /// window ever does — so the two diverge as soon as the model learns
    /// what an opening looks like, which is not overfitting and happens
    /// within a few hundred steps.
    async fn fixed_set_loss(&self, batch_size: u32, held_out: bool) -> Result<f32, JsValue> {
        let batch = {
            let inner = &mut *self.0.borrow_mut();
            if inner.gpu.as_ref().is_none_or(|gpu| gpu.trainer.is_none()) {
                return Ok(-1.0);
            }
            let context_len = inner.config.context_len;
            // The same windows every time. A fresh random draw each
            // measurement makes consecutive numbers differ by which text
            // they happened to pick, and at a few thousand tokens a
            // measurement that term is larger than the learning.
            let picked = if held_out {
                inner.corpus.validation_batch(batch_size as usize, context_len)
            } else {
                inner.corpus.training_probe_batch(batch_size as usize, context_len)
            };
            match picked {
                Some(batch) => batch,
                None => return Ok(-1.0),
            }
        };
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().expect("checked above");
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let mut trainer = trainer.expect("checked above");
        let result = trainer.eval_loss(&ctx, &batch.inputs, &batch.targets).await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        result.map_err(js_err)
    }

    async fn export_optimizer_inner(&self) -> Result<Vec<u8>, JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let Some(gpu) = inner.gpu.as_mut() else { return Ok(Vec::new()) };
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let Some(trainer) = trainer else { return Ok(Vec::new()) };
        let downloaded = trainer.download_optimizer(&ctx).await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        let (m, v, step) = downloaded.map_err(js_err)?;
        Ok(AdamState::from_parts(m, v, step).to_bytes())
    }

    async fn import_optimizer_inner(&self, bytes: &[u8]) -> Result<(), JsValue> {
        if bytes.is_empty() {
            return Ok(());
        }
        let config = self.0.borrow().config;
        let state = AdamState::from_bytes(bytes, &config).map_err(js_err)?;
        // The trainer is built lazily by the first step; build it now so
        // the restored moments have somewhere to live.
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let Some(gpu) = inner.gpu.as_mut() else {
                return Err(js_err("no GPU device to restore optimizer state onto"));
            };
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let mut trainer = match trainer {
            Some(existing) => existing,
            None => {
                let (weights, t_len) = {
                    let inner = self.0.borrow();
                    (inner.weights.clone(), inner.config.context_len)
                };
                llm_gpu::GpuTrainer::new(&ctx, &config, &weights, t_len).map_err(js_err)?
            }
        };
        let (m, v, step) = state.parts();
        let result = trainer.upload_optimizer(&ctx, m, v, step);
        let inner = &mut *self.0.borrow_mut();
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.trainer = Some(trainer);
        }
        result.map_err(js_err)
    }

    /// Bring the trained weights back from the GPU and re-upload them to
    /// the generation path.
    ///
    /// Called before anything that reads the weights on this side —
    /// generating, exporting a checkpoint — rather than after every
    /// step: the weights are megabytes, a step is milliseconds, and
    /// between steps nothing here needs to see them.
    async fn sync_from_gpu_inner(&self) -> Result<(), JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let step = inner.step;
            let Some(gpu) = inner.gpu.as_mut() else { return Ok(()) };
            if gpu.uploaded_at_step == step || gpu.trainer.is_none() {
                return Ok(());
            }
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let trainer = trainer.expect("checked above");
        let downloaded = trainer.download_weights(&ctx).await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        let weights = downloaded.map_err(js_err)?;

        let config = self.0.borrow().config;
        let model = llm_gpu::GpuModel::upload(&ctx, &weights, &config).map_err(js_err)?;
        let inner = &mut *self.0.borrow_mut();
        inner.weights = weights;
        inner.pretrained = true;
        let step = inner.step;
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.model = Some(model);
            gpu.uploaded_at_step = step;
        }
        Ok(())
    }

    /// The largest difference between this GPU forward pass and
    /// `llm-core`'s gradient-checked CPU one, over the same tokens and
    /// the same weights. Float rounding lands near `1e-3`; anything much
    /// larger means a kernel is wrong. Nothing calls this in normal use —
    /// it exists so a machine with a real GPU can check the kernels.
    pub async fn debug_compare_forward(&self, tokens: Vec<u32>) -> Result<f32, JsValue> {
        self.acquire()?;
        let result = self.debug_compare_forward_inner(tokens).await;
        self.release();
        result
    }

    async fn debug_compare_forward_inner(&self, tokens: Vec<u32>) -> Result<f32, JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let Some(gpu) = inner.gpu.as_mut() else {
                return Err(js_err("no GPU device"));
            };
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let Some(mut trainer) = trainer else {
            return Err(js_err("no training state on the GPU yet — run a step first"));
        };
        let result = trainer.debug_compare_forward(&ctx, &tokens).await;
        let inner = &mut *self.0.borrow_mut();
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.trainer = Some(trainer);
        }
        result.map_err(js_err)
    }

    /// Tell the schedule how long the whole project is planned to run —
    /// not just this sitting.
    ///
    /// The warmup and the cosine decay are both shaped by it, and both are
    /// anchored to the model's lifetime step (`start_step` stays 0): a
    /// project planned for 23,000 steps, stopped at 5,000 and resumed,
    /// picks the schedule back up at "5,000 of 23,000" — warmup already
    /// behind it, decay already under way — rather than restarting a
    /// fresh run's worth of warmup on whatever step it happens to resume
    /// at. That was this method's previous behavior
    /// (`start_step = inner.step` on every call); it's a deliberate
    /// reversal, not a bug fix — see the checkpoint's `planned_steps`
    /// field for how this now survives a reload.
    ///
    /// Idempotent: a call that doesn't change the plan is a no-op, so
    /// pressing Train again with the same Steps value neither resets the
    /// schedule nor re-triggers warmup. `steps == 0` means "leave the plan
    /// as it is" — it's the frontend's separate "train until Stop" loop
    /// sentinel, not a schedule length.
    pub fn set_project_plan(&self, steps: u32) {
        let inner = &mut *self.0.borrow_mut();
        if steps == 0 {
            return;
        }
        let steps = steps as u64;
        if inner.train.total_steps == steps && inner.train.start_step == 0 {
            return;
        }
        inner.train.start_step = 0;
        inner.train.total_steps = steps;
        // Recomputed only when the plan actually changes (a fresh model,
        // or the plan deliberately extended/shortened) — not on every
        // Train press.
        inner.train.warmup_steps = TrainConfig::warmup_for(steps);
    }

    /// Where in the current run a given step falls, 0 to 1.
    pub fn run_progress(&self) -> f32 {
        let inner = self.0.borrow();
        inner.train.progress_at(inner.step)
    }

    /// Cut the learning rate because held-out loss stopped improving,
    /// and return the multiplier now in force.
    ///
    /// This multiplies whatever the cosine schedule asks for rather than
    /// replacing it, so a run that plateaus early still finishes on the
    /// schedule's shape — just lower. The floor exists because a rate
    /// small enough stops being training at all, and a run that has cut
    /// four times has a problem no fifth cut will fix.
    pub fn decay_on_plateau(&self, factor: f32, floor: f32) -> f32 {
        let inner = &mut *self.0.borrow_mut();
        let scaled = inner.train.plateau_scale * factor.clamp(0.05, 1.0);
        inner.train.plateau_scale = scaled.max(floor.clamp(0.001, 1.0));
        inner.train.plateau_scale
    }

    /// The plateau multiplier currently in force.
    pub fn plateau_scale(&self) -> f32 {
        self.0.borrow().train.plateau_scale
    }

    /// Put it back to 1.0 — a new run, or a corpus that just grew, is
    /// not on the plateau the last one found.
    pub fn reset_plateau_scale(&self) {
        self.0.borrow_mut().train.plateau_scale = 1.0;
    }

    /// Override the fine-tuning learning rate.
    pub fn set_learning_rate(&self, lr: f32) {
        self.0.borrow_mut().train.lr = lr;
    }

    pub fn step(&self) -> f64 {
        self.0.borrow().step as f64
    }

    // --- Saving and loading ----------------------------------------------

    /// The current model as a checkpoint — tokenizer included, so it can
    /// be loaded back by `from_checkpoint` with nothing else alongside
    /// it. bf16, since this is for saving and sharing rather than for
    /// resuming a training run.
    ///
    /// Async because the trained weights live on the GPU: this is one of
    /// the two places that pulls them back across the bus.
    pub async fn export_checkpoint(&self) -> Result<Vec<u8>, JsValue> {
        // Guarded like every other operation that owns the GPU. It was
        // not, and a generation started while a save was in flight put
        // two futures through `sync_from_gpu_inner` at once — which is
        // how a save and a generate between them exhausted the wasm heap
        // and aborted the module, taking an overnight run with it.
        self.acquire()?;
        let result = self.export_checkpoint_inner().await;
        self.release();
        result
    }

    async fn export_checkpoint_inner(&self) -> Result<Vec<u8>, JsValue> {
        self.sync_from_gpu_inner().await?;
        let inner = self.0.borrow();
        // Serialized from borrowed parts, straight into the output at
        // bf16. Building a `Checkpoint` to serialize would clone the
        // weights — 153 MB for a 38M-parameter model — and the old
        // serializer then held an f32 buffer and a bf16 buffer at the
        // same time on top of that. Four copies of a model, in a heap
        // that also holds the live weights and the copy just pulled off
        // the GPU, is how an export ends in `rust_oom`; and an
        // allocation failure in wasm aborts the module rather than
        // returning an error anybody can catch.
        Ok(llm_core::checkpoint::write_checkpoint(
            &inner.config,
            &inner.weights,
            inner.corpus.tokenizer(),
            inner.step,
            inner.tokens_seen,
            inner.train.total_steps as u32,
            inner.train.plateau_scale,
            WeightDtype::Bf16,
        ))
    }

    /// Replace this model's weights, shape and tokenizer from a
    /// checkpoint. Sources are re-encoded with the new tokenizer, since
    /// token ids from the old one would mean something different.
    pub fn import_checkpoint(&self, bytes: &[u8]) -> Result<(), JsValue> {
        self.acquire()?;
        let result = self.import_checkpoint_inner(bytes);
        self.release();
        result
    }

    fn import_checkpoint_inner(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let checkpoint = Checkpoint::from_bytes(bytes).map_err(js_err)?;
        let mut inner = self.0.borrow_mut();
        inner.config = checkpoint.config;
        inner.weights = checkpoint.weights;
        inner.step = checkpoint.step;
        inner.tokens_seen = checkpoint.tokens_seen;
        inner.corpus.set_tokenizer(checkpoint.tokenizer);
        inner.pretrained = true;
        // Restore the schedule this checkpoint was carrying, the same way
        // `from_checkpoint` does — a plan of 0 means the file predates
        // planned_steps, so today's schedule (if any) is left alone rather
        // than being clobbered with "no plan".
        if checkpoint.planned_steps > 0 {
            inner.train.start_step = 0;
            inner.train.total_steps = checkpoint.planned_steps as u64;
            inner.train.warmup_steps = TrainConfig::warmup_for(checkpoint.planned_steps as u64);
            inner.train.plateau_scale = checkpoint.plateau_scale;
        }
        // Both the uploaded generation weights and any resident training
        // state belong to the model that was just replaced.
        inner.gpu = None;
        Ok(())
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
        "{{\"valid\":{},\"problem\":{:?},\"params\":{},\"vocabSize\":{},\"headDim\":{},\
         \"kvDim\":{},\"ffnDim\":{},\"trainingBytes\":{},\"inferenceBytes\":{},\
         \"memoryLimitBytes\":{},\"tileEfficiency\":{:.4},\"band\":{}}}",
        problem.is_empty(),
        problem,
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
