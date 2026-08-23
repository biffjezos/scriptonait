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
    model: llm_gpu::GpuModel,
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

    /// Loss on held-out text — the number that separates learning from
    /// memorizing. Returns -1 when there is not enough text to hold any
    /// out, or no training state on the GPU yet.
    pub async fn validation_loss(&self, batch_size: u32) -> Result<f32, JsValue> {
        self.acquire()?;
        let result = self.validation_loss_inner(batch_size).await;
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
            seed: 1,
            rng: Rng::seed_from_u64(1),
            corpus: Corpus::with_tokenizer(checkpoint.tokenizer),
            gpu: None,
            // Fine-tuning a trained model wants a small, flat learning
            // rate: the point is to bend it toward your text, not to
            // re-run a pretraining schedule over it.
            train: TrainConfig {
                lr: 5e-5,
                warmup_steps: 20,
                total_steps: 2000,
                min_lr_ratio: 1.0,
                ..TrainConfig::default()
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
             \"trainingTokens\":{},\"validationTokens\":{},\"pretrained\":{},\"mix\":[{}]}}",
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
            model,
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
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let request = self.build_request(&prompt, &extra_context);
        let sampling = SamplingConfig {
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
        };

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
            match self.generate_on_gpu(&request, &sampling, on_token).await {
                Ok(result) => return result,
                Err(_) => {
                    // A device that was lost or a kernel that failed:
                    // drop it and finish on the CPU rather than handing
                    // the user an error where a story should be.
                    self.0.borrow_mut().gpu = None;
                }
            }
        }
        self.generate_on_cpu(&request, &sampling, on_token)
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

    fn generate_on_cpu(
        &self,
        request: &instruct::Request,
        sampling: &SamplingConfig,
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let inner = self.0.borrow();
        let response = instruct::generate_response(
            &inner.weights,
            &inner.config,
            inner.corpus.tokenizer(),
            request,
            sampling,
            &mut |piece, words| report(on_token, piece, words),
        );
        GenerationResult::from_response(response)
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
        on_token: &js_sys::Function,
    ) -> Result<GenerationResult, String> {
        let (weights, config, tokenizer, prompt_tokens) = {
            let inner = self.0.borrow();
            let tokenizer = inner.corpus.tokenizer().clone();
            let prompt_tokens = request.to_prompt_tokens(&tokenizer);
            (inner.weights.clone(), inner.config, tokenizer, prompt_tokens)
        };

        let (mut logits, cache) = llm_core::model::prefill(&weights, &config, &prompt_tokens);
        let ctx = {
            let mut inner = self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().ok_or("no GPU")?;
            gpu.model.seed_from_cpu_cache(&gpu.ctx, &cache);
            gpu.ctx.clone()
        };

        let mut guard = instruct::LengthGuard::new(request.target_words);
        let max_new_tokens = guard.token_budget();
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

            logits = {
                let mut inner = self.0.borrow_mut();
                let gpu = inner.gpu.as_mut().ok_or("no GPU")?;
                gpu.model.decode_step(&ctx, next).await?
            };
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

    // --- Sources (fine-tuning, retrieval, story state) -------------------

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
    pub fn learn_vocabulary(&self, max_vocab_size: u32) -> u32 {
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

    pub fn story_characters(&self) -> Vec<String> {
        self.0.borrow().corpus.story_state().characters
    }

    pub fn story_locations(&self) -> Vec<String> {
        self.0.borrow().corpus.story_state().locations
    }

    pub fn story_scene_count(&self) -> u32 {
        self.0.borrow().corpus.story_state().scene_count as u32
    }

    /// A short "Characters so far: ... / Locations so far: ..." block,
    /// ready to pass as `extra_context`.
    pub fn story_state_preamble(&self) -> String {
        self.0.borrow().corpus.story_state().as_prompt_preamble()
    }

    /// Up to `k` scenes similar to `query`, each formatted as
    /// `"[from: <source id> | score: <0-1>]\n<scene text>"` — for
    /// display, not for the prompt.
    pub fn retrieve_context(&self, query: String, k: u32) -> Vec<String> {
        self.0
            .borrow()
            .corpus
            .retrieve(&query, k as usize)
            .into_iter()
            .map(|c| format!("[from: {} | score: {:.2}]\n{}", c.source_id, c.score, c.text))
            .collect()
    }

    /// The same retrieval as one block, ready to pass as
    /// `extra_context`. Empty if nothing matched.
    pub fn retrieve_context_text(&self, query: String, k: u32) -> String {
        let inner = self.0.borrow();
        inner
            .corpus
            .retrieve(&query, k as usize)
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Runs `llm_core::qa::check_generated` against `text`, returning
    /// each note as `"[INFO] ..."`/`"[WARNING] ..."`.
    /// `target_word_count = 0` means "no target" (skips the length
    /// check).
    pub fn qa_check(&self, text: String, target_word_count: u32) -> Vec<String> {
        let inner = self.0.borrow();
        let known_state = inner.corpus.story_state();
        let target = if target_word_count == 0 { None } else { Some(target_word_count as usize) };
        llm_core::qa::check_generated(&text, &known_state, target)
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
        let (config, train, step, batch) = {
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
            (inner.config, inner.train, inner.step, batch)
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
        Ok(Some(StepReport {
            loss: report.loss,
            lr: report.lr,
            grad_norm: report.grad_norm,
            tokens: report.tokens as u32,
            step: inner.step as f64,
            dispatches: report.dispatches,
            submits: report.submits,
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

        let (ctx, mut trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let gpu = inner.gpu.as_mut().expect("checked above");
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let mut trainer = match trainer.take() {
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
        let batch = {
            let inner = &mut *self.0.borrow_mut();
            if inner.gpu.as_ref().is_none_or(|gpu| gpu.trainer.is_none()) {
                return Ok(-1.0);
            }
            let context_len = inner.config.context_len;
            match inner.corpus.sample_validation_batch(batch_size as usize, context_len, &mut inner.rng) {
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
            gpu.model = model;
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

    /// Tell the schedule how long this run is planned to be.
    ///
    /// The warmup and the cosine decay are both shaped by it. Without
    /// this the defaults assume a 10,000-step run with a 200-step
    /// warmup, so a 231-step look at whether the thing learns at all
    /// spends 200 of those steps at a fraction of the learning rate and
    /// looks like it is doing nothing. Warmup is 2% of the run here,
    /// between 10 and 200 steps.
    pub fn set_planned_steps(&self, steps: u32) {
        let inner = &mut *self.0.borrow_mut();
        if steps == 0 {
            return;
        }
        let steps = steps as u64;
        inner.train.total_steps = steps;
        inner.train.warmup_steps = (steps / 50).clamp(10, 200).min(steps.max(1) / 2);
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
        self.sync_from_gpu_inner().await?;
        let inner = self.0.borrow();
        let checkpoint = Checkpoint {
            config: inner.config,
            weights: inner.weights.clone(),
            tokenizer: inner.corpus.tokenizer().clone(),
            step: inner.step,
        };
        Ok(checkpoint.to_bytes_with(WeightDtype::Bf16))
    }

    /// Replace this model's weights, shape and tokenizer from a
    /// checkpoint. Sources are re-encoded with the new tokenizer, since
    /// token ids from the old one would mean something different.
    pub fn import_checkpoint(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let checkpoint = Checkpoint::from_bytes(bytes).map_err(js_err)?;
        let mut inner = self.0.borrow_mut();
        inner.config = checkpoint.config;
        inner.weights = checkpoint.weights;
        inner.step = checkpoint.step;
        inner.corpus.set_tokenizer(checkpoint.tokenizer);
        inner.pretrained = true;
        // Both the uploaded generation weights and any resident training
        // state belong to the model that was just replaced.
        inner.gpu = None;
        Ok(())
    }
}

/// Parse a prompt without a model, so the UI can echo back what it
/// understood while the checkpoint is still downloading.
#[wasm_bindgen]
pub fn parse_prompt_standalone(prompt: String) -> ParsedPrompt {
    describe(&instruct::parse_prompt(&prompt))
}
