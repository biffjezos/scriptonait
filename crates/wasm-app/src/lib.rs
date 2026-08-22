//! The browser's view of the model: one `WasmLLM` class over `llm-core`.
//!
//! What the browser does now is *inference*. The model that ships with
//! the page was trained natively in CI (see `.github/workflows/pretrain.yml`
//! and `crates/llm-train`) and arrives as a single checkpoint file
//! carrying its own tokenizer, so the page's first job is to load it
//! rather than to start a training loop and ask the user to wait.
//!
//! Fine-tuning on your own text is still here — `upsert_source` plus
//! `train_step` — but it's the secondary path, it's opt-in, and the
//! caller controls its pace (see `worker.js`, which yields between steps
//! against a time budget so a training run can't take over the machine).
//!
//! Generation runs on the GPU when the browser has WebGPU, and on the
//! CPU when it doesn't. That is a fact about the machine, not a setting:
//! there is no toggle, the page just says which one it got.
//!
//! The two paths share their prompt handling, their sampling and their
//! stopping rule (`llm_core::instruct::LengthGuard`) — what differs is
//! only where the arithmetic happens. Both prefill on the CPU, because
//! that forward pass is the gradient-checked one; the GPU takes over
//! for the per-token decode, which is where the time goes.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::checkpoint::{Checkpoint, WeightDtype};
use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::generate::{SamplingConfig, StopReason};
use llm_core::instruct;
use llm_core::tokenizer::Tokenizer;
use llm_core::train::{TrainConfig, Trainer};

/// A ready WebGPU device plus this model's weights uploaded to it.
struct GpuBackend {
    ctx: Rc<llm_gpu::GpuContext>,
    model: llm_gpu::GpuModel,
    /// Training step the uploaded weights came from, so fine-tuning
    /// invalidates them instead of generating from stale ones.
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

struct Inner {
    trainer: Trainer,
    corpus: Corpus,
    gpu: Option<GpuBackend>,
    train: TrainConfig,
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

/// One fine-tuning step's numbers.
#[wasm_bindgen]
pub struct StepReport {
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens: u32,
    pub step: f64,
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

#[wasm_bindgen]
impl WasmLLM {
    /// Load the model the site ships with.
    ///
    /// The checkpoint carries its shape, its weights, and the tokenizer
    /// its token ids belong to, so there is nothing for the caller to
    /// get wrong and nothing to configure.
    pub fn from_checkpoint(bytes: &[u8]) -> Result<WasmLLM, JsValue> {
        let checkpoint = Checkpoint::from_bytes(bytes).map_err(js_err)?;
        let trainer = Trainer::resume(checkpoint.config, checkpoint.weights, checkpoint.step, 1);
        Ok(WasmLLM(Rc::new(RefCell::new(Inner {
            trainer,
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
            trainer: Trainer::new(config, seed as u64),
            corpus: Corpus::with_tokenizer(tokenizer),
            gpu: None,
            train: TrainConfig::default(),
            pretrained: false,
        }))))
    }

    // --- Model info ------------------------------------------------------

    pub fn info(&self) -> ModelInfo {
        let inner = self.0.borrow();
        let config = inner.trainer.config;
        ModelInfo {
            layers: config.num_layers as u32,
            hidden: config.hidden_dim as u32,
            heads: config.num_heads as u32,
            kv_heads: config.num_kv_heads as u32,
            context_len: config.context_len as u32,
            window: config.effective_window() as u32,
            vocab_size: config.vocab_size as u32,
            params: config.param_count() as f64,
            step: inner.trainer.step as f64,
            pretrained: inner.pretrained,
        }
    }

    /// Rough memory estimate in bytes; see `ModelConfig::memory_bytes`.
    pub fn memory_bytes(&self, training: bool) -> f64 {
        self.0.borrow().trainer.config.memory_bytes(training) as f64
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
    /// isn't one. Failure is normal and is not an error: browsers
    /// without WebGPU exist, and the CPU path is always there.
    pub async fn init_gpu(&self) -> Result<String, JsValue> {
        let (config, weights) = {
            let inner = self.0.borrow();
            (inner.trainer.config, inner.trainer.weights.clone())
        };
        if !llm_gpu::supports(&config) {
            return Err(js_err("this model's shape is past what the GPU kernels handle"));
        }
        let ctx = Rc::new(llm_gpu::GpuContext::new().await.map_err(js_err)?);
        let model = llm_gpu::GpuModel::upload(&ctx, &weights, &config).map_err(js_err)?;
        let summary = ctx.adapter_summary.clone();
        let is_software = ctx.is_software;
        let step = self.0.borrow().trainer.step;
        self.0.borrow_mut().gpu = Some(GpuBackend {
            ctx,
            model,
            uploaded_at_step: step,
            summary: summary.clone(),
            is_software,
        });
        Ok(summary)
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
        repetition_penalty: f32,
        seed: f64,
        on_token: &js_sys::Function,
    ) -> GenerationResult {
        let request = self.build_request(&prompt, &extra_context);
        let sampling = SamplingConfig {
            temperature,
            top_k: top_k as usize,
            top_p,
            repetition_penalty,
            seed: seed as u64,
            ..SamplingConfig::default()
        };

        // The GPU path needs the weights to match what was uploaded; a
        // fine-tuning run since then invalidates them, and re-uploading
        // mid-generation would be worse than just using the CPU.
        let gpu_is_current = {
            let inner = self.0.borrow();
            inner.gpu.as_ref().is_some_and(|g| g.uploaded_at_step == inner.trainer.step)
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
            &inner.trainer.weights,
            &inner.trainer.config,
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
            (inner.trainer.weights.clone(), inner.trainer.config, tokenizer, prompt_tokens)
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

    /// Whether there's enough source text to fine-tune on at all.
    pub fn can_train(&self) -> bool {
        let inner = &mut *self.0.borrow_mut();
        let context_len = inner.trainer.config.context_len;
        inner.corpus.can_sample(context_len)
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

    // --- Fine-tuning -----------------------------------------------------

    /// One training step over the current sources. Returns `undefined`
    /// if there isn't enough text yet to fill a single context window.
    pub fn train_step(&self, batch_size: u32) -> Option<StepReport> {
        let inner = &mut *self.0.borrow_mut();
        let train = inner.train;
        let report = inner.trainer.train_step_with(&mut inner.corpus, batch_size as usize, &train)?;
        Some(StepReport {
            loss: report.loss,
            lr: report.lr,
            grad_norm: report.grad_norm,
            tokens: report.tokens as u32,
            step: inner.trainer.step as f64,
        })
    }

    /// Override the fine-tuning learning rate.
    pub fn set_learning_rate(&self, lr: f32) {
        self.0.borrow_mut().train.lr = lr;
    }

    pub fn step(&self) -> f64 {
        self.0.borrow().trainer.step as f64
    }

    // --- Saving and loading ----------------------------------------------

    /// The current model as a checkpoint — tokenizer included, so it can
    /// be loaded back by `from_checkpoint` with nothing else alongside
    /// it. bf16, since this is for saving and sharing rather than for
    /// resuming a pretraining run.
    pub fn export_checkpoint(&self) -> Vec<u8> {
        let inner = self.0.borrow();
        Checkpoint {
            config: inner.trainer.config,
            weights: inner.trainer.weights.clone(),
            tokenizer: inner.corpus.tokenizer().clone(),
            step: inner.trainer.step,
        }
        .to_bytes_with(WeightDtype::Bf16)
    }

    /// Replace this model's weights, shape and tokenizer from a
    /// checkpoint. Sources are re-encoded with the new tokenizer, since
    /// token ids from the old one would mean something different.
    pub fn import_checkpoint(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let checkpoint = Checkpoint::from_bytes(bytes).map_err(js_err)?;
        let mut inner = self.0.borrow_mut();
        inner.trainer = Trainer::resume(checkpoint.config, checkpoint.weights, checkpoint.step, 1);
        inner.corpus.set_tokenizer(checkpoint.tokenizer);
        inner.pretrained = true;
        Ok(())
    }
}

/// Parse a prompt without a model, so the UI can echo back what it
/// understood while the checkpoint is still downloading.
#[wasm_bindgen]
pub fn parse_prompt_standalone(prompt: String) -> ParsedPrompt {
    describe(&instruct::parse_prompt(&prompt))
}
