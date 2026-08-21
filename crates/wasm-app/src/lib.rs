//! wasm-bindgen glue exposing `llm-core` (CPU: tokenizer, corpus, training,
//! generation — gradient-checked, see llm-core's tests) and `llm-gpu`
//! (WebGPU-accelerated forward pass, unverified — see its crate docs) to
//! the frontend as one `WasmLLM` class.
//!
//! Every method takes `&self` (not `&mut self`), including the ones that
//! mutate state: state lives behind `Rc<RefCell<Inner>>` so a JS object
//! only ever holds one shared handle, which is what wasm-bindgen's async
//! methods need (an async method can't hold a Rust `&mut self` borrow
//! across an `.await`). Every method that awaits something takes what it
//! needs out of the `RefCell` in a short synchronous block *before* its
//! first `.await`, so a borrow is never held across one — two overlapping
//! calls from JS (e.g. a double-clicked "Generate" button) won't panic on
//! a `RefCell` re-borrow.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::model::ModelWeights;
use llm_core::train::Trainer;

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
    gpu_ctx: Option<Rc<llm_gpu::GpuContext>>,
    gpu_model: Option<Rc<llm_gpu::GpuModel>>,
    gpu_model_step: u64,
}

#[wasm_bindgen]
pub struct SourceStats {
    pub char_count: u32,
    pub byte_count: u32,
    pub token_count: u32,
}

#[wasm_bindgen]
pub struct WasmLLM(Rc<RefCell<Inner>>);

#[wasm_bindgen]
impl WasmLLM {
    /// `local_window` should usually equal `context_len` (full attention)
    /// unless you're deliberately using sliding-window attention for a
    /// long `context_len` — see `ModelConfig` in llm-core for why.
    #[wasm_bindgen(constructor)]
    pub fn new(
        num_layers: u32,
        hidden_dim: u32,
        num_heads: u32,
        context_len: u32,
        local_window: u32,
        seed: f64,
    ) -> Result<WasmLLM, JsValue> {
        let config = ModelConfig {
            num_layers: num_layers as usize,
            hidden_dim: hidden_dim as usize,
            num_heads: num_heads as usize,
            context_len: context_len as usize,
            local_window: local_window as usize,
        };
        config.validate().map_err(js_err)?;
        let trainer = Trainer::new(config, seed as u64);
        Ok(WasmLLM(Rc::new(RefCell::new(Inner {
            trainer,
            corpus: Corpus::new(),
            gpu_ctx: None,
            gpu_model: None,
            gpu_model_step: u64::MAX,
        }))))
    }

    // --- Config / sizing ---------------------------------------------

    pub fn param_count(&self) -> f64 {
        self.0.borrow().trainer.weights.param_count() as f64
    }

    /// Rough memory estimate in bytes; see `ModelConfig::memory_bytes`.
    pub fn memory_bytes(&self, training: bool) -> f64 {
        self.0.borrow().trainer.config.memory_bytes(training) as f64
    }

    /// Whether this config's attention window fits the GPU backend's
    /// naive-kernel limit (`llm_gpu::MAX_GPU_WINDOW`). If `false`, the UI
    /// should not offer the WebGPU toggle for this config — generation
    /// will only work on the CPU path.
    pub fn gpu_supported(&self) -> bool {
        llm_gpu::supports(&self.0.borrow().trainer.config)
    }

    // --- Sources / corpus ----------------------------------------------

    /// Cleans and tokenizes `raw_text`, storing (or replacing, if `id`
    /// already exists) it as a training source. `is_html` should be true
    /// for text fetched from a URL, false for pasted/uploaded plain text.
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

    // --- Story state (heuristic, non-neural — see llm-core::screenplay) --

    pub fn story_characters(&self) -> Vec<String> {
        self.0.borrow().corpus.story_state().characters
    }

    pub fn story_locations(&self) -> Vec<String> {
        self.0.borrow().corpus.story_state().locations
    }

    pub fn story_scene_count(&self) -> u32 {
        self.0.borrow().corpus.story_state().scene_count as u32
    }

    /// A short "Characters so far: ...\nLocations so far: ...\n" block,
    /// ready to prepend to a generation prompt as a plain-text reminder
    /// of what's already established across the training sources.
    pub fn story_state_preamble(&self) -> String {
        self.0.borrow().corpus.story_state().as_prompt_preamble()
    }

    // --- Retrieval (TF-IDF over the corpus's own scenes) ------------------

    /// Up to `k` scenes similar to `query`, each formatted as
    /// `"[from: <source id> | score: <0-1>]\n<scene text>"` — meant for
    /// display (e.g. a "context used" panel), not directly for the prompt.
    pub fn retrieve_context(&self, query: String, k: u32) -> Vec<String> {
        self.0
            .borrow()
            .corpus
            .retrieve(&query, k as usize)
            .into_iter()
            .map(|c| format!("[from: {} | score: {:.2}]\n{}", c.source_id, c.score, c.text))
            .collect()
    }

    /// Same retrieval, pre-formatted as one block ready to prepend
    /// directly to a generation prompt. Empty string if nothing matched.
    pub fn retrieve_context_text(&self, query: String, k: u32) -> String {
        let chunks = self.0.borrow().corpus.retrieve(&query, k as usize);
        if chunks.is_empty() {
            return String::new();
        }
        let mut out = String::from("Similar scenes from your sources:\n\n");
        for c in &chunks {
            out.push_str(&c.text);
            out.push_str("\n\n");
        }
        out.push_str("---\n\n");
        out
    }

    // --- QA (heuristic checks on generated text) ---------------------------

    /// Runs `llm_core::qa::check_generated` against `text`, returning each
    /// note as `"[INFO] ..."`/`"[WARNING] ..."`. `target_word_count = 0`
    /// means "no target" (skips the length check).
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

    /// Samples one batch and runs a full training step. Returns the
    /// batch's mean loss, or `undefined` if there isn't enough training
    /// data yet (no sources, or fewer tokens than one context window).
    pub fn train_step(&self, batch_size: u32, lr: f32) -> Option<f32> {
        let mut inner = self.0.borrow_mut();
        let Inner { trainer, corpus, .. } = &mut *inner;
        trainer.train_step(corpus, batch_size as usize, lr)
    }

    pub fn step(&self) -> f64 {
        self.0.borrow().trainer.step as f64
    }

    // --- Generation (CPU — always available) -----------------------------

    /// `temperature <= 0.0` means greedy (deterministic) decoding.
    pub fn generate(&self, prompt: String, max_new_tokens: u32, temperature: f32, seed: f64) -> String {
        let inner = self.0.borrow();
        llm_core::generate::generate(
            &inner.trainer.weights,
            &inner.trainer.config,
            &prompt,
            max_new_tokens as usize,
            temperature,
            seed as u64,
        )
    }

    // --- Generation (WebGPU — accelerated, needs a browser with WebGPU) --

    /// Requests a WebGPU device from the browser. Call once before the
    /// first `generate_gpu`/`debug_compare_gpu_cpu`; safe to call again
    /// later (e.g. after a WebGPU context loss) to re-initialize.
    pub async fn init_gpu(&self) -> Result<(), JsValue> {
        let ctx = llm_gpu::GpuContext::new().await.map_err(js_err)?;
        let mut inner = self.0.borrow_mut();
        inner.gpu_ctx = Some(Rc::new(ctx));
        inner.gpu_model = None; // force a fresh upload against the new device
        Ok(())
    }

    /// Same as `generate`, but runs the forward pass on the GPU via
    /// WebGPU. Re-uploads the current weights to the GPU automatically
    /// whenever they've changed since the last call (tracked by training
    /// step count). Requires `init_gpu()` to have succeeded first.
    pub async fn generate_gpu(
        &self,
        prompt: String,
        max_new_tokens: u32,
        temperature: f32,
        seed: f64,
    ) -> Result<String, JsValue> {
        let (ctx, model, config) = self.ensure_gpu_model().await?;

        let mut tokens = llm_core::tokenizer::encode(&prompt);
        let mut rng = llm_core::rng::Rng::seed_from_u64(seed as u64);
        for _ in 0..max_new_tokens {
            let window: Vec<u32> = if tokens.len() > config.context_len {
                tokens[tokens.len() - config.context_len..].to_vec()
            } else {
                tokens.clone()
            };
            if window.is_empty() {
                break;
            }
            let logits = model.forward_last_logits(&ctx, &window).await.map_err(js_err)?;
            let next = llm_core::generate::sample(&logits, temperature, &mut rng);
            if next == llm_core::tokenizer::EOS {
                break;
            }
            tokens.push(next);
        }
        Ok(llm_core::tokenizer::decode(&tokens))
    }

    /// Dev/sanity-check tool: runs the *same* forward pass on both the
    /// GPU (llm-gpu, untested in this project's dev sandbox — see its
    /// crate docs) and the CPU (llm-core, gradient-checked in its test
    /// suite) backends over `prompt`, and returns the largest absolute
    /// difference between their final-token logits. This should come out
    /// tiny (float rounding only, well under 1e-2); a large value means
    /// there's a real bug in the WGSL kernels and the GPU backend
    /// shouldn't be trusted yet. Log this from the browser console after
    /// building — it's the main way to validate llm-gpu since it was
    /// written without the ability to run WebGPU at all.
    pub async fn debug_compare_gpu_cpu(&self, prompt: String) -> Result<f64, JsValue> {
        let (ctx, model, config) = self.ensure_gpu_model().await?;
        let weights = self.0.borrow().trainer.weights.clone();

        let mut tokens = llm_core::tokenizer::encode(&prompt);
        if tokens.is_empty() {
            tokens.push(0);
        }
        let window: Vec<u32> = if tokens.len() > config.context_len {
            tokens[tokens.len() - config.context_len..].to_vec()
        } else {
            tokens
        };

        let gpu_logits = model.forward_last_logits(&ctx, &window).await.map_err(js_err)?;
        let (cpu_logits_full, _) = llm_core::model::forward(&weights, &config, &window);
        let vocab = config.vocab_size();
        let cpu_last = &cpu_logits_full[(window.len() - 1) * vocab..window.len() * vocab];

        let max_diff = gpu_logits
            .iter()
            .zip(cpu_last)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        Ok(max_diff as f64)
    }

    /// Synchronous prep + the borrow_mut() that (re)uploads weights if
    /// needed, returned as owned handles so callers can `.await` GPU work
    /// afterwards without holding a `RefCell` borrow across it.
    async fn ensure_gpu_model(&self) -> Result<(Rc<llm_gpu::GpuContext>, Rc<llm_gpu::GpuModel>, ModelConfig), JsValue> {
        let mut inner = self.0.borrow_mut();
        let ctx = inner.gpu_ctx.clone().ok_or_else(|| js_err("call init_gpu() first"))?;
        if inner.gpu_model.is_none() || inner.gpu_model_step != inner.trainer.step {
            let model = llm_gpu::GpuModel::upload(&ctx, &inner.trainer.weights, &inner.trainer.config).map_err(js_err)?;
            inner.gpu_model = Some(Rc::new(model));
            inner.gpu_model_step = inner.trainer.step;
        }
        let model = inner.gpu_model.clone().unwrap();
        let config = inner.trainer.config;
        Ok((ctx, model, config))
    }

    // --- Save / load -------------------------------------------------------

    /// Raw little-endian f32 weight bytes (no header) — save this
    /// alongside the config (layers/hidden/heads/context_len/local_window)
    /// used to create this `WasmLLM`, since `import_weights` needs an
    /// identically-shaped instance to load into.
    pub fn export_weights(&self) -> Vec<u8> {
        self.0.borrow().trainer.weights.to_bytes()
    }

    /// Loads previously-exported weights, replacing this instance's
    /// current weights *and* resetting its optimizer state (Adam momentum
    /// isn't saved/restored, so mixing it with a different checkpoint's
    /// weights would be more likely to hurt than help) and step counter.
    pub fn import_weights(&self, bytes: &[u8]) -> Result<(), JsValue> {
        let mut inner = self.0.borrow_mut();
        let config = inner.trainer.config;
        let weights = ModelWeights::from_bytes(bytes, &config).map_err(js_err)?;
        let mut fresh = Trainer::new(config, 0);
        fresh.weights = weights;
        inner.trainer = fresh;
        inner.gpu_model = None;
        Ok(())
    }
}
