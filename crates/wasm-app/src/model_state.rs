//! Building a `WasmLLM` (from scratch, or from a checkpoint), reading
//! back its shape and training-plan numbers, and the checkpoint
//! export/import that round-trips the whole thing to a file.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::checkpoint::{Checkpoint, WeightDtype};
use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::model::ModelWeights;
use llm_core::rng::Rng;
use llm_core::tokenizer::Tokenizer;
use llm_core::train::TrainConfig;

use crate::dto::{describe, json_string, ModelInfo, ParsedPrompt};
use crate::{js_err, Inner, WasmLLM, DEFAULT_DISPATCHES_PER_SUBMIT};

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
            busy: Cell::new(false),
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
            warmup_variance: false,
        }))))
    }

    /// A fresh, untrained model — the "train one from scratch on my own
    /// text" path, kept for people who want it.
    /// `layer_sharing_mode`/`unique_layers`/`prelude_layers`/
    /// `coda_layers`/`core_loop_min`/`core_loop_max`: see
    /// `dto::layer_sharing_from_raw`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: u32,
        layer_sharing_mode: u32,
        unique_layers: u32,
        prelude_layers: u32,
        coda_layers: u32,
        core_loop_min: u32,
        core_loop_max: u32,
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
            layer_sharing: crate::dto::layer_sharing_from_raw(
                layer_sharing_mode,
                unique_layers,
                prelude_layers,
                coda_layers,
                core_loop_min,
                core_loop_max,
            ),
            ..Default::default()
        };
        config.validate().map_err(js_err)?;
        Ok(WasmLLM(Rc::new(RefCell::new(Inner {
            busy: Cell::new(false),
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
            warmup_variance: false,
        }))))
    }

    // --- Model info ------------------------------------------------------

    pub fn info(&self) -> ModelInfo {
        let inner = self.0.borrow();
        let config = inner.config;
        let (layer_sharing_mode, unique_layers, prelude_layers, coda_layers, core_loop_min, core_loop_max) =
            crate::dto::layer_sharing_to_raw(config.layer_sharing);
        ModelInfo {
            layers: config.num_layers as u32,
            layer_sharing_mode,
            unique_layers,
            prelude_layers,
            coda_layers,
            core_loop_min,
            core_loop_max,
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
        let warmup_variance = inner.warmup_variance;
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
                format!(
                    "{{\"kind\":{},\"label\":{},\"tokens\":{}}}",
                    json_string(kind.key()),
                    json_string(kind.label()),
                    tokens
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let schedule_kind = match train.schedule {
            llm_core::train::ScheduleKind::Cosine => "cosine",
            llm_core::train::ScheduleKind::Wsd => "wsd",
        };
        format!(
            "{{\"step\":{},\"plannedSteps\":{},\"warmupSteps\":{},\"peakLr\":{},\
             \"minLrRatio\":{},\"lrNow\":{},\"weightDecay\":{},\"gradClip\":{},\
             \"params\":{},\"layers\":{},\"hidden\":{},\"vocabSize\":{},\
             \"contextLen\":{},\"sources\":{},\"corpusTokens\":{},\
             \"trainingTokens\":{},\"validationTokens\":{},\"pretrained\":{},\
             \"plateauScale\":{},\"tokensSeen\":{},\"corpusChars\":{},\"startStep\":{},\
             \"scheduleKind\":{},\"warmupVariance\":{},\"wsdDecayStart\":{},\"mix\":[{}]}}",
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
            json_string(schedule_kind),
            warmup_variance,
            train.wsd_decay_start(),
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
        describe(&llm_core::instruct::parse_prompt(&prompt))
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

    /// Replace this model's weights, shape and tokenizer from a
    /// checkpoint. Sources are re-encoded with the new tokenizer, since
    /// token ids from the old one would mean something different.
    pub fn import_checkpoint(&self, bytes: &[u8]) -> Result<(), JsValue> {
        self.acquire()?;
        let result = self.import_checkpoint_inner(bytes);
        self.release();
        result
    }
}

impl WasmLLM {
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
