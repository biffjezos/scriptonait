//! Generating an answer to a prompt, on the CPU or the GPU.
//!
//! CPU generation never checks `busy` and never awaits the GPU: it has
//! to keep working while training holds the device, using whatever
//! weights are already resident. GPU generation goes through the same
//! `acquire`/`release` guard every other GPU-touching operation does.

use wasm_bindgen::prelude::*;

use llm_core::generate::{SamplingConfig, StopReason};
use llm_core::instruct;

use crate::dto::{stop_reason_label, GenerationResult};
use crate::{js_err, WasmLLM};

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

#[wasm_bindgen]
impl WasmLLM {
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

    /// Suggested token budget for a prompt, so the UI can estimate how
    /// long a generation will take before starting it.
    pub fn estimated_tokens(&self, prompt: String) -> u32 {
        instruct::LengthGuard::new(instruct::parse_prompt(&prompt).target_words).token_budget() as u32
    }
}

impl WasmLLM {
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
}
