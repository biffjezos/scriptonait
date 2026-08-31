//! The training loop itself (one step's worth of GPU work per call, the
//! schedule it runs against, and the Adam moment buffers that ride
//! alongside a checkpoint) and the profiling/benchmarking variants of a
//! step used to measure this machine.

use std::rc::Rc;

use wasm_bindgen::prelude::*;

use llm_core::model::AdamState;
use llm_core::train::{ScheduleKind, TrainConfig};

use crate::dto::{json_batch_draws, StepReport};
use crate::{js_err, WasmLLM};

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
        // Train press. Which formula: see `set_warmup_strategy`.
        inner.train.warmup_steps = if inner.warmup_variance {
            TrainConfig::warmup_for_variance(steps)
        } else {
            TrainConfig::warmup_for(steps)
        };
        // A deliberate replan gets a fresh decay-start decision rather
        // than keeping one an earlier, now-superseded plan's plateau
        // pinned — unlike `extend_plan`, which grows the *same* plan and
        // must not move an already-pinned decay start. See
        // `set_decay_start`/`wsd_decay_start`.
        inner.train.decay_start_override = None;
    }

    /// For the Settings tab's "Decay start" control set to Adaptive:
    /// pin a `Wsd` run's decay window to begin at `step` instead of
    /// letting it fall out of the fixed `WSD_DECAY_FRACTION` point.
    /// Only ever called to move decay earlier than that point — see
    /// worker.js's own trigger condition — and holds until the plan is
    /// next deliberately replanned (`set_project_plan`), which clears
    /// it.
    pub fn set_decay_start(&self, step: u32) {
        let inner = &mut *self.0.borrow_mut();
        inner.train.decay_start_override = Some(step as u64);
    }

    /// For the Settings tab's "Plan length" control set to Adaptive,
    /// extends: grow the plan by `additional_steps` without resetting
    /// anything else `set_project_plan` would — most importantly, a
    /// `Wsd` run's `decay_start_override`, if one is already pinned.
    /// Extending a run already decaying should give its existing decay
    /// window more steps to reach the floor in, not redefine when that
    /// window began.
    pub fn extend_plan(&self, additional_steps: u32) {
        let inner = &mut *self.0.borrow_mut();
        inner.train.total_steps = inner.train.total_steps.saturating_add(additional_steps as u64);
    }

    /// Which formula chooses warmup length when the plan is (re)set: the
    /// existing 2%-of-plan heuristic (`false`) or RAdam's own
    /// beta2-derived length (`true`, `TrainConfig::warmup_for_variance`).
    /// Recomputes `warmup_steps` against the current plan immediately —
    /// flipping the control should be felt right away, not only the next
    /// time `set_project_plan` happens to run.
    pub fn set_warmup_strategy(&self, variance: bool) {
        let inner = &mut *self.0.borrow_mut();
        inner.warmup_variance = variance;
        inner.train.warmup_steps = if variance {
            TrainConfig::warmup_for_variance(inner.train.total_steps)
        } else {
            TrainConfig::warmup_for(inner.train.total_steps)
        };
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

    /// Which shape the schedule takes from here on — `"wsd"` selects
    /// Warmup-Stable-Decay, anything else (including an unrecognized
    /// string) selects the classic cosine shape. See
    /// `llm_core::train::ScheduleKind` for what each one means.
    pub fn set_schedule_kind(&self, kind: &str) {
        self.0.borrow_mut().train.schedule =
            if kind == "wsd" { ScheduleKind::Wsd } else { ScheduleKind::Cosine };
    }

    pub fn step(&self) -> f64 {
        self.0.borrow().step as f64
    }
}

impl WasmLLM {
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
}
