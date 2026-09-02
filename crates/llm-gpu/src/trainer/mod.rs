//! Training on the GPU: forward, backward, and AdamW, all in WGSL.
//!
//! The weights, the gradients and both Adam moment buffers live in GPU
//! memory for the whole run. A step uploads one batch of token ids,
//! dispatches the kernels, and reads back one small summary buffer (the
//! loss and each tensor's gradient norm). Nothing else crosses the bus,
//! and no part of the step runs on the CPU.
//!
//! The work is submitted in small command buffers rather than one per
//! sequence — see `dispatch::Chunks`. An operating system resets any GPU
//! whose single submission runs past its watchdog (about two seconds on
//! Windows), and a sequence of this model's size is far past it.
//!
//! Every kernel here mirrors a function in `llm_core::ops` /
//! `llm_core::model`, named in the shader's own header, and computes it
//! in the same layout — including the banded `probs` cache, where entry
//! `j` of row `t` is the key at absolute position `band_lo(t) + j`.
//! `io::debug_compare_forward` is what turns "mirrors" into a number: it
//! runs the same tokens through both backends from identical weights and
//! reports the largest difference in the logits.
//!
//! Split into: `layout` (the GPU-resident data shapes), `dispatch` (the
//! per-kernel dispatch helpers and the bounded-command-buffer encoder),
//! `forward`/`backward` (the two passes), `profile` (kernel and phase
//! timing), and `io` (checkpoint/optimizer marshalling plus the
//! CPU-comparison check) — all as `impl GpuTrainer` blocks alongside the
//! struct itself, defined here.

mod backward;
mod dispatch;
mod forward;
mod io;
mod layout;
mod profile;

use llm_core::config::{LayerLayout, ModelConfig};
use llm_core::model::ModelWeights;
use llm_core::ops;

use crate::buffers;
use crate::context::GpuContext;

use dispatch::Chunks;
use layout::{LayerActs, ParamSet, Scratch};

/// Operations per command buffer.
///
/// A submission is not free: the profiler measured the zeroing phase -
/// 74 dispatches of trivial work - at 376 ms, which is around 20 ms per
/// submission and nothing at all per dispatch. Chunks of four were
/// paying that cost 250 times a step. Thirty-two keeps a chunk at a few
/// tens of milliseconds of GPU work, well under the driver watchdog that
/// a whole sequence in one submission tripped, while cutting the
/// submissions by eight.
const DEFAULT_DISPATCHES_PER_SUBMIT: u32 = 32;

/// Whether this backend can train `config`.
///
/// Per-layer embeddings are the one architectural feature left out: they
/// are off by default, and a table per layer is the largest thing that
/// would have to be uploaded, scattered into and Adam-updated for a
/// feature nothing currently switches on.
pub fn supports_training(config: &ModelConfig) -> bool {
    crate::model::supports(config) && !config.use_ple
}

pub struct GpuTrainer {
    config: ModelConfig,
    weights: ParamSet,
    grads: ParamSet,
    m: ParamSet,
    v: ParamSet,
    acts: Vec<LayerActs>,
    scratch: Scratch,
    t_len: usize,
    step: i32,
    /// How many operations go into one command buffer. Small on purpose:
    /// a submission that runs longer than the OS GPU watchdog (about two
    /// seconds on Windows) gets the whole device reset out from under the
    /// page — `DXGI_ERROR_DEVICE_HUNG`, which is what a whole sequence in
    /// one submission produced on a small GPU.
    dispatches_per_submit: u32,
}

/// What one step did, mirroring `llm_core::train::StepReport`, plus what
/// it cost to run — the numbers a "why is this slow?" question needs.
pub struct GpuStepReport {
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub tokens: usize,
    /// Milliseconds per phase, filled in only by `profile_step`. A step
    /// pipelines its phases, so these are meaningful only when the
    /// profiler forces a device sync between them - which is why the
    /// normal step does not measure them.
    pub phase_ms: Option<PhaseTimings>,
    /// Compute dispatches issued for this step, and command buffers
    /// submitted. Both scale with the batch, and together they say
    /// whether a step is bound by arithmetic or by per-dispatch overhead.
    pub dispatches: u32,
    pub submits: u32,
}

/// Where a step's time goes, in milliseconds.
#[derive(Clone, Copy, Default)]
pub struct PhaseTimings {
    pub zero: f64,
    pub forward: f64,
    pub loss: f64,
    pub backward: f64,
    pub reduce: f64,
    pub readback: f64,
    pub adam: f64,
    pub total: f64,
}

impl GpuTrainer {
    /// Upload the weights and allocate every training buffer. `t_len` is
    /// the sequence length every step will use; the activation cache is
    /// sized for it once here rather than per step.
    pub fn new(
        ctx: &GpuContext,
        config: &ModelConfig,
        weights: &ModelWeights,
        t_len: usize,
    ) -> Result<Self, String> {
        if !supports_training(config) {
            return Err("this model's shape is not one the GPU training kernels handle".to_string());
        }
        if t_len < 2 || t_len > config.context_len {
            return Err(format!("sequence length {t_len} is outside 2..={}", config.context_len));
        }
        let h = config.hidden_dim;
        let kv = config.kv_dim();
        let ffn = config.ffn_dim();
        let vocab = config.vocab_size();
        let band = ops::band_width(t_len, config.effective_window());
        let dev = &ctx.device;
        let s = |label: &str, len: usize| buffers::storage_f32(dev, label, len, false);

        let acts = (0..config.num_layers)
            .map(|_| LayerActs {
                h_in: s("h_in", t_len * h),
                normed1: s("normed1", t_len * h),
                inv_rms1: s("inv_rms1", t_len),
                q: s("q", t_len * h),
                k: s("k", t_len * kv),
                v: s("v", t_len * kv),
                probs: s("probs", config.num_heads * t_len * band),
                concat: s("concat", t_len * h),
                h_after_attn: s("h_after_attn", t_len * h),
                normed2: s("normed2", t_len * h),
                inv_rms2: s("inv_rms2", t_len),
                gate: s("gate", t_len * ffn),
                up: s("up", t_len * ffn),
            })
            .collect();

        let stats_len = 1 + ParamSet::shapes(config).len();
        let scratch = Scratch {
            tokens: buffers::upload_u32(dev, "tokens", &vec![0u32; t_len]),
            targets: buffers::upload_u32(dev, "targets", &vec![0u32; t_len]),
            row_ids: buffers::upload_u32(dev, "scatter-row-ids", &vec![0u32; t_len]),
            row_offsets: buffers::upload_u32(dev, "scatter-offsets", &vec![0u32; t_len + 1]),
            row_positions: buffers::upload_u32(dev, "scatter-positions", &vec![0u32; t_len]),
            // The residual stream is the source of three
            // `copy_buffer_to_buffer`s per layer (into the activation
            // cache), so it needs COPY_SRC - which is what the `readable`
            // flag adds.
            hidden: buffers::storage_f32(dev, "hidden", t_len * h, true),
            tmp_h: s("tmp_h", t_len * h),
            h_final: s("h_final", t_len * h),
            final_normed: s("final_normed", t_len * h),
            final_inv_rms: s("final_inv_rms", t_len),
            logits: buffers::storage_f32(dev, "logits", t_len * vocab, true),
            d_logits: s("d_logits", t_len * vocab),
            loss_rows: s("loss_rows", t_len),
            d_hidden: s("d_hidden", t_len * h),
            d_a: s("d_a", t_len * h),
            d_b: s("d_b", t_len * h),
            d_c: s("d_c", t_len * h),
            d_q: s("d_q", t_len * h),
            d_k: s("d_k", t_len * kv),
            d_v: s("d_v", t_len * kv),
            d_score: s("d_score", config.num_heads * t_len * band),
            act: s("act", t_len * ffn),
            d_act: s("d_act", t_len * ffn),
            d_gate: s("d_gate", t_len * ffn),
            d_up: s("d_up", t_len * ffn),
            partials: buffers::storage_f32(dev, "partials", dispatch::REDUCE_GROUPS, false),
            stats: buffers::storage_f32(dev, "stats", stats_len, true),
        };

        Ok(Self {
            config: *config,
            weights: ParamSet::from_weights(ctx, weights),
            grads: ParamSet::zeros(ctx, config, false),
            // `download_optimizer` reads these back (for the checkpoint's
            // saved momentum) via `read_f32_concat`, which needs COPY_SRC
            // on every buffer it copies from — `grads` above is never
            // downloaded, so it stays write-only.
            m: ParamSet::zeros(ctx, config, true),
            v: ParamSet::zeros(ctx, config, true),
            acts,
            scratch,
            t_len,
            step: 0,
            dispatches_per_submit: DEFAULT_DISPATCHES_PER_SUBMIT,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Change how many operations share a command buffer. Lower is safer
    /// against the driver watchdog on a slow device, higher trims
    /// submission overhead on a fast one.
    pub fn set_dispatches_per_submit(&mut self, n: u32) {
        self.dispatches_per_submit = n.max(1);
    }

    /// Bytes this trainer holds in GPU memory: four copies of every
    /// parameter (weights, gradients, both Adam moments), the per-layer
    /// activation cache, and the scratch. Logged at startup, because on
    /// an integrated GPU the difference between "fits" and "spills to
    /// system memory" is the difference between fast and not.
    pub fn allocated_bytes(&self) -> u64 {
        let params: usize = self.weights.slots.iter().map(|s| s.len).sum();
        let c = &self.config;
        let t = self.t_len;
        let band = ops::band_width(t, c.effective_window());
        let per_layer = 6 * t * c.hidden_dim
            + 2 * t
            + 2 * t * c.kv_dim()
            + c.num_heads * t * band
            + 2 * t * c.ffn_dim();
        let scratch = 8 * t * c.hidden_dim
            + 2 * t * c.vocab_size()
            + 2 * t * c.kv_dim()
            + c.num_heads * t * band
            + 4 * t * c.ffn_dim()
            + 3 * t;
        ((4 * params + c.num_layers * per_layer + scratch) * std::mem::size_of::<f32>()) as u64
    }

    /// One training step over a batch: `inputs`/`targets` are
    /// `batch_size * t_len` token ids, laid out one sequence after
    /// another — exactly `llm_core::corpus::Batch`'s layout.
    ///
    /// Each sequence is its own submission, but they all accumulate into
    /// the same gradient buffer — zeroed once here, at the top of the
    /// step — so the batch costs no extra memory and no separate
    /// summing pass, exactly as `model::backward_into` does on the CPU.
    /// `core_loops`: which of `unique_layer_count()`'s shared core
    /// applies this step (see `ModelConfig::layer_layout`) — sampled by
    /// the caller once per step, the same depth for every sequence in
    /// this batch, and ignored unless `layer_sharing` is `RecurrentCore`.
    #[allow(clippy::too_many_arguments)]
    pub async fn train_step(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
        lr: f32,
        weight_decay: f32,
        grad_clip: f32,
        core_loops: Option<usize>,
    ) -> Result<GpuStepReport, String> {
        let t = self.t_len;
        if inputs.len() != targets.len() || inputs.is_empty() || inputs.len() % t != 0 {
            return Err(format!(
                "batch must be a whole number of {t}-token sequences, got {} inputs and {} targets",
                inputs.len(),
                targets.len()
            ));
        }
        let batch_size = inputs.len() / t;
        ctx.params.reset();
        ctx.dispatch_count.set(0);
        let mut chunks = Chunks::new(ctx, self.dispatches_per_submit);
        let layout = self.config.layer_layout(core_loops);

        // Zero the gradient accumulator and the stats slots once per
        // step; every backward kernel that writes a gradient accumulates.
        for slot in &self.grads.slots {
            self.dispatch_zero(&mut chunks, ctx, &slot.buffer, slot.len);
        }
        let stats_len = 1 + self.grads.slots.len();
        self.dispatch_zero(&mut chunks, ctx, &self.scratch.stats, stats_len);
        chunks.flush();

        for b in 0..batch_size {
            let seq = &inputs[b * t..(b + 1) * t];
            let tgt = &targets[b * t..(b + 1) * t];
            buffers::write_u32(&ctx.queue, &self.scratch.tokens, seq);
            buffers::write_u32(&ctx.queue, &self.scratch.targets, tgt);
            let groups = self.upload_scatter_index(ctx, seq);

            self.encode_forward(&mut chunks, ctx, &layout);
            self.encode_loss(&mut chunks, ctx);
            self.encode_backward(&mut chunks, ctx, groups, &layout);
            // A sequence's work must be complete before the next one
            // overwrites the shared token buffer and activation cache.
            chunks.flush();
        }

        // Gradient norm: each tensor's sum of squares into its own stats
        // slot, so the host adds a few dozen numbers instead of reading
        // back several megabytes of gradient.
        for (i, slot) in self.grads.slots.iter().enumerate() {
            self.dispatch_reduce(&mut chunks, ctx, &slot.buffer, slot.len, 1 + i, true);
        }
        chunks.flush();
        let stats = buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.stats, stats_len).await?;

        let inv_batch = 1.0 / batch_size as f32;
        // The kernel writes one unnormalized loss per row, and every
        // sequence in the batch reduced into the same slot: dividing by
        // both gives the same number `ops::cross_entropy` returns,
        // averaged over the batch.
        let loss = stats[0] / (batch_size * t) as f32;
        // Slots 1.. already hold each tensor's sum of squares; the global
        // norm is the root of their total.
        let sum_sq: f64 = stats[1..].iter().map(|&x| x as f64).sum();
        let norm_unscaled = sum_sq.sqrt() as f32;
        // The CPU trainer averages the gradient over the batch and then
        // clips; the norm reported here is the norm of that averaged
        // gradient, and both the averaging and the clip factor ride into
        // the Adam kernel as one multiplier.
        let grad_norm = norm_unscaled * inv_batch;
        let clip = if grad_norm > grad_clip && grad_norm.is_finite() && grad_norm > 0.0 {
            grad_clip / grad_norm
        } else {
            1.0
        };

        self.step += 1;
        self.encode_adam(&mut chunks, ctx, lr, weight_decay, inv_batch * clip);
        chunks.flush();

        Ok(GpuStepReport {
            loss,
            lr,
            grad_norm,
            phase_ms: None,
            tokens: batch_size * t,
            dispatches: ctx.dispatch_count.get(),
            submits: chunks.submits,
        })
    }

    /// One step's worth of work, timed, with nothing learned from it.
    ///
    /// The same dispatch sequence `train_step` runs, at learning rate and
    /// weight decay zero, so the weights the caller had are the weights
    /// it keeps and the step counter does not move. The Adam moments do
    /// see these gradients, which is a warm start for the estimates and
    /// nothing more.
    ///
    /// This exists so the page can measure this machine — how much work
    /// per command buffer, how many sequences per batch it wants —
    /// instead of a constant chosen on somebody else's GPU.
    pub async fn bench_step(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
    ) -> Result<f64, String> {
        let before = self.step;
        let start = web_time::Instant::now();
        // grad_clip infinite: the clip factor stays 1.0, so the bench
        // runs the same arithmetic whatever the gradients happen to be.
        // core_loops: None, i.e. this model's maximum depth — the
        // worst-case, deterministic cost a machine profile should be
        // sized against, not a random sample.
        self.train_step(ctx, inputs, targets, 0.0, 0.0, f32::INFINITY, None).await?;
        // The AdamW pass is submitted but not waited for by `train_step`;
        // a measurement has to include it.
        self.sync(ctx).await?;
        self.step = before;
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }
}
