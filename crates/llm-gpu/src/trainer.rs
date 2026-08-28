//! Training on the GPU: forward, backward, and AdamW, all in WGSL.
//!
//! The weights, the gradients and both Adam moment buffers live in GPU
//! memory for the whole run. A step uploads one batch of token ids,
//! dispatches the kernels, and reads back one small summary buffer (the
//! loss and each tensor's gradient norm). Nothing else crosses the bus,
//! and no part of the step runs on the CPU.
//!
//! The work is submitted in small command buffers rather than one per
//! sequence — see `Chunks`. An operating system resets any GPU whose
//! single submission runs past its watchdog (about two seconds on
//! Windows), and a sequence of this model's size is far past it.
//!
//! Every kernel here mirrors a function in `llm_core::ops` /
//! `llm_core::model`, named in the shader's own header, and computes it
//! in the same layout — including the banded `probs` cache, where entry
//! `j` of row `t` is the key at absolute position `band_lo(t) + j`.
//! `debug_compare_forward` is what turns "mirrors" into a number: it
//! runs the same tokens through both backends from identical weights and
//! reports the largest difference in the logits.

use llm_core::config::ModelConfig;
use llm_core::model::ModelWeights;
use llm_core::ops;

use web_time::Instant;

use crate::buffers;
use crate::context::GpuContext;
use crate::model::{ceil_div, dispatch, dispatch_add_inplace, dispatch_linear, dispatch_swiglu, P4, P8};

/// Must match `llm_core::model`'s RMS_EPS, which is private there.
const RMS_EPS: f32 = 1e-6;

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

/// Workgroups a reduction splits its tensor across. Enough to keep a
/// small GPU busy on a multi-million element tensor, small enough that
/// summing the partials afterwards is one cheap dispatch.
const REDUCE_GROUPS: usize = 64;

/// Per-layer tensors, in the order `ParamSet` stores them.
const T_ATTN_GAIN: usize = 0;
const T_WQ: usize = 1;
const T_WK: usize = 2;
const T_WV: usize = 3;
const T_WO: usize = 4;
const T_MLP_GAIN: usize = 5;
const T_W_GATE: usize = 6;
const T_W_UP: usize = 7;
const T_W_DOWN: usize = 8;
const TENSORS_PER_LAYER: usize = 9;

/// Whether this backend can train `config`.
///
/// Per-layer embeddings are the one architectural feature left out: they
/// are off by default, and a table per layer is the largest thing that
/// would have to be uploaded, scattered into and Adam-updated for a
/// feature nothing currently switches on.
pub fn supports_training(config: &ModelConfig) -> bool {
    crate::model::supports(config) && !config.use_ple
}

struct TensorSlot {
    buffer: wgpu::Buffer,
    len: usize,
    /// Weight decay applies to matrices and embedding tables, not to the
    /// RMSNorm gains — matching `ModelWeights::decay_flags`.
    decay: bool,
}

/// One copy of every parameter-shaped tensor: the weights, the gradient
/// accumulator, or an Adam moment buffer.
struct ParamSet {
    slots: Vec<TensorSlot>,
}

impl ParamSet {
    /// The fixed order: embed, then each layer's nine tensors, then the
    /// final norm gain. `ModelWeights::tensors()` uses the same order,
    /// which is what makes a downloaded set restore field-for-field.
    fn shapes(config: &ModelConfig) -> Vec<(&'static str, usize, bool)> {
        let (h, ffn, kv, vocab) = (config.hidden_dim, config.ffn_dim(), config.kv_dim(), config.vocab_size());
        let mut out = vec![("embed", vocab * h, true)];
        for _ in 0..config.num_layers {
            out.extend_from_slice(&[
                ("attn_norm_gain", h, false),
                ("wq", h * h, true),
                ("wk", kv * h, true),
                ("wv", kv * h, true),
                ("wo", h * h, true),
                ("mlp_norm_gain", h, false),
                ("w_gate", ffn * h, true),
                ("w_up", ffn * h, true),
                ("w_down", h * ffn, true),
            ]);
        }
        out.push(("final_norm_gain", h, false));
        out
    }

    fn zeros(ctx: &GpuContext, config: &ModelConfig, readable: bool) -> Self {
        let slots = Self::shapes(config)
            .into_iter()
            .map(|(label, len, decay)| TensorSlot {
                buffer: buffers::storage_f32(&ctx.device, label, len, readable),
                len,
                decay,
            })
            .collect();
        Self { slots }
    }

    fn from_weights(ctx: &GpuContext, weights: &ModelWeights) -> Self {
        let mut slots = Vec::new();
        let mut push = |label: &str, data: &[f32], decay: bool| {
            slots.push(TensorSlot {
                buffer: buffers::upload_f32(&ctx.device, label, data, wgpu::BufferUsages::COPY_SRC),
                len: data.len(),
                decay,
            });
        };
        push("embed", &weights.embed, true);
        for layer in &weights.layers {
            push("attn_norm_gain", &layer.attn_norm_gain, false);
            push("wq", &layer.wq, true);
            push("wk", &layer.wk, true);
            push("wv", &layer.wv, true);
            push("wo", &layer.wo, true);
            push("mlp_norm_gain", &layer.mlp_norm_gain, false);
            push("w_gate", &layer.w_gate, true);
            push("w_up", &layer.w_up, true);
            push("w_down", &layer.w_down, true);
        }
        push("final_norm_gain", &weights.final_norm_gain, false);
        Self { slots }
    }

    fn embed(&self) -> &wgpu::Buffer {
        &self.slots[0].buffer
    }

    fn layer(&self, layer: usize, which: usize) -> &wgpu::Buffer {
        &self.slots[1 + layer * TENSORS_PER_LAYER + which].buffer
    }

    fn final_gain(&self, config: &ModelConfig) -> &wgpu::Buffer {
        &self.slots[1 + config.num_layers * TENSORS_PER_LAYER].buffer
    }
}

/// Everything the backward pass reads back out of the forward pass, for
/// one layer of one sequence. Same contents as `llm_core::model`'s
/// `LayerCache`; the SwiGLU activation is recomputed rather than stored,
/// exactly as the CPU reference does.
struct LayerActs {
    h_in: wgpu::Buffer,        // residual stream entering the layer
    normed1: wgpu::Buffer,
    inv_rms1: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    probs: wgpu::Buffer,       // [heads, t_len, band]
    concat: wgpu::Buffer,
    h_after_attn: wgpu::Buffer,
    normed2: wgpu::Buffer,
    inv_rms2: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
}

/// Scratch reused by every layer and every sequence in the batch.
struct Scratch {
    tokens: wgpu::Buffer,
    targets: wgpu::Buffer,
    /// The per-sequence index the embedding scatter walks: the distinct
    /// token ids, where each one's positions start, and the positions
    /// themselves. Built on the host per sequence — it is `t_len` numbers
    /// of bookkeeping, and it turns the scatter from the most expensive
    /// dispatch of the step into one of the cheapest.
    row_ids: wgpu::Buffer,
    row_offsets: wgpu::Buffer,
    row_positions: wgpu::Buffer,
    hidden: wgpu::Buffer,
    tmp_h: wgpu::Buffer,
    h_final: wgpu::Buffer,
    final_normed: wgpu::Buffer,
    final_inv_rms: wgpu::Buffer,
    logits: wgpu::Buffer,
    d_logits: wgpu::Buffer,
    loss_rows: wgpu::Buffer,
    d_hidden: wgpu::Buffer,
    d_a: wgpu::Buffer,
    d_b: wgpu::Buffer,
    d_c: wgpu::Buffer,
    d_q: wgpu::Buffer,
    d_k: wgpu::Buffer,
    d_v: wgpu::Buffer,
    d_score: wgpu::Buffer,
    act: wgpu::Buffer,
    d_act: wgpu::Buffer,
    d_gate: wgpu::Buffer,
    d_up: wgpu::Buffer,
    /// Per-workgroup partial sums, consumed by reduce_finish.
    partials: wgpu::Buffer,
    /// Slot 0 is the summed loss; slot 1+i is tensor i's summed square.
    /// One small buffer, one readback per step - the only host-device
    /// synchronization a step has.
    stats: wgpu::Buffer,
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
            partials: buffers::storage_f32(dev, "partials", REDUCE_GROUPS, false),
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

    pub fn sequence_len(&self) -> usize {
        self.t_len
    }

    pub fn steps_done(&self) -> i32 {
        self.step
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
    pub async fn train_step(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
        lr: f32,
        weight_decay: f32,
        grad_clip: f32,
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

            self.encode_forward(&mut chunks, ctx);
            self.encode_loss(&mut chunks, ctx);
            self.encode_backward(&mut chunks, ctx, groups);
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
        let start = Instant::now();
        // grad_clip infinite: the clip factor stays 1.0, so the bench
        // runs the same arithmetic whatever the gradients happen to be.
        self.train_step(ctx, inputs, targets, 0.0, 0.0, f32::INFINITY).await?;
        // The AdamW pass is submitted but not waited for by `train_step`;
        // a measurement has to include it.
        self.sync(ctx).await?;
        self.step = before;
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    /// Times each kernel on its own, at the shapes a step actually uses.
    ///
    /// The phase profile says forward and backward are the whole step;
    /// this says which kernels inside them. Each case runs `reps` times
    /// back to back with one sync at the end, so per-dispatch host cost
    /// is amortized and what is left is the kernel itself. Multiplied by
    /// how many times a step runs it, that is where a step's time lives.
    pub async fn profile_kernels(&mut self, ctx: &GpuContext, reps: u32) -> Result<String, String> {
        let c = self.config;
        let t = self.t_len;
        let (h, kv, ffn, vocab) = (c.hidden_dim, c.kv_dim(), c.ffn_dim(), c.vocab_size());
        let band = ops::band_width(t, c.effective_window());
        let layers = c.num_layers;
        let reps = reps.max(1);
        let tensors = self.grads.slots.len();

        // (label, how many of these a single sequence's step runs)
        let cases: [(&str, usize); 14] = [
            ("linear q/o (h x h)", 2 * layers),
            ("linear k/v (h x kv)", 2 * layers),
            ("linear gate/up (h x ffn)", 2 * layers),
            ("linear down (ffn x h)", layers),
            ("linear head (h x vocab)", 1),
            ("linear_bwd_dw (h x ffn)", 2 * layers),
            ("linear_bwd_dx (h x ffn)", 2 * layers),
            ("attention_fwd", layers),
            ("attention_bwd (3 kernels)", layers),
            ("rmsnorm", 2 * layers),
            ("rmsnorm_bwd (dx + dgain)", 4 * layers),
            ("rope", 2 * layers),
            ("swiglu", 2 * layers),
            ("adam (all tensors)", 1),
        ];

        let mut rows = Vec::with_capacity(cases.len());
        for (case, (label, per_step)) in cases.iter().enumerate() {
            // One warm run, uncounted: the first dispatch of a pipeline
            // pays for whatever the driver does lazily.
            for round in 0..2 {
                ctx.params.reset();
                let start = Instant::now();
                let mut chunks = Chunks::new(ctx, 64);
                let times = if round == 0 { 1 } else { reps };
                for _ in 0..times {
                    self.encode_profile_case(&mut chunks, ctx, case, band);
                }
                chunks.flush();
                self.sync(ctx).await?;
                if round == 1 {
                    let each = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
                    rows.push(format!(
                        "{{\"kernel\":{:?},\"msEach\":{:.3},\"perStep\":{},\"msPerStep\":{:.1}}}",
                        label,
                        each,
                        per_step,
                        each * *per_step as f64
                    ));
                }
            }
        }
        let _ = (kv, vocab, tensors);
        Ok(format!("[{}]", rows.join(",")))
    }

    /// One kernel of `profile_kernels`, encoded against real buffers of
    /// the right shape. Nothing here is read back: the numbers it writes
    /// are overwritten by the next real step.
    fn encode_profile_case(&self, chunks: &mut Chunks, ctx: &GpuContext, case: usize, band: usize) {
        let c = &self.config;
        let t = self.t_len;
        let (h, kv, ffn, vocab) = (c.hidden_dim, c.kv_dim(), c.ffn_dim(), c.vocab_size());
        let acts = &self.acts[0];
        let s = &self.scratch;
        match case {
            0 => dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(0, T_WQ), &acts.q, t, h, h),
            1 => dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(0, T_WK), &acts.k, t, h, kv),
            2 => dispatch_linear(chunks.enc(), ctx, &acts.normed2, self.weights.layer(0, T_W_GATE), &acts.gate, t, h, ffn),
            3 => dispatch_linear(chunks.enc(), ctx, &s.act, self.weights.layer(0, T_W_DOWN), &s.tmp_h, t, ffn, h),
            4 => dispatch_linear(chunks.enc(), ctx, &s.final_normed, self.weights.embed(), &s.logits, t, h, vocab),
            5 => self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_gate, &acts.normed2, self.grads.layer(0, T_W_GATE), t, h, ffn),
            6 => self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_gate, self.weights.layer(0, T_W_GATE), &s.d_a, t, h, ffn),
            7 => self.dispatch_attention_fwd(chunks, ctx, acts, band),
            8 => self.dispatch_attention_bwd(chunks, ctx, acts, band),
            9 => self.dispatch_rmsnorm(chunks, ctx, &s.hidden, self.weights.layer(0, T_ATTN_GAIN), &acts.normed1, &acts.inv_rms1, h),
            10 => {
                self.dispatch_rmsnorm_bwd_dx(chunks, ctx, &s.d_a, &acts.h_in, self.weights.layer(0, T_ATTN_GAIN), &acts.inv_rms1, &s.d_c, h);
                self.dispatch_rmsnorm_bwd_dgain(chunks, ctx, &s.d_a, &acts.h_in, &acts.inv_rms1, self.grads.layer(0, T_ATTN_GAIN), h);
            }
            11 => self.dispatch_rope(chunks, ctx, &acts.q, c.num_heads, false),
            12 => dispatch_swiglu(chunks.enc(), ctx, &acts.gate, &acts.up, &s.act, t * ffn),
            _ => self.encode_adam(chunks, ctx, 0.0, 0.0, 0.0),
        }
    }

    /// Loss on a batch the model is not trained on: forward pass and
    /// cross-entropy, no backward pass and no optimizer step, so nothing
    /// about the model changes.
    ///
    /// This is the number that separates learning from memorizing.
    /// Training loss falls either way; if this one stops falling while
    /// training loss keeps going, the model has started learning the
    /// sample rather than the language.
    pub async fn eval_loss(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
    ) -> Result<f32, String> {
        let t = self.t_len;
        if inputs.len() != targets.len() || inputs.is_empty() || inputs.len() % t != 0 {
            return Err("evaluation batch does not match this trainer's sequence length".to_string());
        }
        let batch_size = inputs.len() / t;
        ctx.params.reset();
        ctx.dispatch_count.set(0);

        let mut chunks = Chunks::new(ctx, self.dispatches_per_submit);
        // Only the loss slot is used, and it accumulates, so it has to
        // start at zero.
        self.dispatch_zero(&mut chunks, ctx, &self.scratch.stats, 1);
        for b in 0..batch_size {
            buffers::write_u32(&ctx.queue, &self.scratch.tokens, &inputs[b * t..(b + 1) * t]);
            buffers::write_u32(&ctx.queue, &self.scratch.targets, &targets[b * t..(b + 1) * t]);
            self.encode_forward(&mut chunks, ctx);
            self.encode_loss(&mut chunks, ctx);
            chunks.flush();
        }
        let stats = buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.stats, 1).await?;
        Ok(stats[0] / (batch_size * t) as f32)
    }

    /// One step, with a device sync after each phase and a stopwatch
    /// around each: the answer to "which part of a step is slow".
    ///
    /// The syncs make the step slower than a real one — a normal step
    /// lets the phases pipeline — so the total here is an upper bound.
    /// What matters is the split between the phases, and how it changes
    /// when `dispatches_per_submit` changes: a step dominated by
    /// per-submit cost gets faster with bigger chunks and shows most of
    /// its time in whichever phase issues the most submits, while a step
    /// dominated by arithmetic barely moves.
    pub async fn profile_step(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
        lr: f32,
        weight_decay: f32,
        grad_clip: f32,
    ) -> Result<GpuStepReport, String> {
        let t = self.t_len;
        if inputs.len() != targets.len() || inputs.is_empty() || inputs.len() % t != 0 {
            return Err("batch shape does not match this trainer".to_string());
        }
        let batch_size = inputs.len() / t;
        ctx.params.reset();
        ctx.dispatch_count.set(0);
        let mut timings = PhaseTimings::default();
        let started = Instant::now();

        let mut chunks = Chunks::new(ctx, self.dispatches_per_submit);
        for slot in &self.grads.slots {
            self.dispatch_zero(&mut chunks, ctx, &slot.buffer, slot.len);
        }
        let stats_len = 1 + self.grads.slots.len();
        self.dispatch_zero(&mut chunks, ctx, &self.scratch.stats, stats_len);
        chunks.flush();
        self.sync(ctx).await?;
        let mut mark = Instant::now();
        timings.zero = started.elapsed().as_secs_f64() * 1000.0;
        for b in 0..batch_size {
            let seq = &inputs[b * t..(b + 1) * t];
            let tgt = &targets[b * t..(b + 1) * t];
            buffers::write_u32(&ctx.queue, &self.scratch.tokens, seq);
            buffers::write_u32(&ctx.queue, &self.scratch.targets, tgt);
            let groups = self.upload_scatter_index(ctx, seq);

            self.encode_forward(&mut chunks, ctx);
            chunks.flush();
            self.sync(ctx).await?;
            timings.forward += take(&mut mark);

            self.encode_loss(&mut chunks, ctx);
            chunks.flush();
            self.sync(ctx).await?;
            timings.loss += take(&mut mark);

            self.encode_backward(&mut chunks, ctx, groups);
            chunks.flush();
            self.sync(ctx).await?;
            timings.backward += take(&mut mark);
        }

        mark = Instant::now();
        for (i, slot) in self.grads.slots.iter().enumerate() {
            self.dispatch_reduce(&mut chunks, ctx, &slot.buffer, slot.len, 1 + i, true);
        }
        chunks.flush();
        self.sync(ctx).await?;
        timings.reduce = take(&mut mark);

        let stats = buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.stats, stats_len).await?;
        timings.readback = take(&mut mark);

        let inv_batch = 1.0 / batch_size as f32;
        let loss = stats[0] / (batch_size * t) as f32;
        let sum_sq: f64 = stats[1..].iter().map(|&x| x as f64).sum();
        let grad_norm = (sum_sq.sqrt() as f32) * inv_batch;
        let clip = if grad_norm > grad_clip && grad_norm.is_finite() && grad_norm > 0.0 {
            grad_clip / grad_norm
        } else {
            1.0
        };
        self.step += 1;
        self.encode_adam(&mut chunks, ctx, lr, weight_decay, inv_batch * clip);
        chunks.flush();
        self.sync(ctx).await?;
        timings.adam = take(&mut mark);
        timings.total = started.elapsed().as_secs_f64() * 1000.0;

        Ok(GpuStepReport {
            loss,
            lr,
            grad_norm,
            phase_ms: Some(timings),
            tokens: batch_size * t,
            dispatches: ctx.dispatch_count.get(),
            submits: chunks.submits,
        })
    }

    /// The AdamW pass: one dispatch per tensor, with the batch average
    /// and the gradient-norm clip folded into `grad_scale` so neither
    /// needs its own pass over every parameter.
    fn encode_adam(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        lr: f32,
        weight_decay: f32,
        grad_scale: f32,
    ) {
        let bias1 = 1.0 - 0.9f32.powi(self.step);
        let bias2 = 1.0 - 0.95f32.powi(self.step);
        for (i, slot) in self.grads.slots.iter().enumerate() {
            let wd = if slot.decay { weight_decay } else { 0.0 };
            let (groups, stride) = self.stride_dispatch(ctx, slot.len);
            let params = ctx.params.alloc(
                &ctx.device,
                &ctx.queue,
                AdamParams {
                    len: slot.len as u32,
                    lr,
                    bias1,
                    bias2,
                    weight_decay: wd,
                    grad_scale,
                    stride,
                    _p0: 0,
                },
            );
            let entries = [
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.weights.slots[i].buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: slot.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.m.slots[i].buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.v.slots[i].buffer.as_entire_binding() },
            ];
            dispatch(chunks.enc(), ctx, &ctx.pipelines.adam_update, &entries, (groups, 1, 1));
        }
    }

    /// Wait for everything submitted so far to actually finish on the
    /// device. Without this a "phase timing" is only how long encoding
    /// took, which is not the question.
    async fn sync(&self, ctx: &GpuContext) -> Result<(), String> {
        buffers::sync(&ctx.device, &ctx.queue, &self.scratch.stats).await
    }

    /// Copy the trained weights back into a CPU-side `ModelWeights` —
    /// for checkpoint export, and for handing the model to the
    /// generation backend.
    pub async fn download_weights(&self, ctx: &GpuContext) -> Result<ModelWeights, String> {
        let mut out = ModelWeights::zeros(&self.config);
        let requests: Vec<(&wgpu::Buffer, usize)> =
            self.weights.slots.iter().map(|slot| (&slot.buffer, slot.len)).collect();
        let flat = buffers::read_f32_concat(&ctx.device, &ctx.queue, &requests).await?;
        let mut at = 0usize;
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(self.weights.slots.len());
        for slot in &self.weights.slots {
            values.push(flat[at..at + slot.len].to_vec());
            at += slot.len;
        }
        let mut next = values.into_iter();
        out.embed = next.next().expect("embed");
        for layer in &mut out.layers {
            layer.attn_norm_gain = next.next().expect("attn_norm_gain");
            layer.wq = next.next().expect("wq");
            layer.wk = next.next().expect("wk");
            layer.wv = next.next().expect("wv");
            layer.wo = next.next().expect("wo");
            layer.mlp_norm_gain = next.next().expect("mlp_norm_gain");
            layer.w_gate = next.next().expect("w_gate");
            layer.w_up = next.next().expect("w_up");
            layer.w_down = next.next().expect("w_down");
        }
        out.final_norm_gain = next.next().expect("final_norm_gain");
        Ok(out)
    }

    /// The Adam moment buffers and this trainer's step count.
    ///
    /// A checkpoint without these resumes with the momentum reset, and
    /// the loss visibly jumps at every restart - which, now that the page
    /// saves and restores a model on its own, would happen on every
    /// visit.
    pub async fn download_optimizer(
        &self,
        ctx: &GpuContext,
    ) -> Result<(ModelWeights, ModelWeights, i32), String> {
        let m = self.download_set(ctx, &self.m).await?;
        let v = self.download_set(ctx, &self.v).await?;
        Ok((m, v, self.step))
    }

    /// Put moment buffers back, from a saved state. Shapes must match the
    /// model this trainer holds.
    pub fn upload_optimizer(
        &mut self,
        ctx: &GpuContext,
        m: &ModelWeights,
        v: &ModelWeights,
        step: i32,
    ) -> Result<(), String> {
        self.write_set(ctx, &self.m, m)?;
        self.write_set(ctx, &self.v, v)?;
        self.step = step;
        Ok(())
    }

    /// One `ParamSet` read back into a CPU-side `ModelWeights`.
    async fn download_set(&self, ctx: &GpuContext, set: &ParamSet) -> Result<ModelWeights, String> {
        let requests: Vec<(&wgpu::Buffer, usize)> =
            set.slots.iter().map(|slot| (&slot.buffer, slot.len)).collect();
        let flat = buffers::read_f32_concat(&ctx.device, &ctx.queue, &requests).await?;
        let mut out = ModelWeights::zeros(&self.config);
        let mut at = 0usize;
        let mut next = |len: usize| {
            let slice = flat[at..at + len].to_vec();
            at += len;
            slice
        };
        out.embed = next(self.config.vocab_size() * self.config.hidden_dim);
        let (h, kv, ffn) = (self.config.hidden_dim, self.config.kv_dim(), self.config.ffn_dim());
        for layer in &mut out.layers {
            layer.attn_norm_gain = next(h);
            layer.wq = next(h * h);
            layer.wk = next(kv * h);
            layer.wv = next(kv * h);
            layer.wo = next(h * h);
            layer.mlp_norm_gain = next(h);
            layer.w_gate = next(ffn * h);
            layer.w_up = next(ffn * h);
            layer.w_down = next(h * ffn);
        }
        out.final_norm_gain = next(h);
        Ok(out)
    }

    /// The reverse: a CPU-side `ModelWeights` written into a `ParamSet`.
    fn write_set(&self, ctx: &GpuContext, set: &ParamSet, source: &ModelWeights) -> Result<(), String> {
        let mut tensors: Vec<&[f32]> = vec![&source.embed];
        for layer in &source.layers {
            tensors.extend_from_slice(&[
                &layer.attn_norm_gain[..],
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.mlp_norm_gain,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ]);
        }
        tensors.push(&source.final_norm_gain);
        if tensors.len() != set.slots.len() {
            return Err("optimizer state does not match this model's shape".to_string());
        }
        for (slot, data) in set.slots.iter().zip(tensors) {
            if slot.len != data.len() {
                return Err("optimizer state does not match this model's shape".to_string());
            }
            ctx.queue.write_buffer(&slot.buffer, 0, bytemuck::cast_slice(data));
        }
        Ok(())
    }

    /// The one number that says whether these kernels are right: run the
    /// same sequence through this backend's forward pass and through
    /// `llm_core`'s gradient-checked CPU one, from the same weights, and
    /// return the largest absolute difference between their logits.
    ///
    /// Float rounding over this many accumulations lands around `1e-3`.
    /// Anything much larger means a kernel is wrong.
    pub async fn debug_compare_forward(
        &mut self,
        ctx: &GpuContext,
        tokens: &[u32],
    ) -> Result<f32, String> {
        if tokens.len() != self.t_len {
            return Err(format!("compare needs exactly {} tokens", self.t_len));
        }
        ctx.params.reset();
        ctx.dispatch_count.set(0);
        buffers::write_u32(&ctx.queue, &self.scratch.tokens, tokens);
        let mut chunks = Chunks::new(ctx, self.dispatches_per_submit);
        self.encode_forward(&mut chunks, ctx);
        chunks.flush();
        let vocab = self.config.vocab_size();
        let gpu_logits =
            buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.logits, self.t_len * vocab).await?;

        let weights = self.download_weights(ctx).await?;
        let (cpu_logits, _) = llm_core::model::forward(&weights, &self.config, tokens);
        Ok(cpu_logits
            .iter()
            .zip(&gpu_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max))
    }

    // --- encoding -------------------------------------------------------

    fn encode_forward(&self, chunks: &mut Chunks, ctx: &GpuContext) {
        let t = self.t_len;
        let c = &self.config;
        let (h, kv, ffn) = (c.hidden_dim, c.kv_dim(), c.ffn_dim());
        let band = ops::band_width(t, c.effective_window());

        self.dispatch_gather(chunks, ctx, self.weights.embed(), &self.scratch.tokens, &self.scratch.hidden);

        for (l, acts) in self.acts.iter().enumerate() {
            chunks.copy(&self.scratch.hidden, &acts.h_in, t * h);
            self.dispatch_rmsnorm(
                chunks,
                ctx,
                &self.scratch.hidden,
                self.weights.layer(l, T_ATTN_GAIN),
                &acts.normed1,
                &acts.inv_rms1,
                h,
            );
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(l, T_WQ), &acts.q, t, h, h);
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(l, T_WK), &acts.k, t, h, kv);
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(l, T_WV), &acts.v, t, h, kv);
            self.dispatch_rope(chunks, ctx, &acts.q, c.num_heads, false);
            self.dispatch_rope(chunks, ctx, &acts.k, c.num_kv_heads, false);
            self.dispatch_attention_fwd(chunks, ctx, acts, band);
            dispatch_linear(
                chunks.enc(),
                ctx,
                &acts.concat,
                self.weights.layer(l, T_WO),
                &self.scratch.tmp_h,
                t,
                h,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &self.scratch.hidden, &self.scratch.tmp_h, t * h);
            chunks.copy(&self.scratch.hidden, &acts.h_after_attn, t * h);

            self.dispatch_rmsnorm(
                chunks,
                ctx,
                &self.scratch.hidden,
                self.weights.layer(l, T_MLP_GAIN),
                &acts.normed2,
                &acts.inv_rms2,
                h,
            );
            dispatch_linear(chunks.enc(), ctx, &acts.normed2, self.weights.layer(l, T_W_GATE), &acts.gate, t, h, ffn);
            dispatch_linear(chunks.enc(), ctx, &acts.normed2, self.weights.layer(l, T_W_UP), &acts.up, t, h, ffn);
            dispatch_swiglu(chunks.enc(), ctx, &acts.gate, &acts.up, &self.scratch.act, t * ffn);
            dispatch_linear(
                chunks.enc(),
                ctx,
                &self.scratch.act,
                self.weights.layer(l, T_W_DOWN),
                &self.scratch.tmp_h,
                t,
                ffn,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &self.scratch.hidden, &self.scratch.tmp_h, t * h);
        }

        chunks.copy(&self.scratch.hidden, &self.scratch.h_final, t * h);
        self.dispatch_rmsnorm(
            chunks,
            ctx,
            &self.scratch.hidden,
            self.weights.final_gain(&self.config),
            &self.scratch.final_normed,
            &self.scratch.final_inv_rms,
            h,
        );
        // Weight-tied output head: logits = final_normed @ embed^T.
        dispatch_linear(
            chunks.enc(),
            ctx,
            &self.scratch.final_normed,
            self.weights.embed(),
            &self.scratch.logits,
            t,
            h,
            self.config.vocab_size(),
        );
    }

    fn encode_loss(&self, chunks: &mut Chunks, ctx: &GpuContext) {
        let t = self.t_len;
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: t as u32, b: self.config.vocab_size() as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.scratch.logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.scratch.targets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.scratch.d_logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.scratch.loss_rows.as_entire_binding() },
        ];
        // One workgroup per row; its 256 threads split the vocabulary.
        dispatch(chunks.enc(), ctx, &ctx.pipelines.cross_entropy, &entries, (t as u32, 1, 1));
        // The per-row losses reduce straight into the step's loss slot,
        // so the batch costs one readback in total rather than one per
        // sequence.
        self.dispatch_reduce(chunks, ctx, &self.scratch.loss_rows, t, 0, false);
    }

    /// Groups this sequence's token positions by token id and uploads the
    /// result for the embedding scatter. Returns how many distinct ids
    /// there are, which is the scatter's dispatch width.
    fn upload_scatter_index(&self, ctx: &GpuContext, tokens: &[u32]) -> usize {
        let mut ids: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut positions_by_id: Vec<Vec<u32>> = Vec::with_capacity(tokens.len());
        for (position, &id) in tokens.iter().enumerate() {
            match ids.iter().position(|&existing| existing == id) {
                Some(group) => positions_by_id[group].push(position as u32),
                None => {
                    ids.push(id);
                    positions_by_id.push(vec![position as u32]);
                }
            }
        }
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        let mut positions = Vec::with_capacity(tokens.len());
        offsets.push(0u32);
        for group in &positions_by_id {
            positions.extend_from_slice(group);
            offsets.push(positions.len() as u32);
        }
        buffers::write_u32(&ctx.queue, &self.scratch.row_ids, &ids);
        buffers::write_u32(&ctx.queue, &self.scratch.row_offsets, &offsets);
        buffers::write_u32(&ctx.queue, &self.scratch.row_positions, &positions);
        ids.len()
    }

    fn encode_backward(&self, chunks: &mut Chunks, ctx: &GpuContext, scatter_groups: usize) {
        let t = self.t_len;
        let c = &self.config;
        let (h, kv, ffn, vocab) = (c.hidden_dim, c.kv_dim(), c.ffn_dim(), c.vocab_size());
        let band = ops::band_width(t, c.effective_window());
        let s = &self.scratch;

        // Output head, tied with the embedding table: its gradient gets
        // both this contribution and the input-gather one at the end.
        self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_logits, &s.final_normed, self.grads.embed(), t, h, vocab);
        self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_logits, self.weights.embed(), &s.d_a, t, h, vocab);
        self.dispatch_rmsnorm_bwd_dgain(
            chunks,
            ctx,
            &s.d_a,
            &s.h_final,
            &s.final_inv_rms,
            self.grads.final_gain(c),
            h,
        );
        self.dispatch_rmsnorm_bwd_dx(
            chunks,
            ctx,
            &s.d_a,
            &s.h_final,
            self.weights.final_gain(c),
            &s.final_inv_rms,
            &s.d_hidden,
            h,
        );

        for l in (0..c.num_layers).rev() {
            let acts = &self.acts[l];

            // --- MLP branch ---
            dispatch_swiglu(chunks.enc(), ctx, &acts.gate, &acts.up, &s.act, t * ffn);
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_hidden,
                &s.act,
                self.grads.layer(l, T_W_DOWN),
                t,
                ffn,
                h,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_hidden,
                self.weights.layer(l, T_W_DOWN),
                &s.d_act,
                t,
                ffn,
                h,
            );
            self.dispatch_swiglu_bwd(chunks, ctx, &s.d_act, &acts.gate, &acts.up, t * ffn);
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_gate,
                &acts.normed2,
                self.grads.layer(l, T_W_GATE),
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_up,
                &acts.normed2,
                self.grads.layer(l, T_W_UP),
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_gate,
                self.weights.layer(l, T_W_GATE),
                &s.d_a,
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_up,
                self.weights.layer(l, T_W_UP),
                &s.d_b,
                t,
                h,
                ffn,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_b, t * h);

            self.dispatch_rmsnorm_bwd_dgain(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_after_attn,
                &acts.inv_rms2,
                self.grads.layer(l, T_MLP_GAIN),
                h,
            );
            self.dispatch_rmsnorm_bwd_dx(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_after_attn,
                self.weights.layer(l, T_MLP_GAIN),
                &acts.inv_rms2,
                &s.d_c,
                h,
            );
            // The residual splits the gradient: what arrived plus what
            // came back through the norm branch.
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_hidden, &s.d_c, t * h);

            // --- Attention branch ---
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_hidden,
                &acts.concat,
                self.grads.layer(l, T_WO),
                t,
                h,
                h,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_hidden,
                self.weights.layer(l, T_WO),
                &s.d_a,
                t,
                h,
                h,
            );
            self.dispatch_attention_bwd(chunks, ctx, acts, band);
            self.dispatch_rope(chunks, ctx, &s.d_q, c.num_heads, true);
            self.dispatch_rope(chunks, ctx, &s.d_k, c.num_kv_heads, true);

            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_q, &acts.normed1, self.grads.layer(l, T_WQ), t, h, h);
            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_k, &acts.normed1, self.grads.layer(l, T_WK), t, h, kv);
            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_v, &acts.normed1, self.grads.layer(l, T_WV), t, h, kv);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_q, self.weights.layer(l, T_WQ), &s.d_a, t, h, h);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_k, self.weights.layer(l, T_WK), &s.d_b, t, h, kv);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_v, self.weights.layer(l, T_WV), &s.d_c, t, h, kv);
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_b, t * h);
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_c, t * h);

            self.dispatch_rmsnorm_bwd_dgain(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_in,
                &acts.inv_rms1,
                self.grads.layer(l, T_ATTN_GAIN),
                h,
            );
            self.dispatch_rmsnorm_bwd_dx(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_in,
                self.weights.layer(l, T_ATTN_GAIN),
                &acts.inv_rms1,
                &s.d_c,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_hidden, &s.d_c, t * h);
        }

        // The other half of the tied embedding gradient: the input gather.
        self.dispatch_scatter_add(chunks, ctx, &s.d_hidden, self.grads.embed(), scatter_groups, h);
    }

    // --- dispatch helpers ------------------------------------------------

    /// Workgroups for a grid-stride kernel over `len` elements: enough to
    /// fill the device, never more than the adapter's per-dimension limit.
    /// A model's embedding table is millions of floats, and asking for one
    /// workgroup per 64 of them exceeds that limit (65535 on most
    /// adapters) — which fails validation rather than merely running slow.
    fn stride_dispatch(&self, ctx: &GpuContext, len: usize) -> (u32, u32) {
        let wanted = ceil_div(len, 64);
        let cap = ctx.max_workgroups_per_dimension.max(1).min(4096);
        let groups = wanted.min(cap).max(1);
        (groups, groups * 64)
    }

    fn dispatch_zero(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        buffer: &wgpu::Buffer,
        len: usize,
    ) {
        let (groups, stride) = self.stride_dispatch(ctx, len);
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: len as u32, b: stride, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.zero, &entries, (groups, 1, 1));
    }

    /// Sum a buffer (or its squares) into one stats slot, in two stages:
    /// `REDUCE_GROUPS` workgroups each reduce a slice into `partials`,
    /// then one workgroup sums those. One workgroup walking a whole
    /// tensor alone was serial time proportional to the largest tensor,
    /// paid once per tensor per step.
    fn dispatch_reduce(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        src: &wgpu::Buffer,
        len: usize,
        slot: usize,
        square: bool,
    ) {
        let groups = REDUCE_GROUPS.min(len.div_ceil(256)).max(1) as u32;
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: len as u32, b: 0, c: u32::from(square), d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.scratch.partials.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.reduce, &entries, (groups, 1, 1));

        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: 0, b: groups, c: slot as u32, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.scratch.partials.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.scratch.stats.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.reduce_finish, &entries, (1, 1, 1));
    }

    fn dispatch_gather(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        table: &wgpu::Buffer,
        ids: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let h = self.config.hidden_dim;
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: self.t_len as u32, b: h as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.embedding_gather,
            &entries,
            (ceil_div(self.t_len, 8), ceil_div(h, 8), 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_scatter_add(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        d_rows: &wgpu::Buffer,
        table_grad: &wgpu::Buffer,
        groups: usize,
        hidden: usize,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: groups as u32, b: hidden as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d_rows.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.scratch.row_ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.scratch.row_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.scratch.row_positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: table_grad.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.embedding_scatter_add,
            &entries,
            (ceil_div(groups, 8), ceil_div(hidden, 8), 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_rmsnorm(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        gain: &wgpu::Buffer,
        out: &wgpu::Buffer,
        inv_rms: &wgpu::Buffer,
        dim: usize,
    ) {
        // eps is a float in this shader's Params, so it goes through a
        // struct of its own rather than the all-words P4.
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            RmsParams { rows: self.t_len as u32, dim: dim as u32, eps: RMS_EPS, _pad: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: inv_rms.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.rmsnorm, &entries, (ceil_div(self.t_len, 64), 1, 1));
    }

    fn dispatch_rope(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        heads: usize,
        inverse: bool,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P8 {
                a: self.t_len as u32,
                b: heads as u32,
                c: self.config.head_dim() as u32,
                d: 0,
                e: u32::from(inverse),
                f: 0,
                g: 0,
                h: 0,
            },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.rope,
            &entries,
            (ceil_div(self.t_len, 8), ceil_div(heads, 8), 1),
        );
    }

    fn attention_params(&self, band: usize) -> P8 {
        P8 {
            a: self.t_len as u32,
            b: self.config.num_heads as u32,
            c: self.config.num_kv_heads as u32,
            d: self.config.head_dim() as u32,
            e: band as u32,
            f: 0,
            g: 0,
            h: 0,
        }
    }

    fn dispatch_attention_fwd(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        acts: &LayerActs,
        band: usize,
    ) {
        let params = ctx.params.alloc(&ctx.device, &ctx.queue, self.attention_params(band));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: acts.q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: acts.k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: acts.v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: acts.concat.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: acts.probs.as_entire_binding() },
        ];
        // One workgroup per (row, head); the 64 threads inside split the
        // window - see shaders/attention_fwd.wgsl.
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.attention_fwd,
            &entries,
            (self.t_len as u32, self.config.num_heads as u32, 1),
        );
    }

    /// The three attention-backward kernels, in the order they depend on
    /// each other: softmax backward first, then dQ and dK/dV from it.
    fn dispatch_attention_bwd(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        acts: &LayerActs,
        band: usize,
    ) {
        let s = &self.scratch;
        let heads = self.config.num_heads;
        let kv_heads = self.config.num_kv_heads;

        let params = ctx.params.alloc(&ctx.device, &ctx.queue, self.attention_params(band));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: s.d_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: acts.v.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: acts.probs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: s.d_score.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.attention_bwd_dscore,
            &entries,
            (self.t_len as u32, heads as u32, 1),
        );

        let params = ctx.params.alloc(&ctx.device, &ctx.queue, self.attention_params(band));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: s.d_score.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: acts.k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: s.d_q.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.attention_bwd_dq,
            &entries,
            (self.t_len as u32, heads as u32, 1),
        );

        let params = ctx.params.alloc(&ctx.device, &ctx.queue, self.attention_params(band));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: s.d_score.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: acts.probs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: acts.q.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: s.d_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: s.d_k.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: s.d_v.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.attention_bwd_dkdv,
            &entries,
            (self.t_len as u32, kv_heads as u32, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_bwd_dw(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        dy: &wgpu::Buffer,
        x: &wgpu::Buffer,
        dw: &wgpu::Buffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: rows as u32, b: in_dim as u32, c: out_dim as u32, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dw.as_entire_binding() },
        ];
        // gid.x covers in_dim, gid.y covers out_dim, both in 64-wide
        // tiles - see shaders/linear_bwd_dw.wgsl.
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.linear_bwd_dw,
            &entries,
            (ceil_div(in_dim, 64), ceil_div(out_dim, 64), 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_bwd_dx(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        dy: &wgpu::Buffer,
        w: &wgpu::Buffer,
        dx: &wgpu::Buffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: rows as u32, b: in_dim as u32, c: out_dim as u32, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dx.as_entire_binding() },
        ];
        // gid.x covers in_dim, gid.y covers rows, both in 64-wide tiles.
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.linear_bwd_dx,
            &entries,
            (ceil_div(in_dim, 64), ceil_div(rows, 64), 1),
        );
    }

    fn dispatch_swiglu_bwd(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        d_act: &wgpu::Buffer,
        gate: &wgpu::Buffer,
        up: &wgpu::Buffer,
        len: usize,
    ) {
        let params = ctx.params.alloc(&ctx.device, &ctx.queue, P4 { a: len as u32, b: 0, c: 0, d: 0 });
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d_act.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: up.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.scratch.d_gate.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: self.scratch.d_up.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.swiglu_bwd, &entries, (ceil_div(len, 64), 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_rmsnorm_bwd_dx(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        dy: &wgpu::Buffer,
        x: &wgpu::Buffer,
        gain: &wgpu::Buffer,
        inv_rms: &wgpu::Buffer,
        dx: &wgpu::Buffer,
        dim: usize,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: self.t_len as u32, b: dim as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gain.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: inv_rms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dx.as_entire_binding() },
        ];
        dispatch(
            chunks.enc(),
            ctx,
            &ctx.pipelines.rmsnorm_bwd_dx,
            &entries,
            (ceil_div(self.t_len, 64), 1, 1),
        );
    }

    fn dispatch_rmsnorm_bwd_dgain(
        &self,
        chunks: &mut Chunks,
        ctx: &GpuContext,
        dy: &wgpu::Buffer,
        x: &wgpu::Buffer,
        inv_rms: &wgpu::Buffer,
        dgain: &wgpu::Buffer,
        dim: usize,
    ) {
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: self.t_len as u32, b: dim as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: inv_rms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dgain.as_entire_binding() },
        ];
        dispatch(chunks.enc(), ctx, &ctx.pipelines.rmsnorm_bwd_dgain, &entries, (ceil_div(dim, 64), 1, 1));
    }
}

/// Milliseconds since `mark`, restarting it.
fn take(mark: &mut Instant) -> f64 {
    let ms = mark.elapsed().as_secs_f64() * 1000.0;
    *mark = Instant::now();
    ms
}

/// Encodes work in bounded chunks and submits each one.
///
/// A whole sequence's forward and backward pass in a single command
/// buffer is several hundred GFLOP of work in one submission. Windows
/// resets any GPU whose submission runs past its ~2 second watchdog
/// (`DXGI_ERROR_DEVICE_HUNG`), which is exactly what a step of this size
/// did: the device was killed mid-training. Submitting every few
/// dispatches keeps each submission far under the watchdog, at the cost
/// of a few hundred microseconds per step - and the queue still runs
/// them in order, so nothing about the arithmetic changes.
struct Chunks<'a> {
    ctx: &'a GpuContext,
    encoder: Option<wgpu::CommandEncoder>,
    since_submit: u32,
    per_submit: u32,
    pub submits: u32,
}

impl<'a> Chunks<'a> {
    fn new(ctx: &'a GpuContext, per_submit: u32) -> Self {
        Self { ctx, encoder: None, since_submit: 0, per_submit, submits: 0 }
    }

    /// The encoder to put the next operation into.
    ///
    /// Counting happens here rather than in a separate call the caller
    /// has to remember: one `enc()` is one operation, so the chunk is
    /// submitted the moment it is full and a fresh encoder starts. The
    /// submit happens *between* operations, never inside one.
    fn enc(&mut self) -> &mut wgpu::CommandEncoder {
        if self.since_submit >= self.per_submit {
            self.flush();
        }
        if self.encoder.is_none() {
            self.encoder = Some(self.ctx.device.create_command_encoder(&Default::default()));
        }
        self.since_submit += 1;
        self.encoder.as_mut().expect("just created")
    }

    fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.ctx.queue.submit(Some(encoder.finish()));
            self.submits += 1;
        }
        self.since_submit = 0;
    }

    fn copy(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, len: usize) {
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        self.enc().copy_buffer_to_buffer(src, 0, dst, 0, bytes);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamParams {
    len: u32,
    lr: f32,
    bias1: f32,
    bias2: f32,
    weight_decay: f32,
    grad_scale: f32,
    stride: u32,
    _p0: u32,
}
