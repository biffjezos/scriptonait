//! Timing: `profile_kernels` times each kernel on its own at a step's
//! real shapes, `profile_step` times a whole step's phases against each
//! other. Both exist to answer "why is this slow", not to run in the
//! hot path of a normal step.

use web_time::Instant;

use crate::context::GpuContext;
use crate::model::{dispatch_linear, dispatch_swiglu};

use super::dispatch::Chunks;
use super::layout::{T_ATTN_GAIN, T_WK, T_WQ, T_W_DOWN, T_W_GATE};
use super::{GpuStepReport, GpuTrainer, PhaseTimings};

use llm_core::ops;

/// Milliseconds since `mark`, restarting it.
fn take(mark: &mut Instant) -> f64 {
    let ms = mark.elapsed().as_secs_f64() * 1000.0;
    *mark = Instant::now();
    ms
}

impl GpuTrainer {
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
        // The model's full depth — this profiles the kernel sequence
        // itself, not any particular sampled core depth.
        let layout = self.config.layer_layout(None);

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
            crate::buffers::write_u32(&ctx.queue, &self.scratch.tokens, seq);
            crate::buffers::write_u32(&ctx.queue, &self.scratch.targets, tgt);
            let groups = self.upload_scatter_index(ctx, seq);

            self.encode_forward(&mut chunks, ctx, &layout);
            chunks.flush();
            self.sync(ctx).await?;
            timings.forward += take(&mut mark);

            self.encode_loss(&mut chunks, ctx);
            chunks.flush();
            self.sync(ctx).await?;
            timings.loss += take(&mut mark);

            self.encode_backward(&mut chunks, ctx, groups, &layout);
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

        let stats = crate::buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.stats, stats_len).await?;
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
}
