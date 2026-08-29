//! The low-level GPU dispatch helpers every phase (forward, backward,
//! profiling) is built from, plus `Chunks`, the bounded-command-buffer
//! encoder every one of them threads through.

use crate::context::GpuContext;
use crate::model::{ceil_div, dispatch, P4, P8};

use super::layout::{LayerActs, Scratch};
use super::GpuTrainer;

/// Must match `llm_core::model`'s RMS_EPS, which is private there.
pub(super) const RMS_EPS: f32 = 1e-6;

/// Workgroups a reduction splits its tensor across. Enough to keep a
/// small GPU busy on a multi-million element tensor, small enough that
/// summing the partials afterwards is one cheap dispatch.
///
/// `pub(super)`: `mod.rs` sizes the `partials` scratch buffer to exactly
/// this many slots when it allocates `Scratch`.
pub(super) const REDUCE_GROUPS: usize = 64;

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
pub(super) struct Chunks<'a> {
    ctx: &'a GpuContext,
    encoder: Option<wgpu::CommandEncoder>,
    since_submit: u32,
    per_submit: u32,
    pub(super) submits: u32,
}

impl<'a> Chunks<'a> {
    pub(super) fn new(ctx: &'a GpuContext, per_submit: u32) -> Self {
        Self { ctx, encoder: None, since_submit: 0, per_submit, submits: 0 }
    }

    /// The encoder to put the next operation into.
    ///
    /// Counting happens here rather than in a separate call the caller
    /// has to remember: one `enc()` is one operation, so the chunk is
    /// submitted the moment it is full and a fresh encoder starts. The
    /// submit happens *between* operations, never inside one.
    pub(super) fn enc(&mut self) -> &mut wgpu::CommandEncoder {
        if self.since_submit >= self.per_submit {
            self.flush();
        }
        if self.encoder.is_none() {
            self.encoder = Some(self.ctx.device.create_command_encoder(&Default::default()));
        }
        self.since_submit += 1;
        self.encoder.as_mut().expect("just created")
    }

    pub(super) fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.ctx.queue.submit(Some(encoder.finish()));
            self.submits += 1;
        }
        self.since_submit = 0;
    }

    pub(super) fn copy(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, len: usize) {
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

impl GpuTrainer {
    /// Workgroups for a grid-stride kernel over `len` elements: enough to
    /// fill the device, never more than the adapter's per-dimension limit.
    /// A model's embedding table is millions of floats, and asking for one
    /// workgroup per 64 of them exceeds that limit (65535 on most
    /// adapters) — which fails validation rather than merely running slow.
    pub(super) fn stride_dispatch(&self, ctx: &GpuContext, len: usize) -> (u32, u32) {
        let wanted = ceil_div(len, 64);
        let cap = ctx.max_workgroups_per_dimension.max(1).min(4096);
        let groups = wanted.min(cap).max(1);
        (groups, groups * 64)
    }

    pub(super) fn dispatch_zero(
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
    pub(super) fn dispatch_reduce(
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

    pub(super) fn dispatch_gather(
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
    pub(super) fn dispatch_scatter_add(
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
    pub(super) fn dispatch_rmsnorm(
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

    pub(super) fn dispatch_rope(
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

    pub(super) fn attention_params(&self, band: usize) -> P8 {
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

    pub(super) fn dispatch_attention_fwd(
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
    pub(super) fn dispatch_attention_bwd(
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
    pub(super) fn dispatch_linear_bwd_dw(
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
    pub(super) fn dispatch_linear_bwd_dx(
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

    pub(super) fn dispatch_swiglu_bwd(
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
    pub(super) fn dispatch_rmsnorm_bwd_dx(
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

    pub(super) fn dispatch_rmsnorm_bwd_dgain(
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

    /// The AdamW pass: one dispatch per tensor, with the batch average
    /// and the gradient-norm clip folded into `grad_scale` so neither
    /// needs its own pass over every parameter.
    pub(super) fn encode_adam(
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
}
