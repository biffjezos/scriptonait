//! Uploads a trained `llm_core::model::ModelWeights` to the GPU once, then
//! runs forward, backward, and Adam entirely on-device. This mirrors
//! `llm_core::model::forward`/`backward` and `llm_core::train::Trainer`
//! step for step — see those for the reference this must match; ops.rs's
//! doc comments explain the math each kernel here implements.
//!
//! Every backward kernel is written as a *gather*, never a *scatter*:
//! each output element is computed by exactly one thread reading
//! whatever it needs, rather than many threads writing into a shared
//! accumulator. That's what lets this run with zero atomics anywhere in
//! the crate, even for operations (embedding/PLE gradients, gradient
//! accumulation across a batch) that would naturally be scatters on a
//! CPU. See the individual .wgsl files for how each one avoids it.
//!
//! One forward implementation (`forward`) is shared by both inference
//! (`forward_last_logits`) and training (`train_step`): it always
//! populates the per-layer backward cache, even when a caller only wants
//! the last token's logits. That costs inference a bit of extra scratch
//! memory it doesn't strictly need, in exchange for only one forward
//! pass to keep correct — worth it given this code can't be executed at
//! all in this project's dev sandbox (see the crate root docs).

use llm_core::config::ModelConfig;
use llm_core::corpus::Batch;
use llm_core::model::ModelWeights;

use crate::buffers::{read_f32, storage_f32, uniform, upload_f32, upload_u32, write_u32};
use crate::context::{GpuContext, MAX_GPU_WINDOW};

fn ceil_div(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct P4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

fn dispatch(ctx: &GpuContext, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry], workgroups: (u32, u32, u32)) {
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

// --- Forward-pass dispatch helpers ------------------------------------

fn dispatch_linear(ctx: &GpuContext, x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, in_dim: usize, out_dim: usize) {
    let params = uniform(&ctx.device, "linear-params", P4 { a: rows as u32, b: in_dim as u32, c: out_dim as u32, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.linear, &entries, (ceil_div(rows, 8), ceil_div(out_dim, 8), 1));
}

fn dispatch_add_inplace(ctx: &GpuContext, dst: &wgpu::Buffer, src: &wgpu::Buffer, len: usize) {
    let params = uniform(&ctx.device, "add-params", P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: src.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.add_inplace, &entries, (ceil_div(len, 64), 1, 1));
}

fn dispatch_gather(ctx: &GpuContext, table: &wgpu::Buffer, ids: &wgpu::Buffer, out: &wgpu::Buffer, t_len: usize, hidden: usize) {
    let params = uniform(&ctx.device, "gather-params", P4 { a: t_len as u32, b: hidden as u32, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: table.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: ids.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.embedding_gather, &entries, (ceil_div(t_len, 8), ceil_div(hidden, 8), 1));
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsnormParams {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_rmsnorm(ctx: &GpuContext, x: &wgpu::Buffer, gain: &wgpu::Buffer, y: &wgpu::Buffer, inv_rms_out: &wgpu::Buffer, rows: usize, dim: usize) {
    let params = uniform(&ctx.device, "rmsnorm-params", RmsnormParams { rows: rows as u32, dim: dim as u32, eps: 1e-6, _pad: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: gain.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: inv_rms_out.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.rmsnorm, &entries, (ceil_div(rows, 64), 1, 1));
}

fn dispatch_rope(ctx: &GpuContext, x: &wgpu::Buffer, t_len: usize, heads: usize, head_dim: usize, inverse: bool) {
    let params = uniform(&ctx.device, "rope-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: inverse as u32 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.rope, &entries, (ceil_div(t_len, 8), ceil_div(heads, 8), 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_attention(
    ctx: &GpuContext,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    probs_out: &wgpu::Buffer,
    t_len: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
) {
    let params = uniform(&ctx.device, "attn-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: window as u32 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: v.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: probs_out.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.attention, &entries, (ceil_div(t_len, 8), ceil_div(heads, 8), 1));
}

fn dispatch_swiglu(ctx: &GpuContext, gate: &wgpu::Buffer, up: &wgpu::Buffer, out: &wgpu::Buffer, len: usize) {
    let params = uniform(&ctx.device, "swiglu-params", P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gate.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.swiglu, &entries, (ceil_div(len, 64), 1, 1));
}

// --- Backward-pass dispatch helpers -----------------------------------

fn dispatch_linear_bwd_dx(ctx: &GpuContext, dy: &wgpu::Buffer, w: &wgpu::Buffer, dx: &wgpu::Buffer, rows: usize, in_dim: usize, out_dim: usize) {
    let params = uniform(&ctx.device, "lbdx-params", P4 { a: rows as u32, b: in_dim as u32, c: out_dim as u32, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: dx.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.linear_bwd_dx, &entries, (ceil_div(rows, 8), ceil_div(in_dim, 8), 1));
}

fn dispatch_linear_bwd_dw(ctx: &GpuContext, dy: &wgpu::Buffer, x: &wgpu::Buffer, dw: &wgpu::Buffer, rows: usize, in_dim: usize, out_dim: usize) {
    let params = uniform(&ctx.device, "lbdw-params", P4 { a: rows as u32, b: in_dim as u32, c: out_dim as u32, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: dw.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.linear_bwd_dw, &entries, (ceil_div(out_dim, 8), ceil_div(in_dim, 8), 1));
}

/// Runs both halves of a linear layer's backward pass.
#[allow(clippy::too_many_arguments)]
fn dispatch_linear_bwd(ctx: &GpuContext, dy: &wgpu::Buffer, x: &wgpu::Buffer, w: &wgpu::Buffer, dx: &wgpu::Buffer, dw: &wgpu::Buffer, rows: usize, in_dim: usize, out_dim: usize) {
    dispatch_linear_bwd_dx(ctx, dy, w, dx, rows, in_dim, out_dim);
    dispatch_linear_bwd_dw(ctx, dy, x, dw, rows, in_dim, out_dim);
}

fn dispatch_rmsnorm_bwd(
    ctx: &GpuContext,
    dy: &wgpu::Buffer,
    x: &wgpu::Buffer,
    gain: &wgpu::Buffer,
    inv_rms: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dgain: &wgpu::Buffer,
    rows: usize,
    dim: usize,
) {
    let params_dx = uniform(&ctx.device, "rbdx-params", P4 { a: rows as u32, b: dim as u32, c: 0, d: 0 });
    let entries_dx = [
        wgpu::BindGroupEntry { binding: 0, resource: params_dx.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: gain.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: inv_rms.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: dx.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.rmsnorm_bwd_dx, &entries_dx, (ceil_div(rows, 64), 1, 1));

    let params_dg = uniform(&ctx.device, "rbdg-params", P4 { a: rows as u32, b: dim as u32, c: 0, d: 0 });
    let entries_dg = [
        wgpu::BindGroupEntry { binding: 0, resource: params_dg.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dy.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: inv_rms.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: dgain.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.rmsnorm_bwd_dgain, &entries_dg, (ceil_div(dim, 64), 1, 1));
}

fn dispatch_swiglu_bwd(ctx: &GpuContext, d_act: &wgpu::Buffer, gate: &wgpu::Buffer, up: &wgpu::Buffer, dgate: &wgpu::Buffer, dup: &wgpu::Buffer, len: usize) {
    let params = uniform(&ctx.device, "sgbwd-params", P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d_act.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: gate.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: up.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: dgate.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: dup.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.swiglu_bwd, &entries, (ceil_div(len, 64), 1, 1));
}

/// Runs all three attention-backward passes; see the .wgsl files for why
/// each is a gather. `d_score_scratch` is reused scratch, not an output
/// the caller needs afterward.
#[allow(clippy::too_many_arguments)]
fn dispatch_attention_bwd(
    ctx: &GpuContext,
    d_out: &wgpu::Buffer,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    probs: &wgpu::Buffer,
    d_score_scratch: &wgpu::Buffer,
    dq: &wgpu::Buffer,
    dk: &wgpu::Buffer,
    dv: &wgpu::Buffer,
    t_len: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
) {
    let params = uniform(&ctx.device, "abwd-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: window as u32 });

    let entries_score = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: v.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: probs.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: d_score_scratch.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.attention_bwd_dscore, &entries_score, (ceil_div(t_len, 8), ceil_div(heads, 8), 1));

    let params_q = uniform(&ctx.device, "abwdq-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: window as u32 });
    let entries_q = [
        wgpu::BindGroupEntry { binding: 0, resource: params_q.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d_score_scratch.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: k.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: dq.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.attention_bwd_dq, &entries_q, (ceil_div(t_len, 8), ceil_div(heads, 8), 1));

    let params_kv = uniform(&ctx.device, "abwdkv-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: window as u32 });
    let entries_kv = [
        wgpu::BindGroupEntry { binding: 0, resource: params_kv.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d_score_scratch.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: probs.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: q.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: d_out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: dk.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: dv.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.attention_bwd_dkdv, &entries_kv, (ceil_div(t_len, 8), ceil_div(heads, 8), 1));
}

fn dispatch_embedding_scatter_add(ctx: &GpuContext, d_rows: &wgpu::Buffer, ids: &wgpu::Buffer, table_grad: &wgpu::Buffer, t_len: usize, hidden: usize, vocab: usize) {
    let params = uniform(&ctx.device, "embgrad-params", P4 { a: t_len as u32, b: hidden as u32, c: vocab as u32, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: d_rows.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: ids.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: table_grad.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.embedding_scatter_add, &entries, (ceil_div(vocab, 8), ceil_div(hidden, 8), 1));
}

fn dispatch_cross_entropy(ctx: &GpuContext, logits: &wgpu::Buffer, targets: &wgpu::Buffer, d_logits: &wgpu::Buffer, loss_out: &wgpu::Buffer, t_len: usize) {
    let params = uniform(&ctx.device, "ce-params", P4 { a: t_len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: logits.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: targets.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: d_logits.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: loss_out.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.cross_entropy, &entries, (ceil_div(t_len, 64), 1, 1));
}

fn dispatch_zero(ctx: &GpuContext, buf: &wgpu::Buffer, len: usize) {
    let params = uniform(&ctx.device, "zero-params", P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.zero, &entries, (ceil_div(len, 64), 1, 1));
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    len: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
}

fn dispatch_scale(ctx: &GpuContext, buf: &wgpu::Buffer, scale: f32, len: usize) {
    let params = uniform(&ctx.device, "scale-params", ScaleParams { len: len as u32, scale, _p0: 0, _p1: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.scale_inplace, &entries, (ceil_div(len, 64), 1, 1));
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamParams {
    len: u32,
    lr: f32,
    bias1: f32,
    bias2: f32,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_adam(ctx: &GpuContext, w: &wgpu::Buffer, g: &wgpu::Buffer, m: &wgpu::Buffer, v: &wgpu::Buffer, lr: f32, bias1: f32, bias2: f32, len: usize) {
    let params = uniform(&ctx.device, "adam-params", AdamParams { len: len as u32, lr, bias1, bias2 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: g.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: m.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: v.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.adam_update, &entries, (ceil_div(len, 64), 1, 1));
}

// --- Buffer sets ---------------------------------------------------------

/// The ten per-layer weight/gradient/optimizer-state tensors, in the same
/// fixed order `llm_core::model::LayerWeights::tensors()` uses — every
/// `GpuLayerTensors` (weights, gradients, Adam m, Adam v) shares this
/// shape, so the training loop can walk them all generically.
struct GpuLayerTensors {
    ple: wgpu::Buffer,
    attn_norm_gain: wgpu::Buffer,
    wq: wgpu::Buffer,
    wk: wgpu::Buffer,
    wv: wgpu::Buffer,
    wo: wgpu::Buffer,
    mlp_norm_gain: wgpu::Buffer,
    w_gate: wgpu::Buffer,
    w_up: wgpu::Buffer,
    w_down: wgpu::Buffer,
}

fn layer_tensor_lens(h: usize, ffn: usize, vocab: usize) -> [usize; 10] {
    [vocab * h, h, h * h, h * h, h * h, h * h, h, ffn * h, ffn * h, h * ffn]
}

impl GpuLayerTensors {
    fn zeros(device: &wgpu::Device, label: &str, h: usize, ffn: usize, vocab: usize) -> Self {
        let lens = layer_tensor_lens(h, ffn, vocab);
        Self {
            ple: storage_f32(device, &format!("{label}.ple"), lens[0], false),
            attn_norm_gain: storage_f32(device, &format!("{label}.attn_norm_gain"), lens[1], false),
            wq: storage_f32(device, &format!("{label}.wq"), lens[2], false),
            wk: storage_f32(device, &format!("{label}.wk"), lens[3], false),
            wv: storage_f32(device, &format!("{label}.wv"), lens[4], false),
            wo: storage_f32(device, &format!("{label}.wo"), lens[5], false),
            mlp_norm_gain: storage_f32(device, &format!("{label}.mlp_norm_gain"), lens[6], false),
            w_gate: storage_f32(device, &format!("{label}.w_gate"), lens[7], false),
            w_up: storage_f32(device, &format!("{label}.w_up"), lens[8], false),
            w_down: storage_f32(device, &format!("{label}.w_down"), lens[9], false),
        }
    }

    fn buffers(&self) -> [&wgpu::Buffer; 10] {
        [
            &self.ple,
            &self.attn_norm_gain,
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.mlp_norm_gain,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
        ]
    }
}

struct GpuLayer {
    w: GpuLayerTensors,
}

/// Per-layer activations cached during the forward pass, needed by the
/// backward pass — mirrors `llm_core::model::LayerCache` field for field.
struct GpuLayerCache {
    h_after_ple: wgpu::Buffer,
    normed1: wgpu::Buffer,
    inv_rms1: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    probs: wgpu::Buffer,
    concat: wgpu::Buffer,
    h_after_attn: wgpu::Buffer,
    normed2: wgpu::Buffer,
    inv_rms2: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
}

impl GpuLayerCache {
    fn new(device: &wgpu::Device, ctx_len: usize, heads: usize, h: usize, ffn: usize) -> Self {
        Self {
            h_after_ple: storage_f32(device, "cache.h_after_ple", ctx_len * h, false),
            normed1: storage_f32(device, "cache.normed1", ctx_len * h, false),
            inv_rms1: storage_f32(device, "cache.inv_rms1", ctx_len, false),
            q: storage_f32(device, "cache.q", ctx_len * h, false),
            k: storage_f32(device, "cache.k", ctx_len * h, false),
            v: storage_f32(device, "cache.v", ctx_len * h, false),
            probs: storage_f32(device, "cache.probs", heads * ctx_len * ctx_len, false),
            concat: storage_f32(device, "cache.concat", ctx_len * h, false),
            h_after_attn: storage_f32(device, "cache.h_after_attn", ctx_len * h, false),
            normed2: storage_f32(device, "cache.normed2", ctx_len * h, false),
            inv_rms2: storage_f32(device, "cache.inv_rms2", ctx_len, false),
            gate: storage_f32(device, "cache.gate", ctx_len * ffn, false),
            up: storage_f32(device, "cache.up", ctx_len * ffn, false),
        }
    }
}

/// A trained model resident on the GPU: weights, gradients, Adam state,
/// and forward/backward scratch (sized once for `config.context_len`,
/// reused across every call). See the module docs for the "one shared
/// forward pass" and "no atomics" design notes.
pub struct GpuModel {
    config: ModelConfig,

    embed: wgpu::Buffer,
    layers: Vec<GpuLayer>,
    final_norm_gain: wgpu::Buffer,

    grad_embed: wgpu::Buffer,
    grad_layers: Vec<GpuLayerTensors>,
    grad_final_norm_gain: wgpu::Buffer,

    grad_accum_embed: wgpu::Buffer,
    grad_accum_layers: Vec<GpuLayerTensors>,
    grad_accum_final_norm_gain: wgpu::Buffer,

    adam_m_embed: wgpu::Buffer,
    adam_m_layers: Vec<GpuLayerTensors>,
    adam_m_final_norm_gain: wgpu::Buffer,
    adam_v_embed: wgpu::Buffer,
    adam_v_layers: Vec<GpuLayerTensors>,
    adam_v_final_norm_gain: wgpu::Buffer,
    adam_step: std::cell::Cell<u32>,

    ids: wgpu::Buffer,
    targets: wgpu::Buffer,
    hidden: wgpu::Buffer,
    ple_scratch: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    mlp_out: wgpu::Buffer,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    loss_per_row: wgpu::Buffer,
    layer_caches: Vec<GpuLayerCache>,
    h_final: wgpu::Buffer,
    final_normed: wgpu::Buffer,
    final_inv_rms: wgpu::Buffer,

    // Backward scratch, reused across layers and across the batch loop.
    d_hidden: wgpu::Buffer,
    d_normed: wgpu::Buffer,
    d_normed_tmp: wgpu::Buffer,
    d_q: wgpu::Buffer,
    d_k: wgpu::Buffer,
    d_v: wgpu::Buffer,
    d_score_scratch: wgpu::Buffer,
    d_concat: wgpu::Buffer,
    d_gate: wgpu::Buffer,
    d_up: wgpu::Buffer,
    d_act: wgpu::Buffer,
    d_logits: wgpu::Buffer,
}

/// Whether `config` fits this backend's naive attention kernel *and* its
/// dense `[heads, context_len, context_len]` probs cache (needed for
/// training's backward pass, not just the attention window) — see
/// `shaders/attention.wgsl`. Callers should fall back to the CPU backend
/// when this is `false` instead of trying to construct a `GpuModel`.
pub fn supports(config: &ModelConfig) -> bool {
    config.effective_window() <= MAX_GPU_WINDOW && config.context_len <= MAX_GPU_WINDOW
}

impl GpuModel {
    pub fn upload(ctx: &GpuContext, weights: &ModelWeights, config: &ModelConfig) -> Result<Self, String> {
        if !supports(config) {
            return Err(format!(
                "context_len ({}) or local_window ({}) exceeds this GPU backend's limit ({MAX_GPU_WINDOW}); use the CPU backend for this config",
                config.context_len,
                config.effective_window()
            ));
        }
        let device = &ctx.device;
        let h = config.hidden_dim;
        let heads = config.num_heads;
        let ffn = config.ffn_dim();
        let vocab = config.vocab_size();
        let ctx_len = config.context_len;

        let embed = upload_f32(device, "embed", &weights.embed, wgpu::BufferUsages::empty());
        let layers = weights
            .layers
            .iter()
            .map(|l| GpuLayer {
                w: GpuLayerTensors {
                    ple: upload_f32(device, "ple", &l.ple, wgpu::BufferUsages::empty()),
                    attn_norm_gain: upload_f32(device, "attn_norm_gain", &l.attn_norm_gain, wgpu::BufferUsages::empty()),
                    wq: upload_f32(device, "wq", &l.wq, wgpu::BufferUsages::empty()),
                    wk: upload_f32(device, "wk", &l.wk, wgpu::BufferUsages::empty()),
                    wv: upload_f32(device, "wv", &l.wv, wgpu::BufferUsages::empty()),
                    wo: upload_f32(device, "wo", &l.wo, wgpu::BufferUsages::empty()),
                    mlp_norm_gain: upload_f32(device, "mlp_norm_gain", &l.mlp_norm_gain, wgpu::BufferUsages::empty()),
                    w_gate: upload_f32(device, "w_gate", &l.w_gate, wgpu::BufferUsages::empty()),
                    w_up: upload_f32(device, "w_up", &l.w_up, wgpu::BufferUsages::empty()),
                    w_down: upload_f32(device, "w_down", &l.w_down, wgpu::BufferUsages::empty()),
                },
            })
            .collect::<Vec<_>>();
        let final_norm_gain = upload_f32(device, "final_norm_gain", &weights.final_norm_gain, wgpu::BufferUsages::empty());

        let num_layers = weights.layers.len();
        let mk_layer_tensors = |label: &str| -> Vec<GpuLayerTensors> {
            (0..num_layers).map(|_| GpuLayerTensors::zeros(device, label, h, ffn, vocab)).collect()
        };

        Ok(Self {
            config: *config,
            embed,
            layers,
            final_norm_gain,

            grad_embed: storage_f32(device, "grad_embed", vocab * h, false),
            grad_layers: mk_layer_tensors("grad"),
            grad_final_norm_gain: storage_f32(device, "grad_final_norm_gain", h, false),

            grad_accum_embed: storage_f32(device, "grad_accum_embed", vocab * h, false),
            grad_accum_layers: mk_layer_tensors("grad_accum"),
            grad_accum_final_norm_gain: storage_f32(device, "grad_accum_final_norm_gain", h, false),

            adam_m_embed: storage_f32(device, "adam_m_embed", vocab * h, false),
            adam_m_layers: mk_layer_tensors("adam_m"),
            adam_m_final_norm_gain: storage_f32(device, "adam_m_final_norm_gain", h, false),
            adam_v_embed: storage_f32(device, "adam_v_embed", vocab * h, false),
            adam_v_layers: mk_layer_tensors("adam_v"),
            adam_v_final_norm_gain: storage_f32(device, "adam_v_final_norm_gain", h, false),
            adam_step: std::cell::Cell::new(0),

            ids: upload_u32(device, "ids", &vec![0u32; ctx_len]),
            targets: upload_u32(device, "targets", &vec![0u32; ctx_len]),
            hidden: storage_f32(device, "hidden", ctx_len * h, false),
            ple_scratch: storage_f32(device, "ple_scratch", ctx_len * h, false),
            attn_out: storage_f32(device, "attn_out", ctx_len * h, false),
            mlp_out: storage_f32(device, "mlp_out", ctx_len * h, false),
            act: storage_f32(device, "act", ctx_len * ffn, false),
            logits: storage_f32(device, "logits", ctx_len * vocab, true),
            loss_per_row: storage_f32(device, "loss_per_row", ctx_len, true),
            layer_caches: (0..num_layers).map(|_| GpuLayerCache::new(device, ctx_len, heads, h, ffn)).collect(),
            h_final: storage_f32(device, "h_final", ctx_len * h, false),
            final_normed: storage_f32(device, "final_normed", ctx_len * h, false),
            final_inv_rms: storage_f32(device, "final_inv_rms", ctx_len, false),

            d_hidden: storage_f32(device, "d_hidden", ctx_len * h, false),
            d_normed: storage_f32(device, "d_normed", ctx_len * h, false),
            d_normed_tmp: storage_f32(device, "d_normed_tmp", ctx_len * h, false),
            d_q: storage_f32(device, "d_q", ctx_len * h, false),
            d_k: storage_f32(device, "d_k", ctx_len * h, false),
            d_v: storage_f32(device, "d_v", ctx_len * h, false),
            d_score_scratch: storage_f32(device, "d_score_scratch", heads * ctx_len * ctx_len, false),
            d_concat: storage_f32(device, "d_concat", ctx_len * h, false),
            d_gate: storage_f32(device, "d_gate", ctx_len * ffn, false),
            d_up: storage_f32(device, "d_up", ctx_len * ffn, false),
            d_act: storage_f32(device, "d_act", ctx_len * ffn, false),
            d_logits: storage_f32(device, "d_logits", ctx_len * vocab, false),
        })
    }

    /// Runs the forward pass over `tokens`, populating `self.hidden`,
    /// `self.logits`, and every per-layer cache buffer.
    fn forward(&self, ctx: &GpuContext, tokens: &[u32]) -> usize {
        let t_len = tokens.len();
        let h = self.config.hidden_dim;
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let window = self.config.effective_window();
        let ffn = self.config.ffn_dim();
        let vocab = self.config.vocab_size();

        write_u32(&ctx.queue, &self.ids, tokens);
        dispatch_gather(ctx, &self.embed, &self.ids, &self.hidden, t_len, h);

        for (layer, lc) in self.layers.iter().zip(&self.layer_caches) {
            dispatch_gather(ctx, &layer.w.ple, &self.ids, &self.ple_scratch, t_len, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.ple_scratch, t_len * h);
            // hidden now holds h_after_ple for this layer; snapshot it into the cache.
            copy_buffer(ctx, &self.hidden, &lc.h_after_ple, t_len * h);

            dispatch_rmsnorm(ctx, &lc.h_after_ple, &layer.w.attn_norm_gain, &lc.normed1, &lc.inv_rms1, t_len, h);
            dispatch_linear(ctx, &lc.normed1, &layer.w.wq, &lc.q, t_len, h, h);
            dispatch_linear(ctx, &lc.normed1, &layer.w.wk, &lc.k, t_len, h, h);
            dispatch_linear(ctx, &lc.normed1, &layer.w.wv, &lc.v, t_len, h, h);
            dispatch_rope(ctx, &lc.q, t_len, heads, head_dim, false);
            dispatch_rope(ctx, &lc.k, t_len, heads, head_dim, false);
            dispatch_attention(ctx, &lc.q, &lc.k, &lc.v, &lc.concat, &lc.probs, t_len, heads, head_dim, window);
            dispatch_linear(ctx, &lc.concat, &layer.w.wo, &self.attn_out, t_len, h, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.attn_out, t_len * h);
            copy_buffer(ctx, &self.hidden, &lc.h_after_attn, t_len * h);

            dispatch_rmsnorm(ctx, &lc.h_after_attn, &layer.w.mlp_norm_gain, &lc.normed2, &lc.inv_rms2, t_len, h);
            dispatch_linear(ctx, &lc.normed2, &layer.w.w_gate, &lc.gate, t_len, h, ffn);
            dispatch_linear(ctx, &lc.normed2, &layer.w.w_up, &lc.up, t_len, h, ffn);
            dispatch_swiglu(ctx, &lc.gate, &lc.up, &self.act, t_len * ffn);
            dispatch_linear(ctx, &self.act, &layer.w.w_down, &self.mlp_out, t_len, ffn, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.mlp_out, t_len * h);
        }

        copy_buffer(ctx, &self.hidden, &self.h_final, t_len * h);
        dispatch_rmsnorm(ctx, &self.h_final, &self.final_norm_gain, &self.final_normed, &self.final_inv_rms, t_len, h);
        // Weight-tied output head: logits = final_normed @ embed^T.
        dispatch_linear(ctx, &self.final_normed, &self.embed, &self.logits, t_len, h, vocab);

        t_len
    }

    /// Runs the forward pass over `tokens` (`1..=config.context_len` of
    /// them) and returns just the last position's logits (`[vocab]`) —
    /// all a caller doing next-token sampling needs. Reuses this model's
    /// scratch buffers, so it's not safe to call concurrently with itself.
    pub async fn forward_last_logits(&self, ctx: &GpuContext, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let t_len = tokens.len();
        if t_len == 0 || t_len > self.config.context_len {
            return Err(format!("tokens.len()={t_len} must be in 1..={}", self.config.context_len));
        }
        let vocab = self.config.vocab_size();
        self.forward(ctx, tokens);
        let all_logits = read_f32(&ctx.device, &ctx.queue, &self.logits, t_len * vocab).await;
        Ok(all_logits[(t_len - 1) * vocab..t_len * vocab].to_vec())
    }

    /// Zeroes every gradient buffer in `bufs` (paired with `lens`).
    fn zero_all(&self, ctx: &GpuContext, bufs: &[&wgpu::Buffer], lens: &[usize]) {
        for (buf, &len) in bufs.iter().zip(lens) {
            dispatch_zero(ctx, buf, len);
        }
    }

    fn tensor_lens(&self) -> Vec<usize> {
        let h = self.config.hidden_dim;
        let ffn = self.config.ffn_dim();
        let vocab = self.config.vocab_size();
        let mut lens = vec![vocab * h];
        for _ in &self.layers {
            lens.extend(layer_tensor_lens(h, ffn, vocab));
        }
        lens.push(h);
        lens
    }

    fn weight_buffers(&self) -> Vec<&wgpu::Buffer> {
        let mut v = vec![&self.embed];
        for l in &self.layers {
            v.extend(l.w.buffers());
        }
        v.push(&self.final_norm_gain);
        v
    }
    fn grad_buffers(&self) -> Vec<&wgpu::Buffer> {
        let mut v = vec![&self.grad_embed];
        for l in &self.grad_layers {
            v.extend(l.buffers());
        }
        v.push(&self.grad_final_norm_gain);
        v
    }
    fn grad_accum_buffers(&self) -> Vec<&wgpu::Buffer> {
        let mut v = vec![&self.grad_accum_embed];
        for l in &self.grad_accum_layers {
            v.extend(l.buffers());
        }
        v.push(&self.grad_accum_final_norm_gain);
        v
    }
    fn adam_m_buffers(&self) -> Vec<&wgpu::Buffer> {
        let mut v = vec![&self.adam_m_embed];
        for l in &self.adam_m_layers {
            v.extend(l.buffers());
        }
        v.push(&self.adam_m_final_norm_gain);
        v
    }
    fn adam_v_buffers(&self) -> Vec<&wgpu::Buffer> {
        let mut v = vec![&self.adam_v_embed];
        for l in &self.adam_v_layers {
            v.extend(l.buffers());
        }
        v.push(&self.adam_v_final_norm_gain);
        v
    }

    /// Backward pass for the sequence most recently run through
    /// `forward`, given `d_logits` already populated (see
    /// `dispatch_cross_entropy`). Fills `self.grad_*` (assumed
    /// zeroed beforehand by the caller). Mirrors `llm_core::model::backward`.
    fn backward(&self, ctx: &GpuContext, tokens: &[u32]) {
        let t_len = tokens.len();
        let h = self.config.hidden_dim;
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let window = self.config.effective_window();
        let ffn = self.config.ffn_dim();
        let vocab = self.config.vocab_size();

        // Output head (tied with embed): logits = final_normed @ embed^T.
        dispatch_linear_bwd(ctx, &self.d_logits, &self.final_normed, &self.embed, &self.d_normed, &self.grad_embed, t_len, h, vocab);

        dispatch_rmsnorm_bwd(ctx, &self.d_normed, &self.h_final, &self.final_norm_gain, &self.final_inv_rms, &self.d_hidden, &self.grad_final_norm_gain, t_len, h);

        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let lc = &self.layer_caches[layer_idx];
            let lg = &self.grad_layers[layer_idx];

            // --- MLP branch (residual: h_after_attn + mlp_out) ---
            // d_mlp_out = d_hidden (residual splits equally); act = swiglu(gate,up).
            dispatch_swiglu(ctx, &lc.gate, &lc.up, &self.act, t_len * ffn);
            dispatch_linear_bwd(ctx, &self.d_hidden, &self.act, &layer.w.w_down, &self.d_act, &lg.w_down, t_len, ffn, h);
            dispatch_swiglu_bwd(ctx, &self.d_act, &lc.gate, &lc.up, &self.d_gate, &self.d_up, t_len * ffn);
            dispatch_linear_bwd(ctx, &self.d_gate, &lc.normed2, &layer.w.w_gate, &self.d_normed, &lg.w_gate, t_len, h, ffn);
            dispatch_linear_bwd(ctx, &self.d_up, &lc.normed2, &layer.w.w_up, &self.d_normed_tmp, &lg.w_up, t_len, h, ffn);
            dispatch_add_inplace(ctx, &self.d_normed, &self.d_normed_tmp, t_len * h);

            dispatch_rmsnorm_bwd(ctx, &self.d_normed, &lc.h_after_attn, &layer.w.mlp_norm_gain, &lc.inv_rms2, &self.d_normed_tmp, &lg.mlp_norm_gain, t_len, h);
            // d_h_after_attn = d_hidden (pass-through) + d_normed_tmp (norm branch); accumulate into d_hidden in place.
            dispatch_add_inplace(ctx, &self.d_hidden, &self.d_normed_tmp, t_len * h);

            // --- Attention branch (residual: h_after_ple + attn_out) ---
            dispatch_linear_bwd(ctx, &self.d_hidden, &lc.concat, &layer.w.wo, &self.d_concat, &lg.wo, t_len, h, h);
            dispatch_attention_bwd(
                ctx,
                &self.d_concat,
                &lc.q,
                &lc.k,
                &lc.v,
                &lc.probs,
                &self.d_score_scratch,
                &self.d_q,
                &self.d_k,
                &self.d_v,
                t_len,
                heads,
                head_dim,
                window,
            );
            dispatch_rope(ctx, &self.d_q, t_len, heads, head_dim, true);
            dispatch_rope(ctx, &self.d_k, t_len, heads, head_dim, true);

            dispatch_linear_bwd(ctx, &self.d_q, &lc.normed1, &layer.w.wq, &self.d_normed, &lg.wq, t_len, h, h);
            dispatch_linear_bwd(ctx, &self.d_k, &lc.normed1, &layer.w.wk, &self.d_normed_tmp, &lg.wk, t_len, h, h);
            dispatch_add_inplace(ctx, &self.d_normed, &self.d_normed_tmp, t_len * h);
            dispatch_linear_bwd(ctx, &self.d_v, &lc.normed1, &layer.w.wv, &self.d_normed_tmp, &lg.wv, t_len, h, h);
            dispatch_add_inplace(ctx, &self.d_normed, &self.d_normed_tmp, t_len * h);

            dispatch_rmsnorm_bwd(ctx, &self.d_normed, &lc.h_after_ple, &layer.w.attn_norm_gain, &lc.inv_rms1, &self.d_normed_tmp, &lg.attn_norm_gain, t_len, h);
            // d_h_after_ple = d_hidden (pass-through) + d_normed_tmp (norm branch).
            dispatch_add_inplace(ctx, &self.d_hidden, &self.d_normed_tmp, t_len * h);

            // PLE residual add: gradient passes through unchanged (already
            // in d_hidden) and also scatters into this layer's PLE grad.
            dispatch_embedding_scatter_add(ctx, &self.d_hidden, &self.ids, &lg.ple, t_len, h, vocab);
        }

        // Input embedding gather (the other half of the tied embed/head gradient).
        dispatch_embedding_scatter_add(ctx, &self.d_hidden, &self.ids, &self.grad_embed, t_len, h, vocab);
    }

    /// Dev/sanity-check tool: runs forward, cross-entropy, and backward
    /// for one sequence (no batching, no Adam step — this never touches
    /// the weights) and reads back the embedding-table gradient. The
    /// embedding gradient depends on nearly the entire backward pass
    /// (the output head, every layer's attention/MLP backward, every
    /// layer's PLE scatter, and the input embedding scatter all feed
    /// into it), so comparing this one buffer against
    /// `llm_core::model::backward(...).embed` for the same weights/
    /// tokens/targets is a strong end-to-end check of this crate's
    /// backward pass — see `wasm-app`'s `debug_compare_gpu_cpu_gradient`,
    /// which does exactly that comparison.
    pub async fn debug_grad_embed(&self, ctx: &GpuContext, tokens: &[u32], targets: &[u32]) -> Result<Vec<f32>, String> {
        let t_len = tokens.len();
        if t_len == 0 || t_len != self.config.context_len {
            return Err(format!("tokens.len()={t_len} must equal this model's context_len ({})", self.config.context_len));
        }
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size();

        let lens = self.tensor_lens();
        self.zero_all(ctx, &self.grad_buffers(), &lens);

        self.forward(ctx, tokens);
        write_u32(&ctx.queue, &self.targets, targets);
        dispatch_cross_entropy(ctx, &self.logits, &self.targets, &self.d_logits, &self.loss_per_row, t_len);
        self.backward(ctx, tokens);

        Ok(read_f32(&ctx.device, &ctx.queue, &self.grad_embed, vocab * h).await)
    }

    /// Samples nothing itself (the caller already sampled `batch` via
    /// `llm_core::corpus::Corpus::sample_batch`) — runs forward, cross
    /// entropy, backward, and one Adam step for the whole batch. Returns
    /// the batch's mean loss. Mirrors `llm_core::train::Trainer::train_step`.
    pub async fn train_step(&self, ctx: &GpuContext, batch: &Batch, lr: f32) -> Result<f32, String> {
        if batch.context_len != self.config.context_len {
            return Err(format!(
                "batch context_len ({}) doesn't match this model's context_len ({})",
                batch.context_len, self.config.context_len
            ));
        }
        let t_len = batch.context_len;

        let lens = self.tensor_lens();
        self.zero_all(ctx, &self.grad_accum_buffers(), &lens);

        let mut total_loss = 0.0f32;
        for b in 0..batch.batch_size {
            let start = b * t_len;
            let input = &batch.inputs[start..start + t_len];
            let target = &batch.targets[start..start + t_len];

            self.zero_all(ctx, &self.grad_buffers(), &lens);

            self.forward(ctx, input);
            write_u32(&ctx.queue, &self.targets, target);
            dispatch_cross_entropy(ctx, &self.logits, &self.targets, &self.d_logits, &self.loss_per_row, t_len);
            let row_losses = read_f32(&ctx.device, &ctx.queue, &self.loss_per_row, t_len).await;
            total_loss += row_losses.iter().sum::<f32>() / t_len as f32;

            self.backward(ctx, input);

            // grad_accum_buffers()/grad_buffers() are built from the same
            // shape list as `lens`, so all three zip up in lockstep.
            for ((accum, g), &len) in self.grad_accum_buffers().into_iter().zip(self.grad_buffers()).zip(&lens) {
                dispatch_add_inplace(ctx, accum, g, len);
            }
        }

        for (buf, &len) in self.grad_accum_buffers().into_iter().zip(&lens) {
            dispatch_scale(ctx, buf, 1.0 / batch.batch_size as f32, len);
        }

        let step = self.adam_step.get() + 1;
        self.adam_step.set(step);
        let bias1 = 1.0 - 0.9f32.powi(step as i32);
        let bias2 = 1.0 - 0.999f32.powi(step as i32);
        let weight_bufs = self.weight_buffers();
        let grad_bufs = self.grad_accum_buffers();
        let m_bufs = self.adam_m_buffers();
        let v_bufs = self.adam_v_buffers();
        for i in 0..weight_bufs.len() {
            dispatch_adam(ctx, weight_bufs[i], grad_bufs[i], m_bufs[i], v_bufs[i], lr, bias1, bias2, lens[i]);
        }

        Ok(total_loss / batch.batch_size as f32)
    }

    /// Reads every weight tensor back from the GPU, concatenated in the
    /// same fixed order `llm_core::model::ModelWeights::to_bytes` uses —
    /// pass the result through that same byte layout (little-endian f32)
    /// to `ModelWeights::from_bytes` to get an owned CPU copy. Used to
    /// sync weights this backend trained back to the canonical CPU copy
    /// (see `wasm-app`'s `sync_weights_from_gpu`), since `train_step`
    /// only ever updates the GPU-resident copy.
    pub async fn read_all_weights(&self, ctx: &GpuContext) -> Vec<f32> {
        let lens = self.tensor_lens();
        let mut out = Vec::with_capacity(lens.iter().sum());
        for (buf, &len) in self.weight_buffers().into_iter().zip(&lens) {
            out.extend(read_f32(&ctx.device, &ctx.queue, buf, len).await);
        }
        out
    }
}

/// Copies `len` f32s from `src` to `dst` via a GPU-side command (no
/// host round trip) — used to snapshot the running `hidden` buffer into a
/// per-layer cache slot without a separate shader (a `copy_buffer_to_buffer`
/// command covers it).
fn copy_buffer(ctx: &GpuContext, src: &wgpu::Buffer, dst: &wgpu::Buffer, len: usize) {
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cache-copy") });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));
}
