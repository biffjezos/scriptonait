//! Uploads a trained `llm_core::model::ModelWeights` to the GPU once, then
//! runs the forward pass entirely on-device for fast interactive
//! generation. This mirrors `llm_core::model::forward` step for step —
//! see that function for the reference this must match — but only the
//! forward direction: there is no backward pass here (training stays on
//! the CPU/wasm path; see the crate-level docs for why).

use llm_core::config::ModelConfig;
use llm_core::model::ModelWeights;

use crate::buffers::{read_f32, storage_f32, uniform, upload_f32, upload_u32, write_u32, write_uniform};
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

fn dispatch(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
    workgroups: (u32, u32, u32),
) {
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries,
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    ctx.queue.submit(Some(encoder.finish()));
}

fn dispatch_linear(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) {
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

fn dispatch_rmsnorm(ctx: &GpuContext, x: &wgpu::Buffer, gain: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, dim: usize) {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        rows: u32,
        dim: u32,
        eps: f32,
        _pad: u32,
    }
    let params = uniform(&ctx.device, "rmsnorm-params", Params { rows: rows as u32, dim: dim as u32, eps: 1e-6, _pad: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: gain.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
    ];
    dispatch(ctx, &ctx.pipelines.rmsnorm, &entries, (ceil_div(rows, 64), 1, 1));
}

fn dispatch_rope(ctx: &GpuContext, x: &wgpu::Buffer, t_len: usize, heads: usize, head_dim: usize) {
    let params = uniform(&ctx.device, "rope-params", P4 { a: t_len as u32, b: heads as u32, c: head_dim as u32, d: 0 });
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

struct GpuLayer {
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

/// A trained model's weights, resident on the GPU, plus scratch buffers
/// (sized once for `config.context_len`) reused across every generation
/// call.
pub struct GpuModel {
    config: ModelConfig,
    embed: wgpu::Buffer,
    layers: Vec<GpuLayer>,
    final_norm_gain: wgpu::Buffer,

    ids: wgpu::Buffer,
    hidden: wgpu::Buffer,
    ple_scratch: wgpu::Buffer,
    normed: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    concat: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    act: wgpu::Buffer,
    mlp_out: wgpu::Buffer,
    logits: wgpu::Buffer,
}

/// Whether `config` is small enough for this backend's naive attention
/// kernel (see `shaders/attention.wgsl`). Callers should fall back to the
/// CPU backend (`llm_core::generate::generate`) when this is `false`
/// instead of trying to construct a `GpuModel`.
pub fn supports(config: &ModelConfig) -> bool {
    config.effective_window() <= MAX_GPU_WINDOW
}

impl GpuModel {
    pub fn upload(ctx: &GpuContext, weights: &ModelWeights, config: &ModelConfig) -> Result<Self, String> {
        if !supports(config) {
            return Err(format!(
                "local_window ({}) exceeds this GPU backend's limit ({MAX_GPU_WINDOW}); use the CPU backend for this config",
                config.effective_window()
            ));
        }
        let device = &ctx.device;
        let h = config.hidden_dim;
        let ffn = config.ffn_dim();
        let vocab = config.vocab_size();
        let ctx_len = config.context_len;

        let embed = upload_f32(device, "embed", &weights.embed, wgpu::BufferUsages::empty());
        let layers = weights
            .layers
            .iter()
            .map(|l| GpuLayer {
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
            })
            .collect();
        let final_norm_gain = upload_f32(device, "final_norm_gain", &weights.final_norm_gain, wgpu::BufferUsages::empty());

        Ok(Self {
            config: *config,
            embed,
            layers,
            final_norm_gain,
            ids: upload_u32(device, "ids", &vec![0u32; ctx_len]),
            hidden: storage_f32(device, "hidden", ctx_len * h, false),
            ple_scratch: storage_f32(device, "ple_scratch", ctx_len * h, false),
            normed: storage_f32(device, "normed", ctx_len * h, false),
            q: storage_f32(device, "q", ctx_len * h, false),
            k: storage_f32(device, "k", ctx_len * h, false),
            v: storage_f32(device, "v", ctx_len * h, false),
            concat: storage_f32(device, "concat", ctx_len * h, false),
            attn_out: storage_f32(device, "attn_out", ctx_len * h, false),
            gate: storage_f32(device, "gate", ctx_len * ffn, false),
            up: storage_f32(device, "up", ctx_len * ffn, false),
            act: storage_f32(device, "act", ctx_len * ffn, false),
            mlp_out: storage_f32(device, "mlp_out", ctx_len * h, false),
            logits: storage_f32(device, "logits", ctx_len * vocab, true),
        })
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
        let h = self.config.hidden_dim;
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let window = self.config.effective_window();
        let ffn = self.config.ffn_dim();
        let vocab = self.config.vocab_size();

        write_u32(&ctx.queue, &self.ids, tokens);
        dispatch_gather(ctx, &self.embed, &self.ids, &self.hidden, t_len, h);

        for layer in &self.layers {
            dispatch_gather(ctx, &layer.ple, &self.ids, &self.ple_scratch, t_len, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.ple_scratch, t_len * h);

            dispatch_rmsnorm(ctx, &self.hidden, &layer.attn_norm_gain, &self.normed, t_len, h);
            dispatch_linear(ctx, &self.normed, &layer.wq, &self.q, t_len, h, h);
            dispatch_linear(ctx, &self.normed, &layer.wk, &self.k, t_len, h, h);
            dispatch_linear(ctx, &self.normed, &layer.wv, &self.v, t_len, h, h);
            dispatch_rope(ctx, &self.q, t_len, heads, head_dim);
            dispatch_rope(ctx, &self.k, t_len, heads, head_dim);
            dispatch_attention(ctx, &self.q, &self.k, &self.v, &self.concat, t_len, heads, head_dim, window);
            dispatch_linear(ctx, &self.concat, &layer.wo, &self.attn_out, t_len, h, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.attn_out, t_len * h);

            dispatch_rmsnorm(ctx, &self.hidden, &layer.mlp_norm_gain, &self.normed, t_len, h);
            dispatch_linear(ctx, &self.normed, &layer.w_gate, &self.gate, t_len, h, ffn);
            dispatch_linear(ctx, &self.normed, &layer.w_up, &self.up, t_len, h, ffn);
            dispatch_swiglu(ctx, &self.gate, &self.up, &self.act, t_len * ffn);
            dispatch_linear(ctx, &self.act, &layer.w_down, &self.mlp_out, t_len, ffn, h);
            dispatch_add_inplace(ctx, &self.hidden, &self.mlp_out, t_len * h);
        }

        dispatch_rmsnorm(ctx, &self.hidden, &self.final_norm_gain, &self.normed, t_len, h);
        // Weight-tied output head: logits = final_normed @ embed^T (same
        // buffer used both as the input embedding table and here).
        dispatch_linear(ctx, &self.normed, &self.embed, &self.logits, t_len, h, vocab);

        let all_logits = read_f32(&ctx.device, &ctx.queue, &self.logits, t_len * vocab).await;
        Ok(all_logits[(t_len - 1) * vocab..t_len * vocab].to_vec())
    }
}
