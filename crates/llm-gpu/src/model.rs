//! WebGPU inference: the model's weights live on the GPU and each
//! generated token is decoded there.
//!
//! ## What runs where, and why
//!
//! The prompt is *prefilled on the CPU*, by `llm_core::model::prefill` —
//! the same gradient-checked forward pass everything else uses — and the
//! keys and values it produced are uploaded into this backend's cache.
//! Only the per-token decode step runs here.
//!
//! That split is deliberate. A batched forward pass in WGSL is a lot of
//! code that computes something already computed correctly elsewhere,
//! and it is the part that cannot be checked without a GPU in front of
//! you. The decode step is where the time actually goes (one prompt,
//! hundreds of tokens), it is a much smaller kernel set, and being able
//! to compare its output against the CPU's for the same input makes it
//! checkable — see `debug_compare_step`.
//!
//! ## The cache is a ring
//!
//! Position `p` is written to slot `p % capacity`, where capacity is the
//! attention window. Nothing shifts, ever. This is safe because softmax
//! is order-independent and RoPE has already written each key's position
//! into the key itself, so "the last `window` keys" is the same set of
//! numbers whichever order they sit in.

use llm_core::config::ModelConfig;
use llm_core::model::{GenCache, ModelWeights};

use crate::buffers;
use crate::context::{GpuContext, Kernel};

/// Largest head dimension the decode kernel's per-thread accumulator can
/// hold; see `shaders/attention_decode.wgsl`.
pub const MAX_HEAD_DIM: usize = 256;

/// Whether this backend can run `config`.
///
/// Notably this no longer refuses long contexts. The old kernel kept a
/// fixed 256-entry array of attention scores per thread, so any window
/// past that fell back to the CPU; the streaming softmax in
/// `attention_decode.wgsl` needs no such array.
pub fn supports(config: &ModelConfig) -> bool {
    config.head_dim() <= MAX_HEAD_DIM
        && config.head_dim() % 2 == 0
        && config.num_heads % config.num_kv_heads == 0
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct P4 {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

/// The eight-word parameter block, for kernels that need more than four
/// values. A uniform struct's size has to be a multiple of 16 bytes, so
/// the choices are four words or eight; the pool's slots are sized for
/// the larger one.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct P8 {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub e: u32,
    pub f: u32,
    pub g: u32,
    pub h: u32,
}

pub(crate) fn ceil_div(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

pub(crate) fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    kernel: &Kernel,
    entries: &[wgpu::BindGroupEntry],
    workgroups: (u32, u32, u32),
) {
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &kernel.layout,
        entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&kernel.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups.0.max(1), workgroups.1.max(1), workgroups.2.max(1));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_linear(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
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
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
    ];
    // gid.x indexes out_dim, gid.y indexes rows — matches
    // shaders/linear.wgsl's dispatch convention.
    dispatch(
        encoder,
        ctx,
        &ctx.pipelines.linear,
        &entries,
        (ceil_div(out_dim, 16), ceil_div(rows, 16), 1),
    );
}

pub(crate) fn dispatch_add_inplace(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    dst: &wgpu::Buffer,
    src: &wgpu::Buffer,
    len: usize,
) {
    let params = ctx.params.alloc(&ctx.device, &ctx.queue, P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: src.as_entire_binding() },
    ];
    dispatch(encoder, ctx, &ctx.pipelines.add_inplace, &entries, (ceil_div(len, 64), 1, 1));
}

fn dispatch_gather(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    table: &wgpu::Buffer,
    ids: &wgpu::Buffer,
    out: &wgpu::Buffer,
    hidden: usize,
) {
    let params = ctx.params.alloc(&ctx.device, &ctx.queue, P4 { a: 1, b: hidden as u32, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: table.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: ids.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
    ];
    dispatch(encoder, ctx, &ctx.pipelines.embedding_gather, &entries, (1, ceil_div(hidden, 8), 1));
}

fn dispatch_rmsnorm(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    gain: &wgpu::Buffer,
    out: &wgpu::Buffer,
    inv_rms: &wgpu::Buffer,
    hidden: usize,
) {
    let params = ctx.params.alloc(&ctx.device, &ctx.queue, P4 { a: 1, b: hidden as u32, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: gain.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: inv_rms.as_entire_binding() },
    ];
    dispatch(encoder, ctx, &ctx.pipelines.rmsnorm, &entries, (1, 1, 1));
}

fn dispatch_rope(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    heads: usize,
    head_dim: usize,
    pos: usize,
) {
    let params = ctx.params.alloc(
        &ctx.device,
        &ctx.queue,
        P8 { a: 1, b: heads as u32, c: head_dim as u32, d: pos as u32, e: 0, f: 0, g: 0, h: 0 },
    );
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
    ];
    dispatch(encoder, ctx, &ctx.pipelines.rope, &entries, (1, ceil_div(heads, 8), 1));
}

pub(crate) fn dispatch_swiglu(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    out: &wgpu::Buffer,
    len: usize,
) {
    let params = ctx.params.alloc(&ctx.device, &ctx.queue, P4 { a: len as u32, b: 0, c: 0, d: 0 });
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gate.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: up.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
    ];
    dispatch(encoder, ctx, &ctx.pipelines.swiglu, &entries, (ceil_div(len, 64), 1, 1));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_attention_decode(
    encoder: &mut wgpu::CommandEncoder,
    ctx: &GpuContext,
    q: &wgpu::Buffer,
    k_cache: &wgpu::Buffer,
    v_cache: &wgpu::Buffer,
    out: &wgpu::Buffer,
    cached_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) {
    let params = ctx.params.alloc(
        &ctx.device,
        &ctx.queue,
        P4 {
            a: cached_len as u32,
            b: heads as u32,
            c: kv_heads as u32,
            d: head_dim as u32,
        },
    );
    let entries = [
        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: q.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: k_cache.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: v_cache.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: out.as_entire_binding() },
    ];
    dispatch(
        encoder,
        ctx,
        &ctx.pipelines.attention_decode,
        &entries,
        (ceil_div(heads, 32), 1, 1),
    );
}

struct GpuLayer {
    ple: Option<wgpu::Buffer>,
    attn_norm_gain: wgpu::Buffer,
    wq: wgpu::Buffer,
    wk: wgpu::Buffer,
    wv: wgpu::Buffer,
    wo: wgpu::Buffer,
    mlp_norm_gain: wgpu::Buffer,
    w_gate: wgpu::Buffer,
    w_up: wgpu::Buffer,
    w_down: wgpu::Buffer,
    /// Ring buffers, `[capacity, kv_dim]`.
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
}

/// Scratch buffers for one decoded token. Allocated once and reused —
/// creating them per token would put a GPU allocation in the hot loop.
struct Scratch {
    token: wgpu::Buffer,
    hidden: wgpu::Buffer,
    ple_row: wgpu::Buffer,
    normed: wgpu::Buffer,
    inv_rms: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn: wgpu::Buffer,
    proj: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    act: wgpu::Buffer,
    logits: wgpu::Buffer,
    logits_read: wgpu::Buffer,
}

pub struct GpuModel {
    config: ModelConfig,
    embed: wgpu::Buffer,
    layers: Vec<GpuLayer>,
    final_norm_gain: wgpu::Buffer,
    scratch: Scratch,
    /// Ring capacity: the attention window.
    capacity: usize,
    /// Absolute position of the next token to decode.
    position: usize,
    /// How many ring slots hold live keys.
    cached_len: usize,
}

impl GpuModel {
    /// Upload the weights. Done once per model, not per generation.
    pub fn upload(
        ctx: &GpuContext,
        weights: &ModelWeights,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        if !supports(config) {
            return Err(format!(
                "this model's head dimension ({}) is past what the GPU decode kernel handles",
                config.head_dim()
            ));
        }
        let hidden = config.hidden_dim;
        let ffn = config.ffn_dim();
        let vocab = config.vocab_size();
        let kv_dim = config.kv_dim();
        let capacity = config.effective_window();

        let store = |label: &str, data: &[f32]| {
            buffers::upload_f32(&ctx.device, label, data, wgpu::BufferUsages::empty())
        };

        let layers = weights
            .layers
            .iter()
            .map(|layer| GpuLayer {
                ple: if config.use_ple { Some(store("ple", &layer.ple)) } else { None },
                attn_norm_gain: store("attn_norm_gain", &layer.attn_norm_gain),
                wq: store("wq", &layer.wq),
                wk: store("wk", &layer.wk),
                wv: store("wv", &layer.wv),
                wo: store("wo", &layer.wo),
                mlp_norm_gain: store("mlp_norm_gain", &layer.mlp_norm_gain),
                w_gate: store("w_gate", &layer.w_gate),
                w_up: store("w_up", &layer.w_up),
                w_down: store("w_down", &layer.w_down),
                k_cache: buffers::storage_f32(&ctx.device, "k_cache", capacity * kv_dim, false),
                v_cache: buffers::storage_f32(&ctx.device, "v_cache", capacity * kv_dim, false),
            })
            .collect();

        let scratch = Scratch {
            token: buffers::upload_u32(&ctx.device, "token", &[0]),
            hidden: buffers::storage_f32(&ctx.device, "hidden", hidden, false),
            ple_row: buffers::storage_f32(&ctx.device, "ple_row", hidden, false),
            normed: buffers::storage_f32(&ctx.device, "normed", hidden, false),
            inv_rms: buffers::storage_f32(&ctx.device, "inv_rms", 1, false),
            q: buffers::storage_f32(&ctx.device, "q", hidden, false),
            k: buffers::storage_f32(&ctx.device, "k", kv_dim, false),
            v: buffers::storage_f32(&ctx.device, "v", kv_dim, false),
            attn: buffers::storage_f32(&ctx.device, "attn", hidden, false),
            proj: buffers::storage_f32(&ctx.device, "proj", hidden.max(ffn), false),
            gate: buffers::storage_f32(&ctx.device, "gate", ffn, false),
            up: buffers::storage_f32(&ctx.device, "up", ffn, false),
            act: buffers::storage_f32(&ctx.device, "act", ffn, false),
            logits: buffers::storage_f32(&ctx.device, "logits", vocab, true),
            logits_read: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("logits_read"),
                size: (vocab * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        Ok(Self {
            config: *config,
            embed: store("embed", &weights.embed),
            layers,
            final_norm_gain: store("final_norm_gain", &weights.final_norm_gain),
            scratch,
            capacity,
            position: 0,
            cached_len: 0,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Seed the cache from a prefill the CPU already did.
    ///
    /// The CPU cache holds the last `window` positions oldest-first, and
    /// this backend's ring is the same size, so with a fresh ring the
    /// two layouts coincide and it's a straight copy.
    pub fn seed_from_cpu_cache(&mut self, ctx: &GpuContext, cache: &GenCache) {
        let kv_dim = self.config.kv_dim();
        let mut cached = 0usize;
        for (index, layer) in self.layers.iter().enumerate() {
            let keys = cache.layer_keys(index);
            let values = cache.layer_values(index);
            cached = keys.len() / kv_dim;
            ctx.queue.write_buffer(&layer.k_cache, 0, bytemuck::cast_slice(keys));
            ctx.queue.write_buffer(&layer.v_cache, 0, bytemuck::cast_slice(values));
        }
        self.cached_len = cached.min(self.capacity);
        self.position = cache.position();
    }

    /// Decode one token and return its logits.
    ///
    /// Every kernel for the step is encoded into a single command buffer
    /// and submitted once: one queue submission per token rather than
    /// per operation, which on a browser's WebGPU implementation is the
    /// difference between the GPU being busy and the GPU waiting.
    pub async fn decode_step(&mut self, ctx: &GpuContext, token: u32) -> Result<Vec<f32>, String> {
        let config = self.config;
        let hidden = config.hidden_dim;
        let ffn = config.ffn_dim();
        let vocab = config.vocab_size();
        let kv_dim = config.kv_dim();
        let heads = config.num_heads;
        let kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim();
        let slot = self.position % self.capacity;
        let cached_len = (self.cached_len + 1).min(self.capacity);

        ctx.params.reset();
        ctx.queue.write_buffer(&self.scratch.token, 0, bytemuck::cast_slice(&[token]));

        let mut encoder =
            ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        dispatch_gather(&mut encoder, ctx, &self.embed, &self.scratch.token, &self.scratch.hidden, hidden);

        for layer in &self.layers {
            if let Some(ple) = &layer.ple {
                dispatch_gather(&mut encoder, ctx, ple, &self.scratch.token, &self.scratch.ple_row, hidden);
                dispatch_add_inplace(&mut encoder, ctx, &self.scratch.hidden, &self.scratch.ple_row, hidden);
            }

            // --- attention ---
            dispatch_rmsnorm(&mut encoder, ctx, &self.scratch.hidden, &layer.attn_norm_gain, &self.scratch.normed, &self.scratch.inv_rms, hidden);
            dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &layer.wq, &self.scratch.q, 1, hidden, hidden);
            dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &layer.wk, &self.scratch.k, 1, hidden, kv_dim);
            dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &layer.wv, &self.scratch.v, 1, hidden, kv_dim);
            dispatch_rope(&mut encoder, ctx, &self.scratch.q, heads, head_dim, self.position);
            dispatch_rope(&mut encoder, ctx, &self.scratch.k, kv_heads, head_dim, self.position);

            // Append this position's key and value to the ring. A buffer
            // copy rather than a kernel: there is nothing to compute.
            let offset = (slot * kv_dim * 4) as u64;
            let bytes = (kv_dim * 4) as u64;
            encoder.copy_buffer_to_buffer(&self.scratch.k, 0, &layer.k_cache, offset, bytes);
            encoder.copy_buffer_to_buffer(&self.scratch.v, 0, &layer.v_cache, offset, bytes);

            dispatch_attention_decode(&mut encoder, ctx, &self.scratch.q, &layer.k_cache, &layer.v_cache, &self.scratch.attn, cached_len, heads, kv_heads, head_dim);
            dispatch_linear(&mut encoder, ctx, &self.scratch.attn, &layer.wo, &self.scratch.proj, 1, hidden, hidden);
            dispatch_add_inplace(&mut encoder, ctx, &self.scratch.hidden, &self.scratch.proj, hidden);

            // --- MLP ---
            dispatch_rmsnorm(&mut encoder, ctx, &self.scratch.hidden, &layer.mlp_norm_gain, &self.scratch.normed, &self.scratch.inv_rms, hidden);
            dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &layer.w_gate, &self.scratch.gate, 1, hidden, ffn);
            dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &layer.w_up, &self.scratch.up, 1, hidden, ffn);
            dispatch_swiglu(&mut encoder, ctx, &self.scratch.gate, &self.scratch.up, &self.scratch.act, ffn);
            dispatch_linear(&mut encoder, ctx, &self.scratch.act, &layer.w_down, &self.scratch.proj, 1, ffn, hidden);
            dispatch_add_inplace(&mut encoder, ctx, &self.scratch.hidden, &self.scratch.proj, hidden);
        }

        dispatch_rmsnorm(&mut encoder, ctx, &self.scratch.hidden, &self.final_norm_gain, &self.scratch.normed, &self.scratch.inv_rms, hidden);
        // Weight-tied output head.
        dispatch_linear(&mut encoder, ctx, &self.scratch.normed, &self.embed, &self.scratch.logits, 1, hidden, vocab);
        encoder.copy_buffer_to_buffer(&self.scratch.logits, 0, &self.scratch.logits_read, 0, (vocab * 4) as u64);
        ctx.queue.submit(Some(encoder.finish()));

        self.position += 1;
        self.cached_len = cached_len;

        read_back(ctx, &self.scratch.logits_read, vocab).await
    }

    /// Run one decode step on both backends from the same state and
    /// return the largest difference between their logits.
    ///
    /// This is how the WGSL gets checked without a GPU in the room:
    /// somebody with a browser runs it and reports one number. Anything
    /// past float rounding (well under 1e-2) means these kernels are
    /// computing a different model than the tested CPU one.
    pub async fn debug_compare_step(
        &mut self,
        ctx: &GpuContext,
        weights: &ModelWeights,
        prompt_tokens: &[u32],
        next_token: u32,
    ) -> Result<f64, String> {
        let (_, mut cpu_cache) = llm_core::model::prefill(weights, &self.config, prompt_tokens);
        self.seed_from_cpu_cache(ctx, &cpu_cache);
        let gpu_logits = self.decode_step(ctx, next_token).await?;
        let cpu_logits =
            llm_core::model::decode_step(weights, &self.config, &mut cpu_cache, next_token);
        Ok(gpu_logits
            .iter()
            .zip(&cpu_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max) as f64)
    }
}

async fn read_back(ctx: &GpuContext, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, String> {
    let slice = buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // In a browser there is nothing to poll: the page's event loop is
    // what drives the mapping to completion, and wgpu's blocking wait
    // isn't available on wasm anyway. Natively the device has to be
    // pumped, and nothing in this repo builds this crate natively today
    // — CI compiles it only for wasm32, through wasm-pack — so that arm
    // is a courtesy for anyone who tries.
    #[cfg(not(target_arch = "wasm32"))]
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    #[cfg(target_arch = "wasm32")]
    let _ = ctx;

    receiver
        .await
        .map_err(|_| "the readback was dropped before it completed".to_string())?
        .map_err(|e| format!("readback failed: {e}"))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| format!("mapping the readback buffer failed: {e}"))?;
    let out: Vec<f32> = bytemuck::cast_slice(&data)[..len].to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}
