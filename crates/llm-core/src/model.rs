//! Model weights, forward pass, and backward pass. See `config.rs` for the
//! architecture description and `ops.rs` for the underlying math.
//!
//! [`layer`] holds one transformer layer's weights and its own
//! forward/backward methods; [`forward`]/[`backward_into`] below are
//! thin loops over those, plus the embedding/output-head and final-norm
//! work that sits outside any one layer. [`optimizer`] holds AdamW and
//! gradient clipping.
//!
//! Everything here operates on a single sequence (no batch dimension) —
//! `train.rs` loops over the batch and accumulates gradients, which keeps
//! every index expression in this file about as simple as this kind of
//! model allows.

mod layer;
mod optimizer;

pub use layer::LayerWeights;
pub use optimizer::{clip_global_norm, AdamState, ADAM_BETA1, ADAM_BETA2, ADAM_EPS};

use layer::LayerCache;

use crate::config::{LayerLayout, ModelConfig};
#[cfg(test)]
use crate::config::LayerSharing;
use crate::ops;
use crate::rng::Rng;

const RMS_EPS: f32 = 1e-6;

#[derive(Clone)]
pub struct ModelWeights {
    pub embed: Vec<f32>, // [vocab, hidden] -- weight-tied with the output head
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Vec<f32>, // [hidden]
}

/// Gradients mirror `ModelWeights` field-for-field, same shapes.
pub type Gradients = ModelWeights;

impl ModelWeights {
    pub fn zeros(config: &ModelConfig) -> Self {
        let h = config.hidden_dim;
        let v = config.vocab_size();
        Self {
            embed: vec![0.0; v * h],
            layers: (0..config.unique_layer_count()).map(|_| LayerWeights::zeros(config)).collect(),
            final_norm_gain: vec![0.0; h],
        }
    }

    pub fn init(config: &ModelConfig, seed: u64) -> Self {
        let mut rng = Rng::seed_from_u64(seed);
        let h = config.hidden_dim;
        let v = config.vocab_size();
        // Residual-stream variance scaling (see `LayerWeights::init`) is
        // keyed to the actual depth of the stack, `num_layers` — every
        // depth position adds into the same residual stream once, whether
        // or not its weights are shared with another position — not to
        // `unique_layer_count()`, which only says how many distinct
        // weight sets back those `num_layers` additions.
        let depth = config.num_layers;
        Self {
            embed: (0..v * h).map(|_| rng.next_gaussian() * 0.02).collect(),
            layers: (0..config.unique_layer_count())
                .map(|_| LayerWeights::init(config, depth, &mut rng))
                .collect(),
            final_norm_gain: vec![1.0; h],
        }
    }

    pub fn param_count(&self) -> usize {
        self.tensors().iter().map(|t| t.len()).sum()
    }

    fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut out = vec![&mut self.embed];
        for l in &mut self.layers {
            out.extend(l.tensors_mut());
        }
        out.push(&mut self.final_norm_gain);
        out
    }

    fn tensors(&self) -> Vec<&Vec<f32>> {
        let mut out = vec![&self.embed];
        for l in &self.layers {
            out.extend(l.tensors());
        }
        out.push(&self.final_norm_gain);
        out
    }

    /// Whether weight decay applies to each tensor, in the same fixed
    /// order `tensors()` uses. The three RMSNorm gains per model
    /// (attention, MLP, final) are excluded; everything else is a matrix
    /// or an embedding table and gets decayed.
    fn decay_flags(&self) -> Vec<bool> {
        let mut flags = vec![true]; // embed
        for _ in &self.layers {
            // ple, attn_norm_gain, wq, wk, wv, wo, mlp_norm_gain,
            // w_gate, w_up, w_down
            flags.extend_from_slice(&[true, false, true, true, true, true, false, true, true, true]);
        }
        flags.push(false); // final_norm_gain
        flags
    }

    /// Reset every parameter/gradient buffer to zero in place (reused each
    /// training step to avoid reallocating the gradient buffers).
    pub fn zero_(&mut self) {
        for t in self.tensors_mut() {
            t.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Accumulate `other` into `self`, element-wise (`self += other`). Used
    /// to sum per-sequence gradients across a training batch.
    pub fn add_assign(&mut self, other: &Self) {
        for (dst, src) in self.tensors_mut().into_iter().zip(other.tensors()) {
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }
    }

    /// Scale every parameter/gradient by a constant (e.g. `1/batch_size`).
    pub fn scale_(&mut self, factor: f32) {
        for t in self.tensors_mut() {
            t.iter_mut().for_each(|v| *v *= factor);
        }
    }

    /// Flattens every tensor (in the same fixed order `tensors()` always
    /// uses) to little-endian f32 bytes — the on-disk/IndexedDB weight
    /// format. Carries no shape/config header: the caller already knows
    /// the `ModelConfig` (it's saved alongside, e.g. in IndexedDB) and
    /// passes it back into `from_bytes`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.param_count() * 4);
        self.write_into(&mut out, false);
        out
    }

    /// Append every parameter to `out`, at four bytes each or two.
    ///
    /// Appending rather than returning is the point. A 38M-parameter
    /// model is 153 MB of f32; building that as a `Vec` and then
    /// converting it to bf16 holds 230 MB at once, inside a wasm heap
    /// that also holds the live weights, the copy just downloaded from
    /// the GPU, and whatever the page was doing when it asked. That is
    /// how an export ends in `rust_oom` and takes the whole module with
    /// it — there is no unwinding from an allocation failure, so a model
    /// somebody trained overnight is simply gone.
    ///
    /// Writing straight into the destination at the width it wants holds
    /// one copy instead of three.
    pub fn write_into(&self, out: &mut Vec<u8>, bf16: bool) {
        for t in self.tensors() {
            for v in t.iter() {
                if bf16 {
                    out.extend_from_slice(&crate::bf16::to_bf16(*v).to_le_bytes());
                } else {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }

    /// Inverse of `to_bytes`, validated against `config`'s expected shapes.
    pub fn from_bytes(bytes: &[u8], config: &ModelConfig) -> Result<Self, String> {
        let mut w = Self::zeros(config);
        let expected = w.param_count() * 4;
        if bytes.len() != expected {
            return Err(format!("expected {expected} bytes for this config, got {}", bytes.len()));
        }
        let mut offset = 0usize;
        for t in w.tensors_mut() {
            for v in t.iter_mut() {
                let chunk: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
                *v = f32::from_le_bytes(chunk);
                offset += 4;
            }
        }
        Ok(w)
    }
}

fn gather_rows(table: &[f32], ids: &[u32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * hidden];
    for (t, &id) in ids.iter().enumerate() {
        let row = &table[id as usize * hidden..id as usize * hidden + hidden];
        out[t * hidden..(t + 1) * hidden].copy_from_slice(row);
    }
    out
}

fn scatter_add_rows(table_grad: &mut [f32], ids: &[u32], d_rows: &[f32], hidden: usize) {
    for (t, &id) in ids.iter().enumerate() {
        let src = &d_rows[t * hidden..(t + 1) * hidden];
        let dst = &mut table_grad[id as usize * hidden..id as usize * hidden + hidden];
        for i in 0..hidden {
            dst[i] += src[i];
        }
    }
}

pub struct Cache {
    /// The depth-to-group layout this forward pass actually ran —
    /// `backward_into` walks it in reverse rather than being handed a
    /// fresh one, so a training step's forward and backward always agree
    /// on which depth positions existed and which weights answered for
    /// them, even when `layer_sharing` is `RecurrentCore` and that can
    /// vary from one call to the next.
    layout: LayerLayout,
    tokens: Vec<u32>,
    layers: Vec<LayerCache>,
    h_final: Vec<f32>, // input to the final rmsnorm
    final_normed: Vec<f32>,
    final_inv_rms: Vec<f32>,
}

/// Runs the forward pass for one sequence over `layout`'s depth positions
/// (see `ModelConfig::layer_layout`). `tokens.len()` must be `<=
/// config.context_len`; positions are `0..tokens.len()`.
pub fn forward(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokens: &[u32],
    layout: &LayerLayout,
) -> (Vec<f32>, Cache) {
    let t_len = tokens.len();
    let h = config.hidden_dim;
    let vocab = config.vocab_size();

    let mut hidden = gather_rows(&weights.embed, tokens, h);
    // One cache entry per *depth position* (`layout.depth()`), not per
    // unique weight set (`weights.layers.len()`): activations differ at
    // every depth even where two positions share a weight set, and
    // backward needs each depth's own cache to retrace it.
    let mut layer_caches = Vec::with_capacity(layout.depth());
    for depth in 0..layout.depth() {
        let layer = &weights.layers[layout.group(depth)];
        layer_caches.push(layer.forward(&mut hidden, tokens, config, t_len));
    }

    let h_final = hidden.clone();
    let (final_normed, final_inv_rms) =
        ops::rmsnorm_fwd(&hidden, &weights.final_norm_gain, t_len, h, RMS_EPS);
    // Weight-tied output head: logits = final_normed @ embed^T.
    let logits = ops::linear_fwd(&final_normed, &weights.embed, t_len, h, vocab);

    (
        logits,
        Cache {
            layout: layout.clone(),
            tokens: tokens.to_vec(),
            layers: layer_caches,
            h_final,
            final_normed,
            final_inv_rms,
        },
    )
}

/// Key/value cache for one layer: every key and value still inside the
/// attention window, oldest first, RoPE already applied.
#[derive(Clone)]
struct LayerKv {
    k: Vec<f32>, // [cached_len, kv_dim]
    v: Vec<f32>, // [cached_len, kv_dim]
}

/// Incremental decoding state.
///
/// Without this, generating token *n* re-runs the forward pass over all
/// *n* previous tokens: producing a 900-token story from a 512-token
/// context costs on the order of 460,000 token-forwards. With it, the
/// prompt is processed once and each new token costs one row of work
/// plus attention over the window — about 1,400 token-forwards for the
/// same story. That factor is why generation stopped being the part you
/// wait on.
pub struct GenCache {
    /// The layout this generation is decoding at. Fixed for the whole
    /// generation (Geiping et al. 2025's "pick a depth once" — a KV
    /// cache entry belongs to a specific depth position, so it makes no
    /// sense to decode the next token at a different depth than the one
    /// this cache's keys and values were written under); rebuilding the
    /// cache (`Generator::maybe_reset_cache`) reuses this same layout
    /// rather than resampling it.
    layout: LayerLayout,
    layers: Vec<LayerKv>,
    /// Absolute position of the next token to be generated, which is
    /// also the number of tokens currently cached.
    pos: usize,
    /// Every token fed in so far, kept so the cache can be rebuilt when
    /// `pos` reaches `context_len` (see `Generator` in `generate.rs`).
    tokens: Vec<u32>,
}

impl GenCache {
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn layout(&self) -> &LayerLayout {
        &self.layout
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// One layer's cached keys, oldest first, `[cached, kv_dim]`.
    ///
    /// Exposed so another backend can be seeded from a prefill this one
    /// already computed — the WebGPU path prefills on the CPU (that
    /// forward pass is gradient-checked and fast) and then decodes on the
    /// GPU, rather than reimplementing a whole batched forward pass in
    /// WGSL that nobody could verify.
    pub fn layer_keys(&self, layer: usize) -> &[f32] {
        &self.layers[layer].k
    }

    pub fn layer_values(&self, layer: usize) -> &[f32] {
        &self.layers[layer].v
    }

    /// Drop everything older than the attention window. Positions of the
    /// surviving entries don't change, so their RoPE rotations stay
    /// valid — this only discards keys attention could no longer reach.
    fn trim_to_window(&mut self, window: usize, kv_dim: usize) {
        let cached = self.pos.min(self.cached_len(kv_dim));
        if cached <= window {
            return;
        }
        let drop = cached - window;
        for layer in &mut self.layers {
            layer.k.drain(..drop * kv_dim);
            layer.v.drain(..drop * kv_dim);
        }
    }

    fn cached_len(&self, kv_dim: usize) -> usize {
        self.layers.first().map(|l| l.k.len() / kv_dim).unwrap_or(0)
    }
}

/// Run the prompt through the model at `layout`'s depth and build the
/// decoding cache from it. Returns the logits for the *last* prompt
/// token — the distribution the first generated token is sampled from.
///
/// `tokens.len()` must be at least 1 and at most `config.context_len`.
pub fn prefill(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokens: &[u32],
    layout: &LayerLayout,
) -> (Vec<f32>, GenCache) {
    let vocab = config.vocab_size();
    let (logits, cache) = forward(weights, config, tokens, layout);
    let last = logits[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
    // `forward` already computed and cached every key and value; moving
    // them into the decoding cache is what makes prefill cost one
    // forward pass rather than two.
    let layers = cache.layers.iter().map(|lc| LayerKv { k: lc.k.clone(), v: lc.v.clone() }).collect();
    let mut gen = GenCache { layout: layout.clone(), layers, pos: tokens.len(), tokens: tokens.to_vec() };
    gen.trim_to_window(config.effective_window(), config.kv_dim());
    (last, gen)
}

/// Advance the cache by one token and return that token's logits.
///
/// This is the whole decode step: one row through every layer, attending
/// against the cached keys and values. No activation cache is built —
/// nothing backpropagates through generation.
pub fn decode_step(
    weights: &ModelWeights,
    config: &ModelConfig,
    cache: &mut GenCache,
    token: u32,
) -> Vec<f32> {
    let h = config.hidden_dim;
    let kv_dim = config.kv_dim();
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim();
    let window = config.effective_window();
    let pos = cache.pos;

    let mut hidden = gather_rows(&weights.embed, &[token], h);

    for layer_idx in 0..cache.layout.depth() {
        let layer = &weights.layers[cache.layout.group(layer_idx)];
        if config.use_ple {
            let ple = gather_rows(&layer.ple, &[token], h);
            for i in 0..h {
                hidden[i] += ple[i];
            }
        }

        let (normed1, _) = ops::rmsnorm_fwd(&hidden, &layer.attn_norm_gain, 1, h, RMS_EPS);
        let mut q = ops::linear_fwd(&normed1, &layer.wq, 1, h, h);
        let mut k = ops::linear_fwd(&normed1, &layer.wk, 1, h, kv_dim);
        let v = ops::linear_fwd(&normed1, &layer.wv, 1, h, kv_dim);
        // RoPE at this token's absolute position, once, before caching:
        // a cached key keeps the rotation of the position it was written
        // at, which is exactly what the whole-sequence forward pass would
        // have given it.
        ops::rope_apply_at(&mut q, 1, heads, head_dim, config.rope_theta, pos, false);
        ops::rope_apply_at(&mut k, 1, kv_heads, head_dim, config.rope_theta, pos, false);

        let lk = &mut cache.layers[layer_idx];
        lk.k.extend_from_slice(&k);
        lk.v.extend_from_slice(&v);
        // Trim *before* attending, not after: the whole-sequence
        // forward pass lets query position t see exactly the `window`
        // keys ending at t, so a decode step that attended over
        // window + 1 keys would be computing a slightly different model
        // than the one training trained.
        let mut cached_len = lk.k.len() / kv_dim;
        if cached_len > window {
            let drop = cached_len - window;
            lk.k.drain(..drop * kv_dim);
            lk.v.drain(..drop * kv_dim);
            cached_len = window;
        }
        let concat =
            ops::attention_step(&q, &lk.k, &lk.v, cached_len, heads, kv_heads, head_dim);
        let attn_out = ops::linear_fwd(&concat, &layer.wo, 1, h, h);
        for i in 0..h {
            hidden[i] += attn_out[i];
        }

        let (normed2, _) = ops::rmsnorm_fwd(&hidden, &layer.mlp_norm_gain, 1, h, RMS_EPS);
        let gate = ops::linear_fwd(&normed2, &layer.w_gate, 1, h, config.ffn_dim());
        let up = ops::linear_fwd(&normed2, &layer.w_up, 1, h, config.ffn_dim());
        let act = ops::swiglu_fwd(&gate, &up);
        let mlp_out = ops::linear_fwd(&act, &layer.w_down, 1, config.ffn_dim(), h);
        for i in 0..h {
            hidden[i] += mlp_out[i];
        }
    }

    let (final_normed, _) = ops::rmsnorm_fwd(&hidden, &weights.final_norm_gain, 1, h, RMS_EPS);
    let logits = ops::linear_fwd(&final_normed, &weights.embed, 1, h, config.vocab_size());

    cache.pos += 1;
    cache.tokens.push(token);
    logits
}

/// Backward pass given the upstream gradient wrt the logits (from
/// `ops::cross_entropy`, already mean-reduced over `T`), allocating a
/// fresh gradient buffer. Prefer `backward_into` in a training loop.
pub fn backward(weights: &ModelWeights, config: &ModelConfig, cache: &Cache, d_logits: &[f32]) -> Gradients {
    let mut grads = Gradients::zeros(config);
    backward_into(weights, config, cache, d_logits, &mut grads);
    grads
}

/// Backward pass that *accumulates* into an existing gradient buffer
/// (`grads += dL/dw`) instead of returning a new one.
///
/// Every write in here is already an accumulate, so summing a batch's
/// gradients needs no separate per-sequence buffer and no second
/// add-everything-together pass: the caller zeroes one buffer per step
/// and each sequence adds straight into it. At real model sizes that
/// buffer is several MB, and allocating plus zeroing one per sequence per
/// step was pure garbage-collector pressure inside the hottest loop in
/// the app.
pub fn backward_into(
    weights: &ModelWeights,
    config: &ModelConfig,
    cache: &Cache,
    d_logits: &[f32],
    grads: &mut Gradients,
) {
    let t_len = cache.tokens.len();
    let h = config.hidden_dim;
    let vocab = config.vocab_size();

    // Output head (tied with embed): logits = final_normed @ embed^T.
    let (d_final_normed, d_embed_from_head) =
        ops::linear_bwd(d_logits, &cache.final_normed, &weights.embed, t_len, h, vocab);
    for i in 0..grads.embed.len() {
        grads.embed[i] += d_embed_from_head[i];
    }

    let (mut d_hidden, d_final_gain) = ops::rmsnorm_bwd(
        &d_final_normed,
        &cache.h_final,
        &weights.final_norm_gain,
        &cache.final_inv_rms,
        t_len,
        h,
    );
    for i in 0..h {
        grads.final_norm_gain[i] += d_final_gain[i];
    }

    // Indexed by depth position (`cache.layers`, one entry per depth) and
    // by weight group (`weights.layers`/`grads.layers`, one entry per
    // unique weight set), via `cache.layout` — the same layout `forward`
    // built this cache with, not a fresh one — since those two lengths
    // differ whenever sharing is active, and (for `RecurrentCore`) the
    // grouping itself can differ from one forward pass to the next.
    // Every depth position sharing a group accumulates its gradient into
    // that one shared `grads.layers[group]` buffer in turn
    // (`LayerWeights::backward` already accumulates rather than
    // overwrites — the same principle already proven correct by the tied
    // input/output embedding's own gradient, and by the core in
    // `RecurrentCore` accumulating every one of its own repetitions).
    for depth in (0..cache.layout.depth()).rev() {
        let group = cache.layout.group(depth);
        let layer = &weights.layers[group];
        let lc = &cache.layers[depth];
        let lg = &mut grads.layers[group];
        d_hidden = layer.backward(lc, &cache.tokens, config, t_len, d_hidden, lg);
    }

    // Input embedding gather (the other half of the tied embed/head gradient).
    scatter_add_rows(&mut grads.embed, &cache.tokens, &d_hidden, h);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ModelConfig {
        // num_kv_heads < num_heads deliberately: the gradient check below is
        // the only thing that proves grouped-query attention's shared
        // dk/dv accumulation is right.
        // use_ple is on here (it's off by default) so the gradient check
        // still covers the per-layer embedding scatter.
        ModelConfig {
            num_layers: 2,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 6,
            local_window: 6,
            use_ple: true,
            ..Default::default()
        }
    }

    /// Four depth positions, two unique weight sets: depths 0-1 share one
    /// set, depths 2-3 share the other. Small enough for the same
    /// numerical gradient check as `small_config`, deliberately with
    /// `use_ple` off — a shared PLE table's scatter-add would need every
    /// depth in the group to scatter into the same table, which is
    /// already covered by this same accumulation principle and isn't
    /// worth a second check here.
    fn shared_layers_config() -> ModelConfig {
        ModelConfig {
            num_layers: 4,
            layer_sharing: LayerSharing::UniformGroups { unique_layers: 2 },
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 6,
            local_window: 6,
            ..Default::default()
        }
    }

    fn total_loss(weights: &ModelWeights, config: &ModelConfig, tokens: &[u32], targets: &[u32]) -> f32 {
        let layout = config.layer_layout(None);
        let (logits, _) = forward(weights, config, tokens, &layout);
        ops::cross_entropy(&logits, targets, tokens.len(), config.vocab_size()).0
    }

    #[test]
    fn param_count_matches_config_formula() {
        let config = small_config();
        let w = ModelWeights::init(&config, 1);
        assert_eq!(w.param_count(), config.param_count());
    }

    #[test]
    fn forward_output_shape() {
        let config = small_config();
        let w = ModelWeights::init(&config, 1);
        let tokens = vec![1u32, 2, 3, 4];
        let layout = config.layer_layout(None);
        let (logits, _) = forward(&w, &config, &tokens, &layout);
        assert_eq!(logits.len(), tokens.len() * config.vocab_size());
    }

    #[test]
    fn full_model_gradient_check() {
        let config = small_config();
        let weights = ModelWeights::init(&config, 7);
        let tokens = vec![5u32, 12, 200, 3, 65];
        let targets = vec![12u32, 200, 3, 65, 9];

        let layout = config.layer_layout(None);
        let (logits, cache) = forward(&weights, &config, &tokens, &layout);
        let (_, d_logits) = ops::cross_entropy(&logits, &targets, tokens.len(), config.vocab_size());
        let grads = backward(&weights, &config, &cache, &d_logits);

        // Spot-check a handful of parameters across different tensors
        // (embedding, a PLE table, attention/MLP weights, norm gains) with
        // numerical differentiation, since checking every one of a few
        // thousand parameters would be slow.
        let checks: Vec<(&str, usize)> = vec![
            ("embed", 0),
            ("embed", 2000),
            ("layer0.ple", 0),
            ("layer0.wq", 3),
            ("layer0.attn_norm_gain", 1),
            ("layer1.w_down", 5),
            ("layer1.mlp_norm_gain", 2),
            ("final_norm_gain", 0),
        ];

        let eps = 1e-3;
        for (name, idx) in checks {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            let (analytic, target_len) = match name {
                "embed" => (grads.embed[idx], weights.embed.len()),
                "layer0.ple" => (grads.layers[0].ple[idx], weights.layers[0].ple.len()),
                "layer0.wq" => (grads.layers[0].wq[idx], weights.layers[0].wq.len()),
                "layer0.attn_norm_gain" => {
                    (grads.layers[0].attn_norm_gain[idx], weights.layers[0].attn_norm_gain.len())
                }
                "layer1.w_down" => (grads.layers[1].w_down[idx], weights.layers[1].w_down.len()),
                "layer1.mlp_norm_gain" => {
                    (grads.layers[1].mlp_norm_gain[idx], weights.layers[1].mlp_norm_gain.len())
                }
                "final_norm_gain" => (grads.final_norm_gain[idx], weights.final_norm_gain.len()),
                _ => unreachable!(),
            };
            assert!(idx < target_len);

            let poke = |w: &mut ModelWeights, delta: f32| match name {
                "embed" => w.embed[idx] += delta,
                "layer0.ple" => w.layers[0].ple[idx] += delta,
                "layer0.wq" => w.layers[0].wq[idx] += delta,
                "layer0.attn_norm_gain" => w.layers[0].attn_norm_gain[idx] += delta,
                "layer1.w_down" => w.layers[1].w_down[idx] += delta,
                "layer1.mlp_norm_gain" => w.layers[1].mlp_norm_gain[idx] += delta,
                "final_norm_gain" => w.final_norm_gain[idx] += delta,
                _ => unreachable!(),
            };
            poke(&mut w_plus, eps);
            poke(&mut w_minus, -eps);

            let loss_plus = total_loss(&w_plus, &config, &tokens, &targets);
            let loss_minus = total_loss(&w_minus, &config, &tokens, &targets);
            let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);

            let diff = (analytic - numeric_grad).abs();
            let scale = analytic.abs().max(numeric_grad.abs()).max(1.0);
            assert!(
                diff / scale < 5e-2,
                "{name}[{idx}]: analytic={analytic} numeric={numeric_grad}"
            );
        }
    }

    #[test]
    fn shared_layer_weights_hold_the_right_number_of_sets() {
        let config = shared_layers_config();
        let w = ModelWeights::init(&config, 3);
        assert_eq!(w.layers.len(), config.unique_layer_count());
        assert_eq!(w.param_count(), config.param_count());
    }

    /// The gradient check that actually exercises sharing: depths 0 and 1
    /// both read `weights.layers[0]`, so a poke to one of its parameters
    /// changes the loss through *two* depth positions at once, and the
    /// analytic gradient has to be their sum — exactly what
    /// `backward_into` accumulating into one shared `grads.layers[group]`
    /// buffer across both depths is supposed to produce. A wrong
    /// `layer_group` mapping, or backward overwriting instead of
    /// accumulating, would show up here as a factor-of-two (or wrong
    /// depth entirely) mismatch that `full_model_gradient_check`'s
    /// un-shared config can't catch.
    #[test]
    fn shared_layer_gradient_check() {
        let config = shared_layers_config();
        let weights = ModelWeights::init(&config, 11);
        let tokens = vec![5u32, 12, 200, 3, 65];
        let targets = vec![12u32, 200, 3, 65, 9];

        let layout = config.layer_layout(None);
        let (logits, cache) = forward(&weights, &config, &tokens, &layout);
        let (_, d_logits) = ops::cross_entropy(&logits, &targets, tokens.len(), config.vocab_size());
        let grads = backward(&weights, &config, &cache, &d_logits);

        let eps = 1e-3;
        for idx in [0usize, 3, 9] {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.layers[0].wq[idx] += eps;
            w_minus.layers[0].wq[idx] -= eps;

            let loss_plus = total_loss(&w_plus, &config, &tokens, &targets);
            let loss_minus = total_loss(&w_minus, &config, &tokens, &targets);
            let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);

            let analytic = grads.layers[0].wq[idx];
            let diff = (analytic - numeric_grad).abs();
            let scale = analytic.abs().max(numeric_grad.abs()).max(1.0);
            assert!(
                diff / scale < 5e-2,
                "shared wq[{idx}]: analytic={analytic} numeric={numeric_grad}"
            );
        }
    }

    fn recurrent_core_config() -> ModelConfig {
        ModelConfig {
            num_layers: 5, // 1 prelude + 3 max core loops + 1 coda
            layer_sharing: LayerSharing::RecurrentCore {
                prelude_layers: 1,
                coda_layers: 1,
                core_loop_min: 1,
                core_loop_max: 3,
            },
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 6,
            local_window: 6,
            ..Default::default()
        }
    }

    /// The gradient check that exercises the recurrent core specifically:
    /// the default layout loops the core the maximum number of times, so
    /// the core's weight set gradient has to sum three separate depth
    /// positions' worth of contribution — full backprop through the
    /// repetition, not the paper's own truncated approximation (this
    /// app's small `core_loop_max` makes that unnecessary; see
    /// `PLAN.md`). A wrong depth-to-group mapping, or a backward that
    /// doesn't accumulate across every repetition, shows up here the same
    /// way `shared_layer_gradient_check` catches it for `UniformGroups`.
    #[test]
    fn recurrent_core_gradient_check() {
        let config = recurrent_core_config();
        let weights = ModelWeights::init(&config, 13);
        let tokens = vec![5u32, 12, 200, 3, 65];
        let targets = vec![12u32, 200, 3, 65, 9];

        let layout = config.layer_layout(None);
        assert_eq!(layout.depth(), 5);
        let (logits, cache) = forward(&weights, &config, &tokens, &layout);
        let (_, d_logits) = ops::cross_entropy(&logits, &targets, tokens.len(), config.vocab_size());
        let grads = backward(&weights, &config, &cache, &d_logits);

        let core_group = 1; // group 0 = prelude, group 1 = core, group 2 = coda
        let eps = 1e-3;
        for idx in [0usize, 5] {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.layers[core_group].wq[idx] += eps;
            w_minus.layers[core_group].wq[idx] -= eps;

            let loss_plus = total_loss(&w_plus, &config, &tokens, &targets);
            let loss_minus = total_loss(&w_minus, &config, &tokens, &targets);
            let numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);

            let analytic = grads.layers[core_group].wq[idx];
            let diff = (analytic - numeric_grad).abs();
            let scale = analytic.abs().max(numeric_grad.abs()).max(1.0);
            assert!(
                diff / scale < 5e-2,
                "core wq[{idx}]: analytic={analytic} numeric={numeric_grad}"
            );
        }
    }

    /// A shorter `core_loops` at generation time produces a shallower,
    /// but still runnable, decode — the same checkpoint answering at a
    /// different depth with no retraining (Geiping et al. 2025's
    /// "test-time compute scaling").
    #[test]
    fn recurrent_core_decode_step_runs_at_a_chosen_depth() {
        let config = recurrent_core_config();
        let weights = ModelWeights::init(&config, 4);
        let prompt = vec![1u32, 2, 3];

        for loops in [1usize, 2, 3] {
            let layout = config.layer_layout(Some(loops));
            assert_eq!(layout.depth(), 1 + loops + 1);
            let (_, mut cache) = prefill(&weights, &config, &prompt, &layout);
            let logits = decode_step(&weights, &config, &mut cache, 7);
            assert_eq!(logits.len(), config.vocab_size());
        }
    }

    #[test]
    fn weight_bytes_round_trip() {
        let config = small_config();
        let w = ModelWeights::init(&config, 99);
        let bytes = w.to_bytes();
        assert_eq!(bytes.len(), w.param_count() * 4);
        let w2 = ModelWeights::from_bytes(&bytes, &config).unwrap();
        assert_eq!(w.embed, w2.embed);
        assert_eq!(w.layers[0].wq, w2.layers[0].wq);
        assert_eq!(w.final_norm_gain, w2.final_norm_gain);
    }

    #[test]
    fn from_bytes_rejects_wrong_length() {
        let config = small_config();
        match ModelWeights::from_bytes(&[0u8; 3], &config) {
            Err(err) => assert!(err.contains("expected")),
            Ok(_) => panic!("expected an error for a truncated byte buffer"),
        }
    }

    #[test]
    fn adam_step_reduces_loss_on_a_tiny_batch() {
        let config = small_config();
        let mut weights = ModelWeights::init(&config, 3);
        let mut adam = AdamState::new(&config);
        let tokens = vec![10u32, 20, 30, 40];
        let targets = vec![20u32, 30, 40, 50];

        let layout = config.layer_layout(None);
        let loss_before = total_loss(&weights, &config, &tokens, &targets);
        for _ in 0..20 {
            let (logits, cache) = forward(&weights, &config, &tokens, &layout);
            let (_, d_logits) = ops::cross_entropy(&logits, &targets, tokens.len(), config.vocab_size());
            let grads = backward(&weights, &config, &cache, &d_logits);
            adam.step(&mut weights, &grads, 0.05, 0.0);
        }
        let loss_after = total_loss(&weights, &config, &tokens, &targets);
        assert!(loss_after < loss_before, "loss_before={loss_before} loss_after={loss_after}");
    }
}
