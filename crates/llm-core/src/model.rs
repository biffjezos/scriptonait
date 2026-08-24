//! Model weights, forward pass, and backward pass. See `config.rs` for the
//! architecture description and `ops.rs` for the underlying math; this
//! module is mostly bookkeeping that wires those primitives together
//! per-layer and keeps the activation cache backward needs.
//!
//! Everything here operates on a single sequence (no batch dimension) —
//! `train.rs` loops over the batch and accumulates gradients, which keeps
//! every index expression in this file about as simple as this kind of
//! model allows.

use crate::config::ModelConfig;
use crate::ops;
use crate::rng::Rng;

const RMS_EPS: f32 = 1e-6;

#[derive(Clone)]
pub struct LayerWeights {
    pub ple: Vec<f32>,            // [vocab, hidden]
    pub attn_norm_gain: Vec<f32>, // [hidden]
    pub wq: Vec<f32>,             // [hidden, hidden]
    pub wk: Vec<f32>,             // [hidden, hidden]
    pub wv: Vec<f32>,             // [hidden, hidden]
    pub wo: Vec<f32>,             // [hidden, hidden]
    pub mlp_norm_gain: Vec<f32>,  // [hidden]
    pub w_gate: Vec<f32>,         // [ffn, hidden]
    pub w_up: Vec<f32>,           // [ffn, hidden]
    pub w_down: Vec<f32>,         // [hidden, ffn]
}

#[derive(Clone)]
pub struct ModelWeights {
    pub embed: Vec<f32>, // [vocab, hidden] -- weight-tied with the output head
    pub layers: Vec<LayerWeights>,
    pub final_norm_gain: Vec<f32>, // [hidden]
}

/// Gradients mirror `ModelWeights` field-for-field, same shapes.
pub type Gradients = ModelWeights;

impl LayerWeights {
    fn zeros(config: &ModelConfig) -> Self {
        let (hidden, ffn, kv) = (config.hidden_dim, config.ffn_dim(), config.kv_dim());
        let ple_len = if config.use_ple { config.vocab_size() * hidden } else { 0 };
        Self {
            ple: vec![0.0; ple_len],
            attn_norm_gain: vec![0.0; hidden],
            wq: vec![0.0; hidden * hidden],
            wk: vec![0.0; kv * hidden],
            wv: vec![0.0; kv * hidden],
            wo: vec![0.0; hidden * hidden],
            mlp_norm_gain: vec![0.0; hidden],
            w_gate: vec![0.0; ffn * hidden],
            w_up: vec![0.0; ffn * hidden],
            w_down: vec![0.0; hidden * ffn],
        }
    }

    fn init(config: &ModelConfig, num_layers: usize, rng: &mut Rng) -> Self {
        let (hidden, ffn, kv) = (config.hidden_dim, config.ffn_dim(), config.kv_dim());
        let linear = |out_dim: usize, in_dim: usize, rng: &mut Rng| -> Vec<f32> {
            let std = 1.0 / (in_dim as f32).sqrt();
            (0..out_dim * in_dim).map(|_| rng.next_gaussian() * std).collect()
        };
        // The two projections that write *into* the residual stream get
        // their initial scale divided by sqrt(2 * num_layers). Every
        // layer adds into the same stream, so without this the stream's
        // variance grows with depth and the first few hundred steps are
        // spent undoing that instead of learning (GPT-2's initialization,
        // and the reason deep stacks train stably from step one).
        let residual_scale = 1.0 / (2.0 * num_layers as f32).sqrt();
        let scaled = |out_dim: usize, in_dim: usize, rng: &mut Rng| -> Vec<f32> {
            let std = residual_scale / (in_dim as f32).sqrt();
            (0..out_dim * in_dim).map(|_| rng.next_gaussian() * std).collect()
        };
        let ple_len = if config.use_ple { config.vocab_size() * hidden } else { 0 };
        Self {
            ple: (0..ple_len).map(|_| rng.next_gaussian() * 0.02).collect(),
            attn_norm_gain: vec![1.0; hidden],
            wq: linear(hidden, hidden, rng),
            wk: linear(kv, hidden, rng),
            wv: linear(kv, hidden, rng),
            wo: scaled(hidden, hidden, rng),
            mlp_norm_gain: vec![1.0; hidden],
            w_gate: linear(ffn, hidden, rng),
            w_up: linear(ffn, hidden, rng),
            w_down: scaled(hidden, ffn, rng),
        }
    }

    /// All buffers as `(name, slice)` pairs, in a fixed order shared by
    /// every `LayerWeights` — used to zip weights/grads/optimizer state
    /// together generically instead of repeating field lists everywhere.
    fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![
            &mut self.ple,
            &mut self.attn_norm_gain,
            &mut self.wq,
            &mut self.wk,
            &mut self.wv,
            &mut self.wo,
            &mut self.mlp_norm_gain,
            &mut self.w_gate,
            &mut self.w_up,
            &mut self.w_down,
        ]
    }

    fn tensors(&self) -> Vec<&Vec<f32>> {
        vec![
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

impl ModelWeights {
    pub fn zeros(config: &ModelConfig) -> Self {
        let h = config.hidden_dim;
        let v = config.vocab_size();
        Self {
            embed: vec![0.0; v * h],
            layers: (0..config.num_layers).map(|_| LayerWeights::zeros(config)).collect(),
            final_norm_gain: vec![0.0; h],
        }
    }

    pub fn init(config: &ModelConfig, seed: u64) -> Self {
        let mut rng = Rng::seed_from_u64(seed);
        let h = config.hidden_dim;
        let v = config.vocab_size();
        let n = config.num_layers;
        Self {
            embed: (0..v * h).map(|_| rng.next_gaussian() * 0.02).collect(),
            layers: (0..n).map(|_| LayerWeights::init(config, n, &mut rng)).collect(),
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
                    out.extend_from_slice(&crate::checkpoint::to_bf16(*v).to_le_bytes());
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

/// Rescale `grads` in place so its global L2 norm is at most
/// `max_norm`, and return the norm it had before clipping.
///
/// One bad batch — an unusual run of tokens, a rare character — can
/// produce a gradient orders of magnitude larger than typical, and Adam
/// happily takes a full-size step along it, which is what a loss curve
/// that suddenly jumps and never recovers actually is. Clipping the
/// whole gradient as one vector (rather than per tensor) keeps its
/// direction and only limits how far the step goes.
pub fn clip_global_norm(grads: &mut Gradients, max_norm: f32) -> f32 {
    let mut sum_sq = 0.0f64;
    for t in grads.tensors() {
        for &g in t.iter() {
            sum_sq += (g as f64) * (g as f64);
        }
    }
    let norm = sum_sq.sqrt() as f32;
    if norm > max_norm && norm.is_finite() && norm > 0.0 {
        let scale = max_norm / norm;
        grads.scale_(scale);
    }
    norm
}

/// AdamW optimizer state, shaped like the model.
pub struct AdamState {
    m: ModelWeights,
    v: ModelWeights,
    t: i32,
    /// Which tensors weight decay applies to, in `tensors()` order.
    decay: Vec<bool>,
}

impl AdamState {
    pub fn new(config: &ModelConfig) -> Self {
        let template = ModelWeights::zeros(config);
        let decay = template.decay_flags();
        Self { m: ModelWeights::zeros(config), v: ModelWeights::zeros(config), t: 0, decay }
    }

    /// One AdamW step.
    ///
    /// Decoupled weight decay, not L2 added to the gradient: with Adam
    /// the two are not the same thing, because an L2 term goes through
    /// the same per-parameter normalization as the gradient and so decays
    /// rarely-updated parameters far less than often-updated ones.
    /// Decoupling it (`w -= lr * wd * w`, applied directly) is what
    /// "AdamW" means, and it's the version that actually regularizes.
    ///
    /// Decay is skipped for the RMSNorm gains: they're scale parameters
    /// initialized at 1, and pulling them toward 0 shrinks the whole
    /// residual stream for no benefit.
    pub fn step(&mut self, weights: &mut ModelWeights, grads: &Gradients, lr: f32, weight_decay: f32) {
        self.t += 1;
        let (beta1, beta2, eps) = (0.9f32, 0.95f32, 1e-8f32);
        let bias1 = 1.0 - beta1.powi(self.t);
        let bias2 = 1.0 - beta2.powi(self.t);

        let w_tensors = weights.tensors_mut().into_iter();
        let g_tensors = grads.tensors().into_iter();
        let m_tensors = self.m.tensors_mut().into_iter();
        let v_tensors = self.v.tensors_mut().into_iter();

        for (idx, (((w, g), m), v)) in
            w_tensors.zip(g_tensors).zip(m_tensors).zip(v_tensors).enumerate()
        {
            let wd = if self.decay.get(idx).copied().unwrap_or(true) { weight_decay } else { 0.0 };
            for i in 0..w.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                w[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * w[i]);
            }
        }
    }
}

impl AdamState {
    /// Serialize the moment buffers and the step counter.
    ///
    /// Three times the size of the weights, which is why this is a
    /// separate file from the checkpoint rather than part of it — but a
    /// pretraining run that spans several CI jobs has to carry it, or
    /// every resume throws away Adam's momentum and the loss visibly
    /// jumps at each restart.
    pub fn to_bytes(&self) -> Vec<u8> {
        let m = self.m.to_bytes();
        let v = self.v.to_bytes();
        let mut out = Vec::with_capacity(8 + m.len() + v.len());
        out.extend_from_slice(&(self.t as u32).to_le_bytes());
        out.extend_from_slice(&(m.len() as u32).to_le_bytes());
        out.extend_from_slice(&m);
        out.extend_from_slice(&v);
        out
    }

    /// Build the state from moment buffers held elsewhere — the GPU
    /// trainer keeps its own on the device and downloads them to save.
    pub fn from_parts(m: ModelWeights, v: ModelWeights, t: i32) -> Self {
        let decay = m.decay_flags();
        Self { m, v, t, decay }
    }

    /// The moment buffers and the step count, for a caller that has to
    /// upload them somewhere.
    pub fn parts(&self) -> (&ModelWeights, &ModelWeights, i32) {
        (&self.m, &self.v, self.t)
    }

    pub fn from_bytes(bytes: &[u8], config: &ModelConfig) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("optimizer state truncated".to_string());
        }
        let t = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as i32;
        let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if bytes.len() != 8 + 2 * len {
            return Err(format!(
                "optimizer state is {} bytes, expected {} for this model shape",
                bytes.len(),
                8 + 2 * len
            ));
        }
        let m = ModelWeights::from_bytes(&bytes[8..8 + len], config)?;
        let v = ModelWeights::from_bytes(&bytes[8 + len..], config)?;
        let decay = m.decay_flags();
        Ok(Self { m, v, t, decay })
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

struct LayerCache {
    h_after_ple: Vec<f32>,   // input to attn rmsnorm
    normed1: Vec<f32>,
    inv_rms1: Vec<f32>,
    q: Vec<f32>, // post-RoPE
    k: Vec<f32>, // post-RoPE
    v: Vec<f32>,
    probs: Vec<f32>,
    concat: Vec<f32>, // pre-Wo attention output
    h_after_attn: Vec<f32>,
    normed2: Vec<f32>,
    inv_rms2: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

pub struct Cache {
    tokens: Vec<u32>,
    layers: Vec<LayerCache>,
    h_final: Vec<f32>, // input to the final rmsnorm
    final_normed: Vec<f32>,
    final_inv_rms: Vec<f32>,
}

/// Runs the forward pass for one sequence. `tokens.len()` must be `<=
/// config.context_len`; positions are `0..tokens.len()`.
pub fn forward(weights: &ModelWeights, config: &ModelConfig, tokens: &[u32]) -> (Vec<f32>, Cache) {
    let t_len = tokens.len();
    let h = config.hidden_dim;
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let kv = config.kv_dim();
    let head_dim = config.head_dim();
    let window = config.effective_window();
    let vocab = config.vocab_size();

    let mut hidden = gather_rows(&weights.embed, tokens, h);
    let mut layer_caches = Vec::with_capacity(weights.layers.len());

    for layer in &weights.layers {
        if config.use_ple {
            let ple = gather_rows(&layer.ple, tokens, h);
            for i in 0..hidden.len() {
                hidden[i] += ple[i];
            }
        }
        let h_after_ple = hidden.clone();

        let (normed1, inv_rms1) = ops::rmsnorm_fwd(&hidden, &layer.attn_norm_gain, t_len, h, RMS_EPS);
        let mut q = ops::linear_fwd(&normed1, &layer.wq, t_len, h, h);
        let mut k = ops::linear_fwd(&normed1, &layer.wk, t_len, h, kv);
        let v = ops::linear_fwd(&normed1, &layer.wv, t_len, h, kv);
        ops::rope_apply(&mut q, t_len, heads, head_dim, config.rope_theta, false);
        ops::rope_apply(&mut k, t_len, kv_heads, head_dim, config.rope_theta, false);
        let (concat, probs) =
            ops::attention_fwd(&q, &k, &v, t_len, heads, kv_heads, head_dim, window);
        let attn_out = ops::linear_fwd(&concat, &layer.wo, t_len, h, h);

        for i in 0..hidden.len() {
            hidden[i] += attn_out[i];
        }
        let h_after_attn = hidden.clone();

        let (normed2, inv_rms2) = ops::rmsnorm_fwd(&hidden, &layer.mlp_norm_gain, t_len, h, RMS_EPS);
        let gate = ops::linear_fwd(&normed2, &layer.w_gate, t_len, h, config.ffn_dim());
        let up = ops::linear_fwd(&normed2, &layer.w_up, t_len, h, config.ffn_dim());
        let act = ops::swiglu_fwd(&gate, &up);
        let mlp_out = ops::linear_fwd(&act, &layer.w_down, t_len, config.ffn_dim(), h);

        for i in 0..hidden.len() {
            hidden[i] += mlp_out[i];
        }

        layer_caches.push(LayerCache {
            h_after_ple,
            normed1,
            inv_rms1,
            q,
            k,
            v,
            probs,
            concat,
            h_after_attn,
            normed2,
            inv_rms2,
            gate,
            up,
        });
    }

    let h_final = hidden.clone();
    let (final_normed, final_inv_rms) =
        ops::rmsnorm_fwd(&hidden, &weights.final_norm_gain, t_len, h, RMS_EPS);
    // Weight-tied output head: logits = final_normed @ embed^T.
    let logits = ops::linear_fwd(&final_normed, &weights.embed, t_len, h, vocab);

    (
        logits,
        Cache { tokens: tokens.to_vec(), layers: layer_caches, h_final, final_normed, final_inv_rms },
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

/// Run the prompt through the model and build the decoding cache from it.
/// Returns the logits for the *last* prompt token — the distribution the
/// first generated token is sampled from.
///
/// `tokens.len()` must be at least 1 and at most `config.context_len`.
pub fn prefill(weights: &ModelWeights, config: &ModelConfig, tokens: &[u32]) -> (Vec<f32>, GenCache) {
    let vocab = config.vocab_size();
    let (logits, cache) = forward(weights, config, tokens);
    let last = logits[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
    // `forward` already computed and cached every key and value; moving
    // them into the decoding cache is what makes prefill cost one
    // forward pass rather than two.
    let layers = cache.layers.iter().map(|lc| LayerKv { k: lc.k.clone(), v: lc.v.clone() }).collect();
    let mut gen = GenCache { layers, pos: tokens.len(), tokens: tokens.to_vec() };
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

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
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
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let kv = config.kv_dim();
    let head_dim = config.head_dim();
    let window = config.effective_window();
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

    for (layer_idx, layer) in weights.layers.iter().enumerate().rev() {
        let lc = &cache.layers[layer_idx];
        let lg = &mut grads.layers[layer_idx];

        // --- MLP branch (residual: h_after_attn + mlp_out) ---
        let d_mlp_out = d_hidden.clone(); // gradient splits equally into both residual branches
        let (d_act, d_w_down) =
            ops::linear_bwd(&d_mlp_out, &lc.up_act_input(), &layer.w_down, t_len, config.ffn_dim(), h);
        lg.w_down.iter_mut().zip(&d_w_down).for_each(|(g, d)| *g += d);
        let (d_gate, d_up) = ops::swiglu_bwd(&d_act, &lc.gate, &lc.up);
        let (d_normed2_from_gate, d_w_gate) =
            ops::linear_bwd(&d_gate, &lc.normed2, &layer.w_gate, t_len, h, config.ffn_dim());
        let (d_normed2_from_up, d_w_up) =
            ops::linear_bwd(&d_up, &lc.normed2, &layer.w_up, t_len, h, config.ffn_dim());
        lg.w_gate.iter_mut().zip(&d_w_gate).for_each(|(g, d)| *g += d);
        lg.w_up.iter_mut().zip(&d_w_up).for_each(|(g, d)| *g += d);
        let mut d_normed2 = vec![0.0f32; t_len * h];
        for i in 0..d_normed2.len() {
            d_normed2[i] = d_normed2_from_gate[i] + d_normed2_from_up[i];
        }

        let (d_h_after_attn_from_norm, d_mlp_gain) =
            ops::rmsnorm_bwd(&d_normed2, &lc.h_after_attn, &layer.mlp_norm_gain, &lc.inv_rms2, t_len, h);
        lg.mlp_norm_gain.iter_mut().zip(&d_mlp_gain).for_each(|(g, d)| *g += d);

        // d_hidden at "h_after_attn" = contribution from residual pass-through (d_hidden itself) + from norm branch.
        let mut d_h_after_attn = d_hidden.clone();
        for i in 0..d_h_after_attn.len() {
            d_h_after_attn[i] += d_h_after_attn_from_norm[i];
        }

        // --- Attention branch (residual: h_after_ple + attn_out) ---
        let d_attn_out = d_h_after_attn.clone();
        let (d_concat, d_wo) = ops::linear_bwd(&d_attn_out, &lc.concat, &layer.wo, t_len, h, h);
        lg.wo.iter_mut().zip(&d_wo).for_each(|(g, d)| *g += d);

        let (mut d_q, mut d_k, d_v) = ops::attention_bwd(
            &d_concat, &lc.q, &lc.k, &lc.v, &lc.probs, t_len, heads, kv_heads, head_dim, window,
        );
        ops::rope_apply(&mut d_q, t_len, heads, head_dim, config.rope_theta, true);
        ops::rope_apply(&mut d_k, t_len, kv_heads, head_dim, config.rope_theta, true);

        let (d_normed1_q, d_wq) = ops::linear_bwd(&d_q, &lc.normed1, &layer.wq, t_len, h, h);
        let (d_normed1_k, d_wk) = ops::linear_bwd(&d_k, &lc.normed1, &layer.wk, t_len, h, kv);
        let (d_normed1_v, d_wv) = ops::linear_bwd(&d_v, &lc.normed1, &layer.wv, t_len, h, kv);
        lg.wq.iter_mut().zip(&d_wq).for_each(|(g, d)| *g += d);
        lg.wk.iter_mut().zip(&d_wk).for_each(|(g, d)| *g += d);
        lg.wv.iter_mut().zip(&d_wv).for_each(|(g, d)| *g += d);
        let mut d_normed1 = vec![0.0f32; t_len * h];
        for i in 0..d_normed1.len() {
            d_normed1[i] = d_normed1_q[i] + d_normed1_k[i] + d_normed1_v[i];
        }

        let (d_h_after_ple_from_norm, d_attn_gain) =
            ops::rmsnorm_bwd(&d_normed1, &lc.h_after_ple, &layer.attn_norm_gain, &lc.inv_rms1, t_len, h);
        lg.attn_norm_gain.iter_mut().zip(&d_attn_gain).for_each(|(g, d)| *g += d);

        let mut d_h_after_ple = d_h_after_attn;
        for i in 0..d_h_after_ple.len() {
            d_h_after_ple[i] += d_h_after_ple_from_norm[i];
        }

        // PLE residual add: gradient passes through unchanged, and also
        // scatters into this layer's PLE table at the token positions.
        if config.use_ple {
            scatter_add_rows(&mut lg.ple, &cache.tokens, &d_h_after_ple, h);
        }
        d_hidden = d_h_after_ple;
    }

    // Input embedding gather (the other half of the tied embed/head gradient).
    scatter_add_rows(&mut grads.embed, &cache.tokens, &d_hidden, h);
}

impl LayerCache {
    /// The MLP down-projection's input is the SwiGLU activation, which
    /// isn't stored directly (only its `gate`/`up` inputs are) — recompute
    /// it, which is exact (not an approximation) since SwiGLU is a pure
    /// function of the cached `gate`/`up`.
    fn up_act_input(&self) -> Vec<f32> {
        ops::swiglu_fwd(&self.gate, &self.up)
    }
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

    fn total_loss(weights: &ModelWeights, config: &ModelConfig, tokens: &[u32], targets: &[u32]) -> f32 {
        let (logits, _) = forward(weights, config, tokens);
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
        let (logits, _) = forward(&w, &config, &tokens);
        assert_eq!(logits.len(), tokens.len() * config.vocab_size());
    }

    #[test]
    fn full_model_gradient_check() {
        let config = small_config();
        let weights = ModelWeights::init(&config, 7);
        let tokens = vec![5u32, 12, 200, 3, 65];
        let targets = vec![12u32, 200, 3, 65, 9];

        let (logits, cache) = forward(&weights, &config, &tokens);
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

        let loss_before = total_loss(&weights, &config, &tokens, &targets);
        for _ in 0..20 {
            let (logits, cache) = forward(&weights, &config, &tokens);
            let (_, d_logits) = ops::cross_entropy(&logits, &targets, tokens.len(), config.vocab_size());
            let grads = backward(&weights, &config, &cache, &d_logits);
            adam.step(&mut weights, &grads, 0.05, 0.0);
        }
        let loss_after = total_loss(&weights, &config, &tokens, &targets);
        assert!(loss_after < loss_before, "loss_before={loss_before} loss_after={loss_after}");
    }
}
