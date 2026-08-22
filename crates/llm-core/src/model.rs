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
    fn zeros(hidden: usize, ffn: usize, vocab: usize) -> Self {
        Self {
            ple: vec![0.0; vocab * hidden],
            attn_norm_gain: vec![0.0; hidden],
            wq: vec![0.0; hidden * hidden],
            wk: vec![0.0; hidden * hidden],
            wv: vec![0.0; hidden * hidden],
            wo: vec![0.0; hidden * hidden],
            mlp_norm_gain: vec![0.0; hidden],
            w_gate: vec![0.0; ffn * hidden],
            w_up: vec![0.0; ffn * hidden],
            w_down: vec![0.0; hidden * ffn],
        }
    }

    fn init(hidden: usize, ffn: usize, vocab: usize, rng: &mut Rng) -> Self {
        let linear = |out_dim: usize, in_dim: usize, rng: &mut Rng| -> Vec<f32> {
            let std = 1.0 / (in_dim as f32).sqrt();
            (0..out_dim * in_dim).map(|_| rng.next_gaussian() * std).collect()
        };
        Self {
            ple: (0..vocab * hidden).map(|_| rng.next_gaussian() * 0.02).collect(),
            attn_norm_gain: vec![1.0; hidden],
            wq: linear(hidden, hidden, rng),
            wk: linear(hidden, hidden, rng),
            wv: linear(hidden, hidden, rng),
            wo: linear(hidden, hidden, rng),
            mlp_norm_gain: vec![1.0; hidden],
            w_gate: linear(ffn, hidden, rng),
            w_up: linear(ffn, hidden, rng),
            w_down: linear(hidden, ffn, rng),
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
        let f = config.ffn_dim();
        let v = config.vocab_size();
        Self {
            embed: vec![0.0; v * h],
            layers: (0..config.num_layers).map(|_| LayerWeights::zeros(h, f, v)).collect(),
            final_norm_gain: vec![0.0; h],
        }
    }

    pub fn init(config: &ModelConfig, seed: u64) -> Self {
        let mut rng = Rng::seed_from_u64(seed);
        let h = config.hidden_dim;
        let f = config.ffn_dim();
        let v = config.vocab_size();
        Self {
            embed: (0..v * h).map(|_| rng.next_gaussian() * 0.02).collect(),
            layers: (0..config.num_layers).map(|_| LayerWeights::init(h, f, v, &mut rng)).collect(),
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
        for t in self.tensors() {
            for v in t.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
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

/// Adam optimizer state, shaped like the model.
pub struct AdamState {
    m: ModelWeights,
    v: ModelWeights,
    t: i32,
}

impl AdamState {
    pub fn new(config: &ModelConfig) -> Self {
        Self { m: ModelWeights::zeros(config), v: ModelWeights::zeros(config), t: 0 }
    }

    pub fn step(&mut self, weights: &mut ModelWeights, grads: &Gradients, lr: f32) {
        self.t += 1;
        let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bias1 = 1.0 - beta1.powi(self.t);
        let bias2 = 1.0 - beta2.powi(self.t);

        let w_tensors = weights.tensors_mut().into_iter();
        let g_tensors = grads.tensors().into_iter();
        let m_tensors = self.m.tensors_mut().into_iter();
        let v_tensors = self.v.tensors_mut().into_iter();

        for (((w, g), m), v) in w_tensors.zip(g_tensors).zip(m_tensors).zip(v_tensors) {
            for i in 0..w.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;
                w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
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
    let head_dim = config.head_dim();
    let window = config.effective_window();
    let vocab = config.vocab_size();

    let mut hidden = gather_rows(&weights.embed, tokens, h);
    let mut layer_caches = Vec::with_capacity(weights.layers.len());

    for layer in &weights.layers {
        let ple = gather_rows(&layer.ple, tokens, h);
        for i in 0..hidden.len() {
            hidden[i] += ple[i];
        }
        let h_after_ple = hidden.clone();

        let (normed1, inv_rms1) = ops::rmsnorm_fwd(&hidden, &layer.attn_norm_gain, t_len, h, RMS_EPS);
        let mut q = ops::linear_fwd(&normed1, &layer.wq, t_len, h, h);
        let mut k = ops::linear_fwd(&normed1, &layer.wk, t_len, h, h);
        let v = ops::linear_fwd(&normed1, &layer.wv, t_len, h, h);
        ops::rope_apply(&mut q, t_len, heads, head_dim, false);
        ops::rope_apply(&mut k, t_len, heads, head_dim, false);
        let (concat, probs) = ops::attention_fwd(&q, &k, &v, t_len, heads, head_dim, window);
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

        let (mut d_q, mut d_k, d_v) =
            ops::attention_bwd(&d_concat, &lc.q, &lc.k, &lc.v, &lc.probs, t_len, heads, head_dim, window);
        ops::rope_apply(&mut d_q, t_len, heads, head_dim, true);
        ops::rope_apply(&mut d_k, t_len, heads, head_dim, true);

        let (d_normed1_q, d_wq) = ops::linear_bwd(&d_q, &lc.normed1, &layer.wq, t_len, h, h);
        let (d_normed1_k, d_wk) = ops::linear_bwd(&d_k, &lc.normed1, &layer.wk, t_len, h, h);
        let (d_normed1_v, d_wv) = ops::linear_bwd(&d_v, &lc.normed1, &layer.wv, t_len, h, h);
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
        scatter_add_rows(&mut lg.ple, &cache.tokens, &d_h_after_ple, h);
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
        ModelConfig { num_layers: 2, hidden_dim: 8, num_heads: 2, context_len: 6, local_window: 6 }
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
            adam.step(&mut weights, &grads, 0.05);
        }
        let loss_after = total_loss(&weights, &config, &tokens, &targets);
        assert!(loss_after < loss_before, "loss_before={loss_before} loss_after={loss_after}");
    }
}
