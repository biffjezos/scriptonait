//! One transformer layer's weights, and its forward/backward passes.
//!
//! `super::forward`/`super::backward_into` are thin loops over these
//! methods — each layer runs its own forward/backward independent of
//! the others, given the residual-stream gradient flowing through it.
//! That's what replaces indexing `weights.layers[i]`/`cache.layers[i]`/
//! `grads.layers[i]` in lockstep across three parallel `Vec`s: a `zip`
//! over the three (see `super::backward_into`) can't walk them out of
//! step with each other the way manual indexing could.

use crate::config::ModelConfig;
use crate::ops;
use crate::rng::Rng;

use super::{gather_rows, scatter_add_rows, RMS_EPS};

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

/// One layer's cached activations, for its own `backward`.
pub(super) struct LayerCache {
    pub(super) h_after_ple: Vec<f32>, // input to attn rmsnorm
    pub(super) normed1: Vec<f32>,
    pub(super) inv_rms1: Vec<f32>,
    pub(super) q: Vec<f32>, // post-RoPE
    pub(super) k: Vec<f32>, // post-RoPE
    pub(super) v: Vec<f32>,
    pub(super) probs: Vec<f32>,
    pub(super) concat: Vec<f32>, // pre-Wo attention output
    pub(super) h_after_attn: Vec<f32>,
    pub(super) normed2: Vec<f32>,
    pub(super) inv_rms2: Vec<f32>,
    pub(super) gate: Vec<f32>,
    pub(super) up: Vec<f32>,
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

impl LayerWeights {
    pub(super) fn zeros(config: &ModelConfig) -> Self {
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

    pub(super) fn init(config: &ModelConfig, num_layers: usize, rng: &mut Rng) -> Self {
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
    pub(super) fn tensors_mut(&mut self) -> Vec<&mut Vec<f32>> {
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

    pub(super) fn tensors(&self) -> Vec<&Vec<f32>> {
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

    /// Runs this layer's forward pass: PLE add, attention block, MLP
    /// block, each added into `hidden` (the residual stream) in place.
    /// Returns the activation cache `backward` needs to retrace it.
    pub(super) fn forward(
        &self,
        hidden: &mut [f32],
        tokens: &[u32],
        config: &ModelConfig,
        t_len: usize,
    ) -> LayerCache {
        let h = config.hidden_dim;
        let heads = config.num_heads;
        let kv_heads = config.num_kv_heads;
        let kv = config.kv_dim();
        let head_dim = config.head_dim();
        let window = config.effective_window();

        if config.use_ple {
            let ple = gather_rows(&self.ple, tokens, h);
            for i in 0..hidden.len() {
                hidden[i] += ple[i];
            }
        }
        let h_after_ple = hidden.to_vec();

        let (normed1, inv_rms1) = ops::rmsnorm_fwd(hidden, &self.attn_norm_gain, t_len, h, RMS_EPS);
        let mut q = ops::linear_fwd(&normed1, &self.wq, t_len, h, h);
        let mut k = ops::linear_fwd(&normed1, &self.wk, t_len, h, kv);
        let v = ops::linear_fwd(&normed1, &self.wv, t_len, h, kv);
        ops::rope_apply(&mut q, t_len, heads, head_dim, config.rope_theta, false);
        ops::rope_apply(&mut k, t_len, kv_heads, head_dim, config.rope_theta, false);
        let (concat, probs) =
            ops::attention_fwd(&q, &k, &v, t_len, heads, kv_heads, head_dim, window);
        let attn_out = ops::linear_fwd(&concat, &self.wo, t_len, h, h);

        for i in 0..hidden.len() {
            hidden[i] += attn_out[i];
        }
        let h_after_attn = hidden.to_vec();

        let (normed2, inv_rms2) = ops::rmsnorm_fwd(hidden, &self.mlp_norm_gain, t_len, h, RMS_EPS);
        let gate = ops::linear_fwd(&normed2, &self.w_gate, t_len, h, config.ffn_dim());
        let up = ops::linear_fwd(&normed2, &self.w_up, t_len, h, config.ffn_dim());
        let act = ops::swiglu_fwd(&gate, &up);
        let mlp_out = ops::linear_fwd(&act, &self.w_down, t_len, config.ffn_dim(), h);

        for i in 0..hidden.len() {
            hidden[i] += mlp_out[i];
        }

        LayerCache {
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
        }
    }

    /// Runs this layer's backward pass given the downstream gradient
    /// `d_hidden` (gradient w.r.t. this layer's output residual stream)
    /// and this layer's own forward `cache`, accumulating this layer's
    /// parameter gradients into `grad_out` (same shape as `self`).
    /// Returns the gradient to propagate to the previous layer.
    pub(super) fn backward(
        &self,
        cache: &LayerCache,
        tokens: &[u32],
        config: &ModelConfig,
        t_len: usize,
        d_hidden: Vec<f32>,
        grad_out: &mut LayerWeights,
    ) -> Vec<f32> {
        let h = config.hidden_dim;
        let heads = config.num_heads;
        let kv_heads = config.num_kv_heads;
        let kv = config.kv_dim();
        let head_dim = config.head_dim();
        let window = config.effective_window();
        let lc = cache;
        let lg = grad_out;

        // --- MLP branch (residual: h_after_attn + mlp_out) ---
        let d_mlp_out = d_hidden.clone(); // gradient splits equally into both residual branches
        let (d_act, d_w_down) =
            ops::linear_bwd(&d_mlp_out, &lc.up_act_input(), &self.w_down, t_len, config.ffn_dim(), h);
        lg.w_down.iter_mut().zip(&d_w_down).for_each(|(g, d)| *g += d);
        let (d_gate, d_up) = ops::swiglu_bwd(&d_act, &lc.gate, &lc.up);
        let (d_normed2_from_gate, d_w_gate) =
            ops::linear_bwd(&d_gate, &lc.normed2, &self.w_gate, t_len, h, config.ffn_dim());
        let (d_normed2_from_up, d_w_up) =
            ops::linear_bwd(&d_up, &lc.normed2, &self.w_up, t_len, h, config.ffn_dim());
        lg.w_gate.iter_mut().zip(&d_w_gate).for_each(|(g, d)| *g += d);
        lg.w_up.iter_mut().zip(&d_w_up).for_each(|(g, d)| *g += d);
        let mut d_normed2 = vec![0.0f32; t_len * h];
        for i in 0..d_normed2.len() {
            d_normed2[i] = d_normed2_from_gate[i] + d_normed2_from_up[i];
        }

        let (d_h_after_attn_from_norm, d_mlp_gain) =
            ops::rmsnorm_bwd(&d_normed2, &lc.h_after_attn, &self.mlp_norm_gain, &lc.inv_rms2, t_len, h);
        lg.mlp_norm_gain.iter_mut().zip(&d_mlp_gain).for_each(|(g, d)| *g += d);

        // d_hidden at "h_after_attn" = contribution from residual pass-through (d_hidden itself) + from norm branch.
        let mut d_h_after_attn = d_hidden.clone();
        for i in 0..d_h_after_attn.len() {
            d_h_after_attn[i] += d_h_after_attn_from_norm[i];
        }

        // --- Attention branch (residual: h_after_ple + attn_out) ---
        let d_attn_out = d_h_after_attn.clone();
        let (d_concat, d_wo) = ops::linear_bwd(&d_attn_out, &lc.concat, &self.wo, t_len, h, h);
        lg.wo.iter_mut().zip(&d_wo).for_each(|(g, d)| *g += d);

        let (mut d_q, mut d_k, d_v) = ops::attention_bwd(
            &d_concat, &lc.q, &lc.k, &lc.v, &lc.probs, t_len, heads, kv_heads, head_dim, window,
        );
        ops::rope_apply(&mut d_q, t_len, heads, head_dim, config.rope_theta, true);
        ops::rope_apply(&mut d_k, t_len, kv_heads, head_dim, config.rope_theta, true);

        let (d_normed1_q, d_wq) = ops::linear_bwd(&d_q, &lc.normed1, &self.wq, t_len, h, h);
        let (d_normed1_k, d_wk) = ops::linear_bwd(&d_k, &lc.normed1, &self.wk, t_len, h, kv);
        let (d_normed1_v, d_wv) = ops::linear_bwd(&d_v, &lc.normed1, &self.wv, t_len, h, kv);
        lg.wq.iter_mut().zip(&d_wq).for_each(|(g, d)| *g += d);
        lg.wk.iter_mut().zip(&d_wk).for_each(|(g, d)| *g += d);
        lg.wv.iter_mut().zip(&d_wv).for_each(|(g, d)| *g += d);
        let mut d_normed1 = vec![0.0f32; t_len * h];
        for i in 0..d_normed1.len() {
            d_normed1[i] = d_normed1_q[i] + d_normed1_k[i] + d_normed1_v[i];
        }

        let (d_h_after_ple_from_norm, d_attn_gain) =
            ops::rmsnorm_bwd(&d_normed1, &lc.h_after_ple, &self.attn_norm_gain, &lc.inv_rms1, t_len, h);
        lg.attn_norm_gain.iter_mut().zip(&d_attn_gain).for_each(|(g, d)| *g += d);

        let mut d_h_after_ple = d_h_after_attn;
        for i in 0..d_h_after_ple.len() {
            d_h_after_ple[i] += d_h_after_ple_from_norm[i];
        }

        // PLE residual add: gradient passes through unchanged, and also
        // scatters into this layer's PLE table at the token positions.
        if config.use_ple {
            scatter_add_rows(&mut lg.ple, tokens, &d_h_after_ple, h);
        }
        d_h_after_ple
    }
}
