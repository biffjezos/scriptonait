//! Gradient clipping and the AdamW optimizer state.

use crate::config::ModelConfig;

use super::{Gradients, ModelWeights};

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
