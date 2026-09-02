//! The GPU-resident data shapes a training step reads and writes:
//! per-tensor buffers (`ParamSet`, used for the weights, the gradients,
//! and both Adam moment buffers), the per-layer activation cache kept
//! for backward (`LayerActs`), and the scratch shared across every layer
//! and sequence (`Scratch`).
//!
//! Every field here is `pub(super)`: forward, backward, dispatch, profile
//! and I/O all reach into these shapes directly rather than through
//! accessor methods, since an accessor per field would just be
//! boilerplate around what is, in every case, "the buffer this kernel
//! reads or writes."

use llm_core::config::ModelConfig;
use llm_core::model::ModelWeights;

use crate::buffers;
use crate::context::GpuContext;

/// Per-layer tensors, in the order `ParamSet` stores them.
///
/// A Mixture-of-Experts FFN (see `llm_core::model::layer`'s `ffn_forward`
/// for where that seam lives on the CPU side) would need N sets of
/// `T_W_GATE`/`T_W_UP`/`T_W_DOWN` per layer instead of one, plus a
/// router. That means `TENSORS_PER_LAYER` becoming a value computed from
/// `ModelConfig` (an expert count) rather than this fixed `const`, and
/// `T_W_GATE`/`T_W_UP`/`T_W_DOWN` becoming index-computing functions of
/// an expert number rather than plain constants — `layer()`'s offset
/// arithmetic below, and `shapes()`/`from_weights()`'s per-layer literal
/// arrays, would all need to loop over experts instead of listing 9
/// fixed fields. Not done here: no expert count, router shape, or
/// checkpoint format for either exists yet (see `PLAN.md` and
/// `checkpoint.rs`'s own note on its version scheme) — this note exists
/// so that design starts from what actually has to change instead of
/// rediscovering it.
pub(super) const T_ATTN_GAIN: usize = 0;
pub(super) const T_WQ: usize = 1;
pub(super) const T_WK: usize = 2;
pub(super) const T_WV: usize = 3;
pub(super) const T_WO: usize = 4;
pub(super) const T_MLP_GAIN: usize = 5;
pub(super) const T_W_GATE: usize = 6;
pub(super) const T_W_UP: usize = 7;
pub(super) const T_W_DOWN: usize = 8;
pub(super) const TENSORS_PER_LAYER: usize = 9;

pub(super) struct TensorSlot {
    pub(super) buffer: wgpu::Buffer,
    pub(super) len: usize,
    /// Weight decay applies to matrices and embedding tables, not to the
    /// RMSNorm gains — matching `ModelWeights::decay_flags`.
    pub(super) decay: bool,
}

/// One copy of every parameter-shaped tensor: the weights, the gradient
/// accumulator, or an Adam moment buffer.
pub(super) struct ParamSet {
    pub(super) slots: Vec<TensorSlot>,
}

impl ParamSet {
    /// The fixed order: embed, then each *unique* layer's nine tensors
    /// (`unique_layers` of them, not `num_layers` — see
    /// `ModelConfig::layer_group`), then the final norm gain.
    /// `ModelWeights::tensors()` uses the same order, which is what makes
    /// a downloaded set restore field-for-field.
    pub(super) fn shapes(config: &ModelConfig) -> Vec<(&'static str, usize, bool)> {
        let (h, ffn, kv, vocab) = (config.hidden_dim, config.ffn_dim(), config.kv_dim(), config.vocab_size());
        let mut out = vec![("embed", vocab * h, true)];
        for _ in 0..config.unique_layer_count() {
            out.extend_from_slice(&[
                ("attn_norm_gain", h, false),
                ("wq", h * h, true),
                ("wk", kv * h, true),
                ("wv", kv * h, true),
                ("wo", h * h, true),
                ("mlp_norm_gain", h, false),
                ("w_gate", ffn * h, true),
                ("w_up", ffn * h, true),
                ("w_down", h * ffn, true),
            ]);
        }
        out.push(("final_norm_gain", h, false));
        out
    }

    pub(super) fn zeros(ctx: &GpuContext, config: &ModelConfig, readable: bool) -> Self {
        let slots = Self::shapes(config)
            .into_iter()
            .map(|(label, len, decay)| TensorSlot {
                buffer: buffers::storage_f32(&ctx.device, label, len, readable),
                len,
                decay,
            })
            .collect();
        Self { slots }
    }

    pub(super) fn from_weights(ctx: &GpuContext, weights: &ModelWeights) -> Self {
        let mut slots = Vec::new();
        let mut push = |label: &str, data: &[f32], decay: bool| {
            slots.push(TensorSlot {
                buffer: buffers::upload_f32(&ctx.device, label, data, wgpu::BufferUsages::COPY_SRC),
                len: data.len(),
                decay,
            });
        };
        push("embed", &weights.embed, true);
        for layer in &weights.layers {
            push("attn_norm_gain", &layer.attn_norm_gain, false);
            push("wq", &layer.wq, true);
            push("wk", &layer.wk, true);
            push("wv", &layer.wv, true);
            push("wo", &layer.wo, true);
            push("mlp_norm_gain", &layer.mlp_norm_gain, false);
            push("w_gate", &layer.w_gate, true);
            push("w_up", &layer.w_up, true);
            push("w_down", &layer.w_down, true);
        }
        push("final_norm_gain", &weights.final_norm_gain, false);
        Self { slots }
    }

    pub(super) fn embed(&self) -> &wgpu::Buffer {
        &self.slots[0].buffer
    }

    /// `group` is a *weight-group* index (`0..unique_layer_count()`), not
    /// a depth position — callers walking depth positions (a step's
    /// `LayerLayout`) must resolve one to the other via
    /// `LayerLayout::group` first, the same way `ModelWeights.layers
    /// [group]` does on the CPU side.
    pub(super) fn layer(&self, group: usize, which: usize) -> &wgpu::Buffer {
        &self.slots[1 + group * TENSORS_PER_LAYER + which].buffer
    }

    pub(super) fn final_gain(&self, config: &ModelConfig) -> &wgpu::Buffer {
        &self.slots[1 + config.unique_layer_count() * TENSORS_PER_LAYER].buffer
    }
}

/// Everything the backward pass reads back out of the forward pass, for
/// one layer of one sequence. Same contents as `llm_core::model`'s
/// `LayerCache`; the SwiGLU activation is recomputed rather than stored,
/// exactly as the CPU reference does.
pub(super) struct LayerActs {
    pub(super) h_in: wgpu::Buffer, // residual stream entering the layer
    pub(super) normed1: wgpu::Buffer,
    pub(super) inv_rms1: wgpu::Buffer,
    pub(super) q: wgpu::Buffer,
    pub(super) k: wgpu::Buffer,
    pub(super) v: wgpu::Buffer,
    pub(super) probs: wgpu::Buffer, // [heads, t_len, band]
    pub(super) concat: wgpu::Buffer,
    pub(super) h_after_attn: wgpu::Buffer,
    pub(super) normed2: wgpu::Buffer,
    pub(super) inv_rms2: wgpu::Buffer,
    pub(super) gate: wgpu::Buffer,
    pub(super) up: wgpu::Buffer,
}

/// Scratch reused by every layer and every sequence in the batch.
pub(super) struct Scratch {
    pub(super) tokens: wgpu::Buffer,
    pub(super) targets: wgpu::Buffer,
    /// The per-sequence index the embedding scatter walks: the distinct
    /// token ids, where each one's positions start, and the positions
    /// themselves. Built on the host per sequence — it is `t_len` numbers
    /// of bookkeeping, and it turns the scatter from the most expensive
    /// dispatch of the step into one of the cheapest.
    pub(super) row_ids: wgpu::Buffer,
    pub(super) row_offsets: wgpu::Buffer,
    pub(super) row_positions: wgpu::Buffer,
    pub(super) hidden: wgpu::Buffer,
    pub(super) tmp_h: wgpu::Buffer,
    pub(super) h_final: wgpu::Buffer,
    pub(super) final_normed: wgpu::Buffer,
    pub(super) final_inv_rms: wgpu::Buffer,
    pub(super) logits: wgpu::Buffer,
    pub(super) d_logits: wgpu::Buffer,
    pub(super) loss_rows: wgpu::Buffer,
    pub(super) d_hidden: wgpu::Buffer,
    pub(super) d_a: wgpu::Buffer,
    pub(super) d_b: wgpu::Buffer,
    pub(super) d_c: wgpu::Buffer,
    pub(super) d_q: wgpu::Buffer,
    pub(super) d_k: wgpu::Buffer,
    pub(super) d_v: wgpu::Buffer,
    pub(super) d_score: wgpu::Buffer,
    pub(super) act: wgpu::Buffer,
    pub(super) d_act: wgpu::Buffer,
    pub(super) d_gate: wgpu::Buffer,
    pub(super) d_up: wgpu::Buffer,
    /// Per-workgroup partial sums, consumed by reduce_finish.
    pub(super) partials: wgpu::Buffer,
    /// Slot 0 is the summed loss; slot 1+i is tensor i's summed square.
    /// One small buffer, one readback per step - the only host-device
    /// synchronization a step has.
    pub(super) stats: wgpu::Buffer,
}
