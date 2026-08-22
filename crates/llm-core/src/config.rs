//! Model topology configuration: the handful of "layers / nodes" knobs the
//! UI exposes, plus derived sizes and a memory estimator so the UI can warn
//! the user before they pick a config that won't fit in the browser tab.
//!
//! Architecture (a small Llama/Gemma-style decoder-only transformer):
//!   - byte-level tokenizer, vocab_size = 259 (256 bytes + PAD/BOS/EOS)
//!   - one shared input embedding table, weight-tied with the output head
//!   - per layer: RMSNorm -> RoPE causal (optionally sliding-window) self
//!                attention -> residual
//!                RMSNorm -> SwiGLU MLP -> residual
//!                + a per-layer embedding (PLE) added into the residual
//!                  stream, gathered by token id (Gemma-3n-style): a plain
//!                  vector lookup with no matmul, so it costs no GPU
//!                  compute time.
//!   - final RMSNorm before the tied output projection
//!
//! The default vertical slice (byte-level tokens + sliding-window
//! attention) is tuned for long-form narrative material — film scripts,
//! stories, books — where documents are long but attention only needs to
//! reach back a bounded number of tokens to pick up local structure
//! (a scene, a line of dialogue, a paragraph), so `local_window` can stay
//! small and cheap even as `context_len` is pushed up to fit a whole scene
//! or chapter.

use crate::tokenizer::BASE_VOCAB_SIZE;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Model/residual width ("nodes" in the simple UI).
    pub hidden_dim: usize,
    /// Number of attention heads. Must divide `hidden_dim`.
    pub num_heads: usize,
    /// Number of key/value heads (grouped-query attention). Must divide
    /// `num_heads`; equal to it means ordinary multi-head attention.
    ///
    /// Query heads are cheap — they read from the residual stream and
    /// throw the result away each step. Key/value heads are expensive
    /// twice over: they add parameters to `Wk`/`Wv`, and every token
    /// generated has to keep its keys and values around for the rest of
    /// the generation. Sharing one KV head across a group of query heads
    /// is the standard way to buy attention capacity without paying for
    /// either, and it's what makes a long generation's KV cache fit.
    pub num_kv_heads: usize,
    /// Max sequence length the model is trained/run with.
    pub context_len: usize,
    /// Size of the tokenizer's vocabulary, which fixes the embedding
    /// table and the tied output head. It is not a free knob: it must
    /// match the tokenizer the weights were trained with, or every token
    /// id means something different. `BASE_VOCAB_SIZE` is plain byte
    /// level (no merges learned).
    pub vocab_size: usize,
    /// RoPE frequency base. 10000 is the original paper's value and is
    /// right for short contexts; larger bases slow the low-frequency
    /// rotations down so positions stay distinguishable further out.
    pub rope_theta: f32,
    /// Whether each layer carries its own embedding table (Gemma 3n's
    /// per-layer embeddings), gathered by token id and added straight
    /// into that layer's residual stream.
    ///
    /// Off by default now, and that's a real reversal: with the old
    /// 259-token byte vocabulary a PLE table was a few tens of KB and
    /// essentially free. At an 8k BPE vocabulary each one is
    /// `vocab * hidden` — the same size as the entire input embedding —
    /// so a 6-layer model spends *more* parameters on PLE tables than on
    /// attention and MLP combined. That is not where this size of model
    /// should be spending them.
    pub use_ple: bool,
    /// Sliding-window attention span (Mistral-style): each position only
    /// attends to the `local_window` tokens before it instead of the full
    /// causal history. Attention cost scales as `context_len * local_window`
    /// instead of `context_len^2`, which matters once `context_len` is
    /// pushed up for book/script-length material. Set equal to (or above)
    /// `context_len` to disable and use plain full causal attention.
    pub local_window: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 4,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            context_len: 256,
            local_window: 256,
            vocab_size: BASE_VOCAB_SIZE,
            rope_theta: DEFAULT_ROPE_THETA,
            use_ple: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    HeadsMustDivideHidden { hidden_dim: usize, num_heads: usize },
    KvHeadsMustDivideHeads { num_heads: usize, num_kv_heads: usize },
    HeadDimMustBeEven { head_dim: usize },
    TooSmall { field: &'static str, min: usize },
    /// The config's training memory can't fit in a 32-bit wasm heap. See
    /// `MAX_TRAINING_BYTES`.
    TooLarge { training_bytes: usize, limit: usize },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::HeadsMustDivideHidden { hidden_dim, num_heads } => write!(
                f,
                "num_heads ({num_heads}) must evenly divide hidden_dim ({hidden_dim})"
            ),
            ConfigError::KvHeadsMustDivideHeads { num_heads, num_kv_heads } => write!(
                f,
                "num_kv_heads ({num_kv_heads}) must evenly divide num_heads ({num_heads})"
            ),
            ConfigError::HeadDimMustBeEven { head_dim } => write!(
                f,
                "hidden_dim / num_heads ({head_dim}) must be even (needed for RoPE pairing)"
            ),
            ConfigError::TooSmall { field, min } => {
                write!(f, "{field} must be at least {min}")
            }
            ConfigError::TooLarge { training_bytes, limit } => write!(
                f,
                "this shape needs about {} MB to train, over the {} MB a browser tab can \
                 address — reduce context length, attention window, layers, or nodes \
                 (the attention cache alone costs layers x heads x context x window floats)",
                training_bytes / (1024 * 1024),
                limit / (1024 * 1024),
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Llama-style SwiGLU hidden size: 2/3 of the usual 4x MLP expansion,
/// rounded up to a multiple of 32 so it's a friendly size for GPU tiling.
pub fn default_ffn_dim(hidden_dim: usize) -> usize {
    let raw = (hidden_dim * 8) / 3;
    let multiple = 32;
    raw.div_ceil(multiple) * multiple
}

/// Ceiling on a config's estimated training memory.
///
/// wasm32 tops out at a 4 GiB linear heap and browsers refuse to grow one
/// anywhere near that in practice, so a config past this point doesn't
/// train slowly — it allocates until the tab dies, and takes the machine
/// into swap on the way there. Rejecting it up front, with a message that
/// names the knob to turn, beats letting the UI offer a shape that can
/// only ever end that way.
pub const MAX_TRAINING_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// RoPE frequency base used unless a config says otherwise.
pub const DEFAULT_ROPE_THETA: f32 = 10000.0;

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_layers == 0 {
            return Err(ConfigError::TooSmall { field: "num_layers", min: 1 });
        }
        if self.hidden_dim < 8 {
            return Err(ConfigError::TooSmall { field: "hidden_dim", min: 8 });
        }
        if self.num_heads == 0 {
            return Err(ConfigError::TooSmall { field: "num_heads", min: 1 });
        }
        if self.context_len < 2 {
            return Err(ConfigError::TooSmall { field: "context_len", min: 2 });
        }
        if self.local_window == 0 {
            return Err(ConfigError::TooSmall { field: "local_window", min: 1 });
        }
        if self.vocab_size < BASE_VOCAB_SIZE {
            // Below this the tokenizer's own byte alphabet doesn't fit,
            // so some byte would have no embedding row at all.
            return Err(ConfigError::TooSmall { field: "vocab_size", min: BASE_VOCAB_SIZE });
        }
        if self.hidden_dim % self.num_heads != 0 {
            return Err(ConfigError::HeadsMustDivideHidden {
                hidden_dim: self.hidden_dim,
                num_heads: self.num_heads,
            });
        }
        let head_dim = self.hidden_dim / self.num_heads;
        if head_dim % 2 != 0 {
            return Err(ConfigError::HeadDimMustBeEven { head_dim });
        }
        if self.num_kv_heads == 0 {
            return Err(ConfigError::TooSmall { field: "num_kv_heads", min: 1 });
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(ConfigError::KvHeadsMustDivideHeads {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        if self.rope_theta <= 1.0 {
            return Err(ConfigError::TooSmall { field: "rope_theta", min: 2 });
        }
        let training_bytes = self.memory_bytes(true);
        if training_bytes > MAX_TRAINING_BYTES {
            return Err(ConfigError::TooLarge { training_bytes, limit: MAX_TRAINING_BYTES });
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Width of the key and value projections: `num_kv_heads * head_dim`,
    /// which equals `hidden_dim` only when attention isn't grouped.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// The window actually used at inference/training time, clamped to
    /// `context_len` (a `local_window` larger than `context_len` is just
    /// full attention).
    pub fn effective_window(&self) -> usize {
        self.local_window.min(self.context_len)
    }

    pub fn ffn_dim(&self) -> usize {
        default_ffn_dim(self.hidden_dim)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Total trainable scalar parameters.
    pub fn param_count(&self) -> usize {
        let v = self.vocab_size();
        let h = self.hidden_dim;
        let f = self.ffn_dim();

        let kv = self.kv_dim();

        let embedding = v * h; // weight-tied with the output head, counted once
        // One PLE table per layer, same shape as the input embedding -
        // which is why it's off by default at a BPE-sized vocabulary.
        let per_layer_ple = if self.use_ple { v * h } else { 0 };
        // Wq[h,h], Wo[h,h], and Wk/Wv[kv,h] - grouped-query attention
        // shrinks exactly those two.
        let per_layer_attn = h /* rmsnorm gain */ + 2 * h * h + 2 * kv * h;
        let per_layer_mlp = h /* rmsnorm gain */ + 3 * h * f; // Wgate,Wup,Wdown
        let per_layer = per_layer_ple + per_layer_attn + per_layer_mlp;
        let final_norm = h;

        embedding + self.num_layers * per_layer + final_norm
    }

    /// Bytes of activation memory one training step holds live, f32
    /// throughout.
    ///
    /// This is *not* the rounding error the old estimate implied by
    /// leaving it out. `forward` keeps a full activation cache for every
    /// layer because `backward` needs all of it, and the
    /// attention-probability part of that cache is
    /// `num_layers * num_heads * context_len * effective_window` floats —
    /// at a long context that single term dwarfs every parameter in the
    /// model. Surfacing it is what lets the UI say "this config needs
    /// 3 GB" *before* the tab dies trying.
    ///
    /// Batch size doesn't appear here: `train.rs` runs one sequence at a
    /// time and accumulates gradients, so a bigger batch costs time, not
    /// activation memory.
    pub fn activation_bytes(&self) -> usize {
        let t = self.context_len;
        let h = self.hidden_dim;
        let f = self.ffn_dim();
        let band = self.effective_window();

        // Per layer, from model.rs's LayerCache: h_after_ple, normed1, q,
        // concat, h_after_attn, normed2 (6 x [T,h]), k and v (2 x [T,kv]),
        // gate and up (2 x [T,f]), the banded attention probabilities, and
        // the two per-row inv_rms vectors.
        let kv = self.kv_dim();
        let per_layer =
            6 * t * h + 2 * t * kv + 2 * t * f + self.num_heads * t * band + 2 * t;
        // Plus the residual stream and final-norm cache, logits and
        // d_logits, and - generously - the handful of [T,h]/[T,f]
        // temporaries backward holds while walking one layer.
        let shared = 3 * t * h + 2 * t * self.vocab_size() + 6 * t * h + 3 * t * f;

        (self.num_layers * per_layer + shared) * 4
    }

    /// Rough memory estimate in bytes, f32 throughout (WebGPU's baseline
    /// numeric type). `training` accounts for gradients plus Adam's two
    /// moment buffers (~4x the raw weight bytes) *and* the per-step
    /// activation memory from `activation_bytes`.
    pub fn memory_bytes(&self, training: bool) -> usize {
        let bytes_per_param = 4;
        let weight_bytes = self.param_count() * bytes_per_param;
        if training {
            // weights + grad + adam.m + adam.v, plus one step's activations
            weight_bytes * 4 + self.activation_bytes()
        } else {
            weight_bytes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(ModelConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_heads_not_dividing_hidden() {
        let cfg = ModelConfig { hidden_dim: 100, num_heads: 3, num_kv_heads: 3, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::HeadsMustDivideHidden { .. })));
    }

    #[test]
    fn rejects_kv_heads_that_do_not_divide_heads() {
        let cfg = ModelConfig { num_heads: 6, num_kv_heads: 4, hidden_dim: 96, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::KvHeadsMustDivideHeads { .. })));
        let ok = ModelConfig { num_heads: 6, num_kv_heads: 3, hidden_dim: 96, ..Default::default() };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn grouped_query_attention_shrinks_the_parameter_count() {
        let mha = ModelConfig { num_heads: 8, num_kv_heads: 8, hidden_dim: 256, ..Default::default() };
        let gqa = ModelConfig { num_kv_heads: 2, ..mha };
        assert!(gqa.param_count() < mha.param_count());
        assert_eq!(gqa.kv_dim(), 2 * gqa.head_dim());
    }

    #[test]
    fn per_layer_embeddings_dominate_at_a_bpe_vocabulary() {
        // The reason use_ple defaults to false now: one PLE table per
        // layer is the size of the whole input embedding.
        let off = ModelConfig { hidden_dim: 256, num_heads: 8, num_kv_heads: 8, vocab_size: 8192, ..Default::default() };
        let on = ModelConfig { use_ple: true, ..off };
        assert!(on.param_count() > 2 * off.param_count());
    }

    #[test]
    fn rejects_odd_head_dim() {
        // hidden_dim / num_heads = 5, odd -> RoPE can't pair dims.
        let cfg = ModelConfig { hidden_dim: 10, num_heads: 2, num_kv_heads: 2, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::HeadDimMustBeEven { .. })));
    }

    #[test]
    fn ffn_dim_is_multiple_of_32() {
        assert_eq!(default_ffn_dim(128) % 32, 0);
        assert_eq!(default_ffn_dim(128), 352); // 128*8/3 = 341.33 -> ceil to 352
    }

    #[test]
    fn param_count_grows_with_layers() {
        let small = ModelConfig { num_layers: 2, ..Default::default() };
        let big = ModelConfig { num_layers: 8, ..Default::default() };
        assert!(big.param_count() > small.param_count());
    }

    #[test]
    fn effective_window_clamps_to_context_len() {
        let cfg = ModelConfig { context_len: 100, local_window: 1000, ..Default::default() };
        assert_eq!(cfg.effective_window(), 100);
        let cfg2 = ModelConfig { context_len: 1000, local_window: 128, ..Default::default() };
        assert_eq!(cfg2.effective_window(), 128);
    }

    #[test]
    fn rejects_a_config_too_big_for_a_wasm_heap() {
        // The largest shape the UI's own input limits allowed: 16 layers,
        // 1024 nodes, 16 heads, full 4096-token attention. Its activation
        // cache alone is ~20 GB, which used to be reported to the user as
        // "3.1 GB while training" and then simply killed the tab.
        let cfg = ModelConfig { num_layers: 16, hidden_dim: 1024, num_heads: 16, num_kv_heads: 16, context_len: 4096, local_window: 4096, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::TooLarge { .. })));
        assert!(ModelConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_zero_local_window() {
        let cfg = ModelConfig { local_window: 0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::TooSmall { field: "local_window", .. })));
    }

    #[test]
    fn training_memory_includes_optimizer_state_and_activations() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.memory_bytes(true), cfg.memory_bytes(false) * 4 + cfg.activation_bytes());
    }

    #[test]
    fn activation_memory_follows_the_window_not_context_squared() {
        // The whole point of sliding-window attention: doubling the
        // context at a fixed window must not quadruple anything.
        let narrow = ModelConfig { context_len: 2048, local_window: 128, ..Default::default() };
        let wide = ModelConfig { context_len: 4096, local_window: 128, ..Default::default() };
        assert!(
            wide.activation_bytes() < narrow.activation_bytes() * 3,
            "narrow={} wide={}",
            narrow.activation_bytes(),
            wide.activation_bytes()
        );
        // ...whereas full attention at that context genuinely is enormous
        // next to the weights, and the estimate has to say so.
        let full = ModelConfig { context_len: 4096, local_window: 4096, ..Default::default() };
        assert!(full.activation_bytes() > 10 * full.memory_bytes(false));
    }

    #[test]
    fn small_config_stays_under_a_few_tens_of_mb() {
        let cfg = ModelConfig { num_layers: 4, hidden_dim: 128, num_heads: 4, num_kv_heads: 4, context_len: 256, local_window: 256, ..Default::default() };
        // Byte-level vocab keeps embedding/PLE tables tiny (tens of KB
        // each); the attention/MLP matrices dominate the weights, and the
        // per-step activation cache is the same order again at this
        // context length. Both together still fit comfortably in a browser
        // tab - unlike the long-context configs the UI also allows, which
        // is exactly why the estimate counts activations at all.
        assert!(cfg.memory_bytes(true) < 48 * 1024 * 1024);
        assert!(cfg.memory_bytes(true) > cfg.memory_bytes(false) * 4);
    }
}
