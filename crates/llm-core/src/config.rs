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

use crate::tokenizer::VOCAB_SIZE;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    /// Number of transformer blocks.
    pub num_layers: usize,
    /// Model/residual width ("nodes" in the simple UI).
    pub hidden_dim: usize,
    /// Number of attention heads. Must divide `hidden_dim`.
    pub num_heads: usize,
    /// Max sequence length the model is trained/run with.
    pub context_len: usize,
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
            context_len: 256,
            local_window: 256,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    HeadsMustDivideHidden { hidden_dim: usize, num_heads: usize },
    HeadDimMustBeEven { head_dim: usize },
    TooSmall { field: &'static str, min: usize },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::HeadsMustDivideHidden { hidden_dim, num_heads } => write!(
                f,
                "num_heads ({num_heads}) must evenly divide hidden_dim ({hidden_dim})"
            ),
            ConfigError::HeadDimMustBeEven { head_dim } => write!(
                f,
                "hidden_dim / num_heads ({head_dim}) must be even (needed for RoPE pairing)"
            ),
            ConfigError::TooSmall { field, min } => {
                write!(f, "{field} must be at least {min}")
            }
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
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
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
        VOCAB_SIZE
    }

    /// Total trainable scalar parameters.
    pub fn param_count(&self) -> usize {
        let v = self.vocab_size();
        let h = self.hidden_dim;
        let f = self.ffn_dim();

        let embedding = v * h; // weight-tied with the output head, counted once
        let per_layer_ple = v * h; // one PLE table per layer, same shape as embedding
        let per_layer_attn = h /* rmsnorm gain */ + 4 * h * h; // Wq,Wk,Wv,Wo
        let per_layer_mlp = h /* rmsnorm gain */ + 3 * h * f; // Wgate,Wup,Wdown
        let per_layer = per_layer_ple + per_layer_attn + per_layer_mlp;
        let final_norm = h;

        embedding + self.num_layers * per_layer + final_norm
    }

    /// Rough memory estimate in bytes, f32 throughout (WebGPU's baseline
    /// numeric type). `training` additionally accounts for gradients plus
    /// Adam's two moment buffers (so ~4x the raw weight bytes), which is
    /// the dominant cost; per-step activation memory is small at these
    /// sizes and scales with batch_size * context_len, not shown here.
    pub fn memory_bytes(&self, training: bool) -> usize {
        let bytes_per_param = 4;
        let weight_bytes = self.param_count() * bytes_per_param;
        if training {
            weight_bytes * 4 // weights + grad + adam.m + adam.v
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
        let cfg = ModelConfig { hidden_dim: 100, num_heads: 3, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::HeadsMustDivideHidden { .. })));
    }

    #[test]
    fn rejects_odd_head_dim() {
        // hidden_dim / num_heads = 5, odd -> RoPE can't pair dims.
        let cfg = ModelConfig { hidden_dim: 10, num_heads: 2, ..Default::default() };
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
    fn rejects_zero_local_window() {
        let cfg = ModelConfig { local_window: 0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(ConfigError::TooSmall { field: "local_window", .. })));
    }

    #[test]
    fn training_memory_is_four_x_inference() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.memory_bytes(true), cfg.memory_bytes(false) * 4);
    }

    #[test]
    fn small_config_stays_under_a_few_tens_of_mb() {
        let cfg = ModelConfig { num_layers: 4, hidden_dim: 128, num_heads: 4, context_len: 256, local_window: 256 };
        // Byte-level vocab keeps embedding/PLE tables tiny (tens of KB
        // each); the attention/MLP matrices dominate, but at this size
        // even full training memory (weights+grad+Adam) stays well under
        // 20 MB, far below what a browser tab can comfortably hold.
        assert!(cfg.memory_bytes(true) < 20 * 1024 * 1024);
    }
}
