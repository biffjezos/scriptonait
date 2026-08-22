//! One file that holds everything needed to run the model: its shape,
//! its weights, and the tokenizer those weights were trained against.
//!
//! Keeping the tokenizer *inside* the checkpoint is the point. Token id
//! 4,312 means whatever the merge list says it means; weights trained
//! against one vocabulary and run against another produce fluent-looking
//! nonsense with no error anywhere. Shipping them as one file makes that
//! mismatch impossible to construct by accident — the browser fetches a
//! single `.ckpt` and cannot get half of it.
//!
//! ```text
//!   magic   "SCCK"
//!   u32     format version
//!   u32     num_layers, hidden_dim, num_heads, num_kv_heads,
//!           context_len, local_window, vocab_size
//!   f32     rope_theta
//!   u32     use_ple (0/1)
//!   u64     training step the weights are from
//!   u32     tokenizer length, then that many bytes (see tokenizer.rs)
//!   u32     weight length in bytes, then that many f32 LE
//! ```
//!
//! Optimizer state is deliberately *not* here: it is three times the
//! size of the weights and only a resuming trainer wants it (see
//! `llm-train`, which writes it to a separate file that CI caches rather
//! than commits).

use crate::config::ModelConfig;
use crate::model::ModelWeights;
use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 4] = b"SCCK";
const VERSION: u32 = 1;

pub struct Checkpoint {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    /// Training step these weights came from, so a resumed run picks the
    /// learning-rate schedule back up where it left off instead of
    /// restarting warmup.
    pub step: u64,
}

impl Checkpoint {
    pub fn to_bytes(&self) -> Vec<u8> {
        let tokenizer_bytes = self.tokenizer.to_bytes();
        let weight_bytes = self.weights.to_bytes();
        let mut out = Vec::with_capacity(64 + tokenizer_bytes.len() + weight_bytes.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        for value in [
            self.config.num_layers,
            self.config.hidden_dim,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.context_len,
            self.config.local_window,
            self.config.vocab_size,
        ] {
            out.extend_from_slice(&(value as u32).to_le_bytes());
        }
        out.extend_from_slice(&self.config.rope_theta.to_le_bytes());
        out.extend_from_slice(&u32::from(self.config.use_ple).to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&(tokenizer_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&tokenizer_bytes);
        out.extend_from_slice(&(weight_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&weight_bytes);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a scriptonait checkpoint".to_string());
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(format!("checkpoint format version {version}, expected {VERSION}"));
        }
        let config = ModelConfig {
            num_layers: r.u32()? as usize,
            hidden_dim: r.u32()? as usize,
            num_heads: r.u32()? as usize,
            num_kv_heads: r.u32()? as usize,
            context_len: r.u32()? as usize,
            local_window: r.u32()? as usize,
            vocab_size: r.u32()? as usize,
            rope_theta: r.f32()?,
            use_ple: r.u32()? != 0,
        };
        config.validate().map_err(|e| format!("checkpoint config is invalid: {e}"))?;
        let step = r.u64()?;

        let tokenizer_len = r.u32()? as usize;
        let tokenizer = Tokenizer::from_bytes(r.take(tokenizer_len)?)?;
        // The one consistency check that matters: if these disagree, the
        // model is indexing an embedding table with ids from a different
        // vocabulary.
        if tokenizer.vocab_size() != config.vocab_size {
            return Err(format!(
                "checkpoint is inconsistent: config says vocab_size {}, its tokenizer has {}",
                config.vocab_size,
                tokenizer.vocab_size()
            ));
        }

        let weight_len = r.u32()? as usize;
        let weights = ModelWeights::from_bytes(r.take(weight_len)?, &config)?;
        Ok(Checkpoint { config, weights, tokenizer, step })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.at + n > self.bytes.len() {
            return Err(format!(
                "checkpoint truncated: wanted {n} bytes at offset {}, file is {} bytes",
                self.at,
                self.bytes.len()
            ));
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Checkpoint {
        let tokenizer = Tokenizer::train(&[&"the cave and the fire and the shadows. ".repeat(30)], 320);
        let config = ModelConfig {
            num_layers: 2,
            hidden_dim: 16,
            num_heads: 4,
            num_kv_heads: 2,
            context_len: 32,
            local_window: 16,
            vocab_size: tokenizer.vocab_size(),
            rope_theta: 50000.0,
            use_ple: false,
        };
        let weights = ModelWeights::init(&config, 5);
        Checkpoint { config, weights, tokenizer, step: 4242 }
    }

    #[test]
    fn round_trips() {
        let original = sample();
        let restored = Checkpoint::from_bytes(&original.to_bytes()).unwrap();
        assert_eq!(restored.config, original.config);
        assert_eq!(restored.step, original.step);
        assert_eq!(restored.weights.to_bytes(), original.weights.to_bytes());
        assert_eq!(restored.tokenizer.to_bytes(), original.tokenizer.to_bytes());
    }

    #[test]
    fn rejects_a_truncated_file() {
        let bytes = sample().to_bytes();
        for cut in [0, 4, 20, bytes.len() / 2, bytes.len() - 1] {
            assert!(Checkpoint::from_bytes(&bytes[..cut]).is_err(), "accepted a {cut}-byte file");
        }
    }

    #[test]
    fn rejects_a_foreign_file() {
        assert!(Checkpoint::from_bytes(b"not a checkpoint at all").is_err());
    }

    #[test]
    fn rejects_a_tokenizer_that_does_not_match_the_config() {
        // The failure this format exists to prevent: weights indexed by
        // one vocabulary, run against another.
        let mut original = sample();
        original.config.vocab_size += 1;
        original.weights = ModelWeights::init(&original.config, 5);
        let err = match Checkpoint::from_bytes(&original.to_bytes()) {
            Err(e) => e,
            Ok(_) => panic!("accepted a checkpoint whose tokenizer and config disagree"),
        };
        assert!(err.contains("inconsistent"), "unexpected error: {err}");
    }
}
