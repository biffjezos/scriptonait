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
//!   u32     weight dtype: 0 = f32, 1 = bf16
//!   u32     tokenizer length, then that many bytes (see tokenizer.rs)
//!   u32     weight length in bytes, then the weights
//! ```
//!
//! ## bf16
//!
//! The copy served to browsers is stored as bf16 — the top 16 bits of
//! each f32 — which halves the download for a file people wait on before
//! the page can do anything. bf16 rather than f16 because the conversion
//! is a shift and a rounding add in each direction, with no subnormal or
//! overflow handling to get subtly wrong: bf16 has f32's exponent range
//! and simply fewer mantissa bits. Inference at this model size doesn't
//! miss them (it is what large models are *trained* in), and the trainer
//! keeps f32 on disk for its own checkpoints, so nothing accumulates
//! rounding across a resume.
//!
//! Optimizer state is deliberately *not* here: it is three times the
//! size of the weights and only a resuming trainer wants it (see
//! `llm-train`, which writes it to a separate file that CI caches rather
//! than commits).

use crate::config::ModelConfig;
use crate::model::ModelWeights;
use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 4] = b"SCCK";
const VERSION: u32 = 2;

/// How the weights are stored in a checkpoint file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// Full precision. What the trainer writes for its own resumes.
    F32,
    /// Half the bytes, same exponent range. What the site ships.
    Bf16,
}

impl WeightDtype {
    fn tag(self) -> u32 {
        match self {
            WeightDtype::F32 => 0,
            WeightDtype::Bf16 => 1,
        }
    }

    fn from_tag(tag: u32) -> Result<Self, String> {
        match tag {
            0 => Ok(WeightDtype::F32),
            1 => Ok(WeightDtype::Bf16),
            other => Err(format!("unknown weight dtype {other}")),
        }
    }
}

/// f32 -> bf16: keep the top 16 bits, rounding to nearest even.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Keep it a NaN rather than letting the rounding add turn it
        // into an infinity.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

/// bf16 -> f32: the bits are already an f32 prefix.
fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

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
    /// Serialize at full precision — the trainer's own format.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_with(WeightDtype::F32)
    }

    /// Serialize with the weights in `dtype`. `Bf16` is what gets
    /// published to the site.
    pub fn to_bytes_with(&self, dtype: WeightDtype) -> Vec<u8> {
        let tokenizer_bytes = self.tokenizer.to_bytes();
        let f32_bytes = self.weights.to_bytes();
        let weight_bytes = match dtype {
            WeightDtype::F32 => f32_bytes,
            WeightDtype::Bf16 => f32_bytes
                .chunks_exact(4)
                .flat_map(|c| {
                    to_bf16(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()
                })
                .collect(),
        };
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
        out.extend_from_slice(&dtype.tag().to_le_bytes());
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
        let dtype = WeightDtype::from_tag(r.u32()?)?;

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
        let stored = r.take(weight_len)?;
        let weights = match dtype {
            WeightDtype::F32 => ModelWeights::from_bytes(stored, &config)?,
            WeightDtype::Bf16 => {
                if stored.len() % 2 != 0 {
                    return Err("bf16 weights have an odd byte count".to_string());
                }
                let widened: Vec<u8> = stored
                    .chunks_exact(2)
                    .flat_map(|c| {
                        from_bf16(u16::from_le_bytes(c.try_into().unwrap())).to_le_bytes()
                    })
                    .collect();
                ModelWeights::from_bytes(&widened, &config)?
            }
        };
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
    fn bf16_round_trips_within_its_precision() {
        for value in [0.0f32, 1.0, -1.0, 0.5, -0.017, 3.4e30, -2.1e-30, f32::MIN_POSITIVE] {
            let back = from_bf16(to_bf16(value));
            let tolerance = value.abs() * 0.01 + 1e-38;
            assert!(
                (back - value).abs() <= tolerance,
                "bf16 round trip of {value} gave {back}"
            );
        }
        assert!(from_bf16(to_bf16(f32::NAN)).is_nan());
        assert_eq!(from_bf16(to_bf16(f32::INFINITY)), f32::INFINITY);
    }

    #[test]
    fn bf16_checkpoints_are_half_the_size_and_close_enough() {
        let original = sample();
        let f32_bytes = original.to_bytes();
        let bf16_bytes = original.to_bytes_with(WeightDtype::Bf16);
        // Only the weights shrink; the header and tokenizer don't.
        assert!(
            bf16_bytes.len() < f32_bytes.len() * 3 / 5,
            "bf16 file is {} bytes against f32's {}",
            bf16_bytes.len(),
            f32_bytes.len()
        );

        let restored = Checkpoint::from_bytes(&bf16_bytes).unwrap();
        assert_eq!(restored.config, original.config);
        assert_eq!(restored.step, original.step);
        let worst = restored
            .weights
            .to_bytes()
            .chunks_exact(4)
            .zip(original.weights.to_bytes().chunks_exact(4))
            .map(|(a, b)| {
                let (a, b) = (
                    f32::from_le_bytes(a.try_into().unwrap()),
                    f32::from_le_bytes(b.try_into().unwrap()),
                );
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 0.01, "bf16 weights drifted by {worst}");
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
