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
//!   u64     tokens this model has been trained on, cumulative (v3+)
//!   u32     planned total steps for the schedule, 0 = none set (v4+)
//!   f32     plateau-cut multiplier on the learning rate, 1.0 = none (v4+)
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
//! size of the weights, and the GPU trainer keeps it in device memory
//! for the life of the run rather than round-tripping it through a
//! file.

use crate::bf16::from_bf16;
use crate::config::ModelConfig;
use crate::model::ModelWeights;
use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 4] = b"SCCK";
/// Version 4 added the schedule's planned-step target and the plateau-cut
/// multiplier, so a resumed run continues the same absolute schedule
/// instead of starting a fresh one anchored to whatever step it happened
/// to resume at. Version 3 added the cumulative token count. Files as old
/// as version 2 still load; fields newer than the file's version read as
/// their "not set" default rather than a wrong number.
///
/// Every version so far has only ever added *scalar* fields, read
/// conditionally by version (see `Checkpoint::from_bytes` below) —
/// there has never been more than one tensor group per layer to
/// describe. A future architecture needing a variable number of them
/// (Mixture-of-Experts: N sets of FFN weights per layer instead of one
/// — see `llm_core::model::layer`'s `ffn_forward` and this crate's
/// `PLAN.md`) would extend this the same way: bump `VERSION`, add a
/// conditionally-read field (an expert count), and teach
/// `ModelWeights::from_bytes` to size itself from it instead of purely
/// from `ModelConfig` as it does today. Not done here — there is no
/// expert count to store yet.
const VERSION: u32 = 4;
const MIN_READABLE_VERSION: u32 = 2;

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

pub struct Checkpoint {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub tokenizer: Tokenizer,
    /// Training step these weights came from, so a resumed run picks the
    /// learning-rate schedule back up where it left off instead of
    /// restarting warmup.
    pub step: u64,
    /// Tokens this model has been trained on, summed over every step of
    /// every run, and 0 when the file predates the count.
    ///
    /// The step count cannot stand in for this. A step is one batch, and
    /// a batch is however many sequences the machine could afford that
    /// day — so "step 4,573" means four times as much training at batch
    /// four as at batch one, and multiplying the step count by whatever
    /// the batch size happens to be *now* gets both runs wrong. This is
    /// the number that actually says how trained a model is, so it is
    /// counted as it happens and carried with the weights.
    pub tokens_seen: u64,
    /// The schedule's planned total steps, 0 when none has ever been set
    /// (files older than v4, or a model nobody has given a plan to yet).
    ///
    /// Carried here rather than left in memory so a resumed run continues
    /// the same absolute schedule — warmup already done, decay already
    /// under way — instead of restarting one anchored to whatever step it
    /// happens to resume at.
    pub planned_steps: u32,
    /// The plateau-cut multiplier on the learning rate, 1.0 (untouched)
    /// on files older than v4. See `train::TrainConfig::plateau_scale`.
    pub plateau_scale: f32,
}

/// Write a checkpoint from borrowed parts, at the requested width.
///
/// Borrowed, and streaming, for one reason: a 38M-parameter model is
/// 153 MB of f32 weights. Cloning them to build a `Checkpoint`, then
/// serializing that to an f32 `Vec`, then converting that to bf16, holds
/// four copies at once inside a wasm heap that also contains the live
/// weights and the copy just downloaded from the GPU. That is around
/// 750 MB, it is how an export ends in `rust_oom`, and an allocation
/// failure in wasm is not an error anybody catches — it aborts the
/// module and takes the model with it.
///
/// This holds one copy: the output.
#[allow(clippy::too_many_arguments)]
pub fn write_checkpoint(
    config: &ModelConfig,
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    step: u64,
    tokens_seen: u64,
    planned_steps: u32,
    plateau_scale: f32,
    dtype: WeightDtype,
) -> Vec<u8> {
    let tokenizer_bytes = tokenizer.to_bytes();
    let bf16 = dtype == WeightDtype::Bf16;
    let weight_len = weights.param_count() * if bf16 { 2 } else { 4 };
    let mut out = Vec::with_capacity(64 + tokenizer_bytes.len() + weight_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    for value in [
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.context_len,
        config.local_window,
        config.vocab_size,
    ] {
        out.extend_from_slice(&(value as u32).to_le_bytes());
    }
    out.extend_from_slice(&config.rope_theta.to_le_bytes());
    out.extend_from_slice(&u32::from(config.use_ple).to_le_bytes());
    out.extend_from_slice(&step.to_le_bytes());
    out.extend_from_slice(&tokens_seen.to_le_bytes());
    out.extend_from_slice(&planned_steps.to_le_bytes());
    out.extend_from_slice(&plateau_scale.to_le_bytes());
    out.extend_from_slice(&dtype.tag().to_le_bytes());
    out.extend_from_slice(&(tokenizer_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&tokenizer_bytes);
    out.extend_from_slice(&(weight_len as u32).to_le_bytes());
    weights.write_into(&mut out, bf16);
    out
}

impl Checkpoint {
    /// Serialize at full precision — the trainer's own format.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_with(WeightDtype::F32)
    }

    /// Serialize with the weights in `dtype`. `Bf16` is what gets
    /// published to the site.
    pub fn to_bytes_with(&self, dtype: WeightDtype) -> Vec<u8> {
        write_checkpoint(
            &self.config,
            &self.weights,
            &self.tokenizer,
            self.step,
            self.tokens_seen,
            self.planned_steps,
            self.plateau_scale,
            dtype,
        )
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a scriptonait checkpoint".to_string());
        }
        let version = r.u32()?;
        if !(MIN_READABLE_VERSION..=VERSION).contains(&version) {
            return Err(format!(
                "checkpoint format version {version}, expected {MIN_READABLE_VERSION} to {VERSION}"
            ));
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
        // Absent before version 3. Zero reads as "not recorded" rather
        // than as "trained on nothing", and the page distinguishes them.
        let tokens_seen = if version >= 3 { r.u64()? } else { 0 };
        // Absent before version 4. 0 planned_steps and a 1.0 plateau_scale
        // are both "not set" — the schedule falls back to whatever the
        // caller's own default is, same as if nothing had ever set them.
        let (planned_steps, plateau_scale) =
            if version >= 4 { (r.u32()?, r.f32()?) } else { (0, 1.0) };
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
        Ok(Checkpoint { config, weights, tokenizer, step, tokens_seen, planned_steps, plateau_scale })
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
        Checkpoint {
            config,
            weights,
            tokenizer,
            step: 4242,
            tokens_seen: 9_000_000,
            planned_steps: 23_000,
            plateau_scale: 0.5,
        }
    }

    #[test]
    fn round_trips() {
        let original = sample();
        let restored = Checkpoint::from_bytes(&original.to_bytes()).unwrap();
        assert_eq!(restored.config, original.config);
        assert_eq!(restored.step, original.step);
        assert_eq!(restored.tokens_seen, original.tokens_seen);
        assert_eq!(restored.planned_steps, original.planned_steps);
        assert_eq!(restored.plateau_scale, original.plateau_scale);
        assert_eq!(restored.weights.to_bytes(), original.weights.to_bytes());
        assert_eq!(restored.tokenizer.to_bytes(), original.tokenizer.to_bytes());
    }

    /// A file written before the token count existed still loads, and
    /// reads as "not recorded" rather than as a wrong number.
    #[test]
    fn a_version_2_checkpoint_still_loads() {
        let mut bytes = sample().to_bytes();
        // Rewrite the version and cut everything from tokens_seen up to
        // (not including) the dtype tag, which version 2 did not have:
        // tokens_seen (u64) + planned_steps (u32) + plateau_scale (f32).
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        let step_at = 8 + 7 * 4 + 4 + 4;
        bytes.drain(step_at + 8..step_at + 8 + 8 + 4 + 4);
        let restored = Checkpoint::from_bytes(&bytes).expect("version 2 should still load");
        assert_eq!(restored.step, 4242);
        assert_eq!(restored.tokens_seen, 0);
        assert_eq!(restored.planned_steps, 0);
        assert_eq!(restored.plateau_scale, 1.0);
    }

    /// A file written before the schedule fields existed still loads, with
    /// its token count intact and the schedule fields at their defaults.
    #[test]
    fn a_version_3_checkpoint_still_loads() {
        let mut bytes = sample().to_bytes();
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        // Cut planned_steps (u32) + plateau_scale (f32), which sit right
        // after tokens_seen.
        let tokens_seen_at = 8 + 7 * 4 + 4 + 4 + 8;
        bytes.drain(tokens_seen_at + 8..tokens_seen_at + 8 + 4 + 4);
        let restored = Checkpoint::from_bytes(&bytes).expect("version 3 should still load");
        assert_eq!(restored.step, 4242);
        assert_eq!(restored.tokens_seen, 9_000_000);
        assert_eq!(restored.planned_steps, 0);
        assert_eq!(restored.plateau_scale, 1.0);
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
