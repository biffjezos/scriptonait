//! f32 <-> bf16 conversion.
//!
//! Its own module rather than living in either neighbor: `model.rs`
//! writes bf16 checkpoint bytes (`ModelWeights::write_into`) and
//! `checkpoint.rs` reads them back, so putting the conversion in either
//! file would make the other import from it — a needless conceptual
//! cycle between two modules that otherwise have nothing to do with each
//! other. Both depend one-way on this instead.

/// f32 -> bf16: keep the top 16 bits, rounding to nearest even.
pub(crate) fn to_bf16(x: f32) -> u16 {
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
pub(crate) fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
