//! A tiny, dependency-free deterministic PRNG (SplitMix64) used for weight
//! initialization and batch sampling. Not cryptographically secure and not
//! meant to be — just seedable and reproducible, which is all training
//! needs, without pulling in the `rand` crate.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit output (SplitMix64).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (bound must be > 0).
    pub fn gen_range(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "gen_range bound must be > 0");
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform f32 in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a value evenly spread over [0, 1).
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Standard normal sample via Box-Muller.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::seed_from_u64(7);
        let mut b = Rng::seed_from_u64(7);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = Rng::seed_from_u64(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn gen_range_stays_in_bounds() {
        let mut r = Rng::seed_from_u64(123);
        for _ in 0..1000 {
            let v = r.gen_range(17);
            assert!(v < 17);
        }
    }

    #[test]
    fn next_f32_in_unit_interval() {
        let mut r = Rng::seed_from_u64(999);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn gaussian_has_roughly_zero_mean_and_unit_variance() {
        let mut r = Rng::seed_from_u64(42);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| r.next_gaussian()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.1, "mean was {mean}");
        assert!((var - 1.0).abs() < 0.15, "var was {var}");
    }
}
