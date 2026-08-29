//! Numeric primitives shared by the forward and backward pass. Every `_fwd`
//! function has a matching `_bwd` that computes the exact analytic
//! gradient of that op; each is gradient-checked against finite
//! differences in its own module's test suite. `model.rs` composes these
//! into the full transformer; keeping them as free functions here makes
//! each one independently testable.
//!
//! Conventions used throughout:
//!   - Sequences are laid out row-major as `[T, dim]` (or `[T, heads*head_dim]`).
//!   - Linear layers store weights as `[out_dim, in_dim]` (PyTorch's
//!     `nn.Linear` convention) and have no bias, so `y = x @ w^T`.
//!   - No batch dimension here: `model.rs` loops one sequence at a time
//!     and accumulates gradients across the batch itself.

mod activation;
mod attention;
mod linear;
mod loss;
mod rope;

pub use activation::{rmsnorm_bwd, rmsnorm_fwd, sigmoid, silu, silu_grad, swiglu_bwd, swiglu_fwd};
pub use attention::{attention_bwd, attention_fwd, attention_step, band_width, softmax_row_inplace};
pub use linear::{linear_bwd, linear_fwd};
pub use loss::{argmax, cross_entropy};
pub use rope::{rope_apply, rope_apply_at};

/// Test-only helpers shared across every submodule's gradient checks.
///
/// `pub(super)` on each function rather than private: a submodule's own
/// test code (e.g. `attention::tests`) is a descendant of `ops`, not of
/// this module, so reaching these from there needs visibility starting
/// at `ops` and going back down — exactly what `pub(super)` grants.
#[cfg(test)]
mod test_support {
    use crate::rng::Rng;

    /// Central-difference numerical gradient of `f` wrt each element of `x`.
    pub(super) fn numerical_grad<F: Fn(&[f32]) -> f32>(x: &[f32], f: F, eps: f32) -> Vec<f32> {
        let mut g = vec![0.0f32; x.len()];
        let mut xm = x.to_vec();
        for i in 0..x.len() {
            let orig = xm[i];
            xm[i] = orig + eps;
            let f_plus = f(&xm);
            xm[i] = orig - eps;
            let f_minus = f(&xm);
            xm[i] = orig;
            g[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        g
    }

    pub(super) fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            let scale = a[i].abs().max(b[i].abs()).max(1.0);
            assert!(
                diff / scale < tol,
                "{label}[{i}]: analytic={} numeric={} diff={diff}",
                a[i],
                b[i]
            );
        }
    }

    pub(super) fn seeded_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = Rng::seed_from_u64(seed);
        (0..len).map(|_| rng.next_gaussian() * 0.5).collect()
    }
}
