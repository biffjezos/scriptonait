//! RMSNorm and SwiGLU, the two nonlinearities the transformer block uses.

/// RMSNorm forward. Returns `(y, inv_rms)`; `inv_rms` (one scalar per row)
/// must be kept for the backward pass.
pub fn rmsnorm_fwd(x: &[f32], gain: &[f32], rows: usize, dim: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; rows * dim];
    let mut inv_rms = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let n = 1.0 / (ms + eps).sqrt();
        inv_rms[r] = n;
        for i in 0..dim {
            y[r * dim + i] = row[i] * n * gain[i];
        }
    }
    (y, inv_rms)
}

/// RMSNorm backward. Returns `(dx, dgain)`; `dgain` is summed over rows
/// since the gain vector is shared across every row (token position).
pub fn rmsnorm_bwd(
    dy: &[f32],
    x: &[f32],
    gain: &[f32],
    inv_rms: &[f32],
    rows: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; rows * dim];
    let mut dgain = vec![0.0f32; dim];
    for r in 0..rows {
        let x_row = &x[r * dim..(r + 1) * dim];
        let dy_row = &dy[r * dim..(r + 1) * dim];
        let n = inv_rms[r];
        let mut s = 0.0f32;
        for i in 0..dim {
            dgain[i] += dy_row[i] * x_row[i] * n;
            s += dy_row[i] * gain[i] * x_row[i];
        }
        let n3_over_d = n * n * n / dim as f32;
        for i in 0..dim {
            dx[r * dim + i] = n * gain[i] * dy_row[i] - n3_over_d * x_row[i] * s;
        }
    }
    (dx, dgain)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// d(silu(x))/dx.
pub fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

/// `SwiGLU(gate, up) = silu(gate) * up`, elementwise.
pub fn swiglu_fwd(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// Returns `(dgate, dup)`.
pub fn swiglu_bwd(d_act: &[f32], gate: &[f32], up: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dgate = vec![0.0f32; gate.len()];
    let mut dup = vec![0.0f32; up.len()];
    for i in 0..gate.len() {
        dgate[i] = d_act[i] * up[i] * silu_grad(gate[i]);
        dup[i] = d_act[i] * silu(gate[i]);
    }
    (dgate, dup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::test_support::{assert_close, numerical_grad, seeded_vec};

    #[test]
    fn rmsnorm_gradient_check() {
        let rows = 3;
        let dim = 5;
        let x = seeded_vec(1, rows * dim);
        let gain = seeded_vec(2, dim).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let upstream = seeded_vec(3, rows * dim);
        let eps = 1e-4;

        let (y, inv_rms) = rmsnorm_fwd(&x, &gain, rows, dim, eps);
        let (dx, dgain) = rmsnorm_bwd(&upstream, &x, &gain, &inv_rms, rows, dim);
        let loss_of = |xx: &[f32], gg: &[f32]| {
            let (y, _) = rmsnorm_fwd(xx, gg, rows, dim, eps);
            y.iter().zip(&upstream).map(|(a, b)| a * b).sum::<f32>()
        };
        let _ = &y;

        let num_dx = numerical_grad(&x, |xx| loss_of(xx, &gain), 1e-3);
        assert_close(&dx, &num_dx, 5e-2, "rmsnorm dx");

        let num_dgain = numerical_grad(&gain, |gg| loss_of(&x, gg), 1e-3);
        assert_close(&dgain, &num_dgain, 5e-2, "rmsnorm dgain");
    }

    #[test]
    fn swiglu_gradient_check() {
        let n = 6;
        let gate = seeded_vec(30, n);
        let up = seeded_vec(31, n);
        let upstream = seeded_vec(32, n);

        let (dgate, dup) = swiglu_bwd(&upstream, &gate, &up);
        let loss_of = |g: &[f32], u: &[f32]| {
            swiglu_fwd(g, u).iter().zip(&upstream).map(|(a, b)| a * b).sum::<f32>()
        };
        let num_dgate = numerical_grad(&gate, |g| loss_of(g, &up), 1e-3);
        assert_close(&dgate, &num_dgate, 1e-3, "swiglu dgate");
        let num_dup = numerical_grad(&up, |u| loss_of(&gate, u), 1e-3);
        assert_close(&dup, &num_dup, 1e-3, "swiglu dup");
    }
}
