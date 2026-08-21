//! Numeric primitives shared by the forward and backward pass. Every `_fwd`
//! function has a matching `_bwd` that computes the exact analytic
//! gradient of that op; each is gradient-checked against finite
//! differences in the test module below. `model.rs` composes these into
//! the full transformer; keeping them as free functions here makes each
//! one independently testable.
//!
//! Conventions used throughout:
//!   - Sequences are laid out row-major as `[T, dim]` (or `[T, heads*head_dim]`).
//!   - Linear layers store weights as `[out_dim, in_dim]` (PyTorch's
//!     `nn.Linear` convention) and have no bias, so `y = x @ w^T`.
//!   - No batch dimension here: `model.rs` loops one sequence at a time
//!     and accumulates gradients across the batch itself.

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

/// `y[rows,out_dim] = x[rows,in_dim] @ w[out_dim,in_dim]^T`, no bias.
pub fn linear_fwd(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let x_row = &x[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x_row[i] * w_row[i];
            }
            y[r * out_dim + o] = acc;
        }
    }
    y
}

/// Returns `(dx, dw)` for the same linear op.
pub fn linear_bwd(
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; rows * in_dim];
    let mut dw = vec![0.0f32; out_dim * in_dim];
    for r in 0..rows {
        let x_row = &x[r * in_dim..(r + 1) * in_dim];
        let dy_row = &dy[r * out_dim..(r + 1) * out_dim];
        for o in 0..out_dim {
            let dyo = dy_row[o];
            if dyo == 0.0 {
                continue;
            }
            let w_row = &w[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                dx[r * in_dim + i] += dyo * w_row[i];
                dw[o * in_dim + i] += dyo * x_row[i];
            }
        }
    }
    (dx, dw)
}

/// Applies rotary position embeddings in place to a `[rows, heads*head_dim]`
/// buffer, one rotation per (row, head, dim-pair) using `base=10000`
/// frequencies. `inverse` negates the rotation angle, which is exactly the
/// backward pass (rotation matrices are orthogonal, so the transpose is the
/// inverse rotation).
pub fn rope_apply(x: &mut [f32], rows: usize, heads: usize, head_dim: usize, inverse: bool) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let sign = if inverse { -1.0 } else { 1.0 };
    for t in 0..rows {
        for h in 0..heads {
            let base_idx = t * heads * head_dim + h * head_dim;
            for k in 0..half {
                let freq = 1.0f32 / 10000f32.powf(2.0 * k as f32 / head_dim as f32);
                let angle = sign * t as f32 * freq;
                let (s, c) = angle.sin_cos();
                let a = x[base_idx + 2 * k];
                let b = x[base_idx + 2 * k + 1];
                x[base_idx + 2 * k] = a * c - b * s;
                x[base_idx + 2 * k + 1] = a * s + b * c;
            }
        }
    }
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

/// In-place softmax over one row, treating `f32::NEG_INFINITY` entries as
/// masked (they come out exactly 0).
pub fn softmax_row_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        // Fully masked row (shouldn't happen with a causal mask that always
        // includes at least the diagonal) — leave as zeros.
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

/// Multi-head causal, optionally windowed, scaled dot-product attention.
/// `q`/`k`/`v` are `[T, heads*head_dim]` (`q`/`k` already RoPE'd). Returns
/// `(concat_out[T, heads*head_dim], probs[heads*T*T])`; `probs` is the
/// dense (masked entries = 0) attention matrix per head, cached for the
/// backward pass.
pub fn attention_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_len: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hd = heads * head_dim;
    let mut out = vec![0.0f32; t_len * hd];
    let mut probs = vec![0.0f32; heads * t_len * t_len];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..heads {
        for t in 0..t_len {
            let lo = t.saturating_sub(window.saturating_sub(1));
            let q_t = &q[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let row = &mut probs[h * t_len * t_len + t * t_len..h * t_len * t_len + t * t_len + t_len];
            for v_ in row.iter_mut() {
                *v_ = f32::NEG_INFINITY;
            }
            for s in lo..=t {
                let k_s = &k[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                let dot: f32 = q_t.iter().zip(k_s).map(|(a, b)| a * b).sum();
                row[s] = dot * scale;
            }
            softmax_row_inplace(row);
            let out_t = &mut out[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            for s in lo..=t {
                let p = row[s];
                if p == 0.0 {
                    continue;
                }
                let v_s = &v[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                for d in 0..head_dim {
                    out_t[d] += p * v_s[d];
                }
            }
        }
    }
    (out, probs)
}

/// Returns `(dq, dk, dv)`, all `[T, heads*head_dim]`.
pub fn attention_bwd(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t_len: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = heads * head_dim;
    let mut dq = vec![0.0f32; t_len * hd];
    let mut dk = vec![0.0f32; t_len * hd];
    let mut dv = vec![0.0f32; t_len * hd];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..heads {
        let probs_h = &probs[h * t_len * t_len..(h + 1) * t_len * t_len];
        let mut d_probs = vec![0.0f32; t_len * t_len];
        for t in 0..t_len {
            let lo = t.saturating_sub(window.saturating_sub(1));
            let d_out_t = &d_out[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let probs_row = &probs_h[t * t_len..t * t_len + t_len];
            for s in lo..=t {
                let v_s = &v[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                let dot: f32 = d_out_t.iter().zip(v_s).map(|(a, b)| a * b).sum();
                d_probs[t * t_len + s] = dot;
                let p = probs_row[s];
                if p != 0.0 {
                    let dv_s = &mut dv[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                    for d in 0..head_dim {
                        dv_s[d] += p * d_out_t[d];
                    }
                }
            }
        }
        // Softmax backward per row, then project into dq/dk.
        for t in 0..t_len {
            let lo = t.saturating_sub(window.saturating_sub(1));
            let probs_row = &probs_h[t * t_len..t * t_len + t_len];
            let dprobs_row = &d_probs[t * t_len..t * t_len + t_len];
            let s_sum: f32 = (lo..=t).map(|s| probs_row[s] * dprobs_row[s]).sum();
            let q_t = &q[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let dq_t = &mut dq[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            for s in lo..=t {
                let d_score = probs_row[s] * (dprobs_row[s] - s_sum) * scale;
                if d_score == 0.0 {
                    continue;
                }
                let k_s = &k[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                for d in 0..head_dim {
                    dq_t[d] += d_score * k_s[d];
                }
                let dk_s = &mut dk[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                for d in 0..head_dim {
                    dk_s[d] += d_score * q_t[d];
                }
            }
        }
    }
    (dq, dk, dv)
}

/// Index of the largest value (ties broken by first occurrence).
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Mean cross-entropy loss over `T` rows plus its gradient wrt the logits
/// (already divided by `T`, i.e. ready to use as `d_logits` for backward).
pub fn cross_entropy(logits: &[f32], targets: &[u32], t_len: usize, vocab: usize) -> (f32, Vec<f32>) {
    let mut d_logits = vec![0.0f32; t_len * vocab];
    let mut total_loss = 0.0f32;
    for t in 0..t_len {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut exps = vec![0.0f32; vocab];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - max).exp();
            exps[i] = e;
            sum += e;
        }
        let target = targets[t] as usize;
        let log_z = sum.ln() + max;
        total_loss += log_z - row[target];
        let d_row = &mut d_logits[t * vocab..(t + 1) * vocab];
        for i in 0..vocab {
            let p = exps[i] / sum;
            d_row[i] = (p - if i == target { 1.0 } else { 0.0 }) / t_len as f32;
        }
    }
    (total_loss / t_len as f32, d_logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference numerical gradient of `f` wrt each element of `x`.
    fn numerical_grad<F: Fn(&[f32]) -> f32>(x: &[f32], f: F, eps: f32) -> Vec<f32> {
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

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
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

    fn seeded_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = crate::rng::Rng::seed_from_u64(seed);
        (0..len).map(|_| rng.next_gaussian() * 0.5).collect()
    }

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
    fn linear_gradient_check() {
        let rows = 4;
        let in_dim = 3;
        let out_dim = 5;
        let x = seeded_vec(10, rows * in_dim);
        let w = seeded_vec(11, out_dim * in_dim);
        let upstream = seeded_vec(12, rows * out_dim);

        let dy = &upstream;
        let (dx, dw) = linear_bwd(dy, &x, &w, rows, in_dim, out_dim);

        let loss_of = |xx: &[f32], ww: &[f32]| {
            linear_fwd(xx, ww, rows, in_dim, out_dim)
                .iter()
                .zip(&upstream)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let num_dx = numerical_grad(&x, |xx| loss_of(xx, &w), 1e-3);
        assert_close(&dx, &num_dx, 1e-3, "linear dx");
        let num_dw = numerical_grad(&w, |ww| loss_of(&x, ww), 1e-3);
        assert_close(&dw, &num_dw, 1e-3, "linear dw");
    }

    #[test]
    fn rope_is_orthogonal_round_trip() {
        let rows = 4;
        let heads = 2;
        let head_dim = 4;
        let mut x = seeded_vec(20, rows * heads * head_dim);
        let orig = x.clone();
        rope_apply(&mut x, rows, heads, head_dim, false);
        rope_apply(&mut x, rows, heads, head_dim, true);
        assert_close(&x, &orig, 1e-4, "rope round trip");
    }

    #[test]
    fn rope_preserves_vector_norm_per_head() {
        let rows = 3;
        let heads = 2;
        let head_dim = 4;
        let mut x = seeded_vec(21, rows * heads * head_dim);
        let norm_before: Vec<f32> = (0..rows * heads)
            .map(|i| {
                let s = i * head_dim;
                x[s..s + head_dim].iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();
        rope_apply(&mut x, rows, heads, head_dim, false);
        let norm_after: Vec<f32> = (0..rows * heads)
            .map(|i| {
                let s = i * head_dim;
                x[s..s + head_dim].iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();
        assert_close(&norm_before, &norm_after, 1e-4, "rope norm preservation");
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

    #[test]
    fn attention_gradient_check_full_causal() {
        let t_len = 4;
        let heads = 2;
        let head_dim = 4;
        let hd = heads * head_dim;
        let window = t_len; // full causal, no windowing
        let q = seeded_vec(40, t_len * hd);
        let k = seeded_vec(41, t_len * hd);
        let v = seeded_vec(42, t_len * hd);
        let upstream = seeded_vec(43, t_len * hd);

        let (_, probs) = attention_fwd(&q, &k, &v, t_len, heads, head_dim, window);
        let (dq, dk, dv) = attention_bwd(&upstream, &q, &k, &v, &probs, t_len, heads, head_dim, window);

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32]| {
            let (out, _) = attention_fwd(qq, kk, vv, t_len, heads, head_dim, window);
            out.iter().zip(&upstream).map(|(a, b)| a * b).sum::<f32>()
        };
        let num_dq = numerical_grad(&q, |qq| loss_of(qq, &k, &v), 1e-3);
        assert_close(&dq, &num_dq, 5e-2, "attention dq");
        let num_dk = numerical_grad(&k, |kk| loss_of(&q, kk, &v), 1e-3);
        assert_close(&dk, &num_dk, 5e-2, "attention dk");
        let num_dv = numerical_grad(&v, |vv| loss_of(&q, &k, vv), 1e-3);
        assert_close(&dv, &num_dv, 5e-2, "attention dv");
    }

    #[test]
    fn attention_gradient_check_sliding_window() {
        let t_len = 6;
        let heads = 2;
        let head_dim = 4;
        let hd = heads * head_dim;
        let window = 3; // strictly less than t_len: exercises the mask
        let q = seeded_vec(50, t_len * hd);
        let k = seeded_vec(51, t_len * hd);
        let v = seeded_vec(52, t_len * hd);
        let upstream = seeded_vec(53, t_len * hd);

        let (_, probs) = attention_fwd(&q, &k, &v, t_len, heads, head_dim, window);
        let (dq, dk, dv) = attention_bwd(&upstream, &q, &k, &v, &probs, t_len, heads, head_dim, window);

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32]| {
            let (out, _) = attention_fwd(qq, kk, vv, t_len, heads, head_dim, window);
            out.iter().zip(&upstream).map(|(a, b)| a * b).sum::<f32>()
        };
        let num_dq = numerical_grad(&q, |qq| loss_of(qq, &k, &v), 1e-3);
        assert_close(&dq, &num_dq, 5e-2, "windowed attention dq");
        let num_dk = numerical_grad(&k, |kk| loss_of(&q, kk, &v), 1e-3);
        assert_close(&dk, &num_dk, 5e-2, "windowed attention dk");
        let num_dv = numerical_grad(&v, |vv| loss_of(&q, &k, vv), 1e-3);
        assert_close(&dv, &num_dv, 5e-2, "windowed attention dv");
    }

    #[test]
    fn attention_window_actually_masks_far_tokens() {
        // With window=1, position t can only attend to itself, so its
        // output must equal exactly V[t] (softmax of a single logit is 1).
        let t_len = 5;
        let heads = 1;
        let head_dim = 4;
        let q = seeded_vec(60, t_len * head_dim);
        let k = seeded_vec(61, t_len * head_dim);
        let v = seeded_vec(62, t_len * head_dim);
        let (out, _) = attention_fwd(&q, &k, &v, t_len, heads, head_dim, 1);
        assert_close(&out, &v, 1e-5, "window=1 attention == V");
    }

    #[test]
    fn cross_entropy_gradient_check() {
        let t_len = 3;
        let vocab = 6;
        let logits = seeded_vec(70, t_len * vocab);
        let targets = vec![1u32, 4, 0];

        let (loss, d_logits) = cross_entropy(&logits, &targets, t_len, vocab);
        assert!(loss > 0.0);

        // d(loss)/d(logits) checked directly (loss itself is the scalar).
        let num = numerical_grad(&logits, |l| cross_entropy(l, &targets, t_len, vocab).0, 1e-3);
        assert_close(&d_logits, &num, 1e-3, "cross entropy d_logits");
    }

    #[test]
    fn cross_entropy_matches_uniform_prediction_baseline() {
        // All-zero logits -> uniform distribution -> loss = ln(vocab).
        let t_len = 1;
        let vocab = 10;
        let logits = vec![0.0f32; vocab];
        let (loss, _) = cross_entropy(&logits, &[3], t_len, vocab);
        assert!((loss - (vocab as f32).ln()).abs() < 1e-4);
    }
}
