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

/// Dot product with four independent accumulators.
///
/// Float addition isn't associative, so a single-accumulator loop forces
/// the compiler into one serial dependency chain - no vectorization, and
/// one FP-add latency per element. Four partial sums break that chain and
/// let LLVM emit SIMD (real SIMD on wasm when built with
/// `-C target-feature=+simd128`; see the deploy workflow). The summation
/// order differs from a naive left fold, which is fine: both are equally
/// valid float reductions, and this one is if anything slightly more
/// accurate.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let mut ac = a.chunks_exact(4);
    let mut bc = b.chunks_exact(4);
    for (av, bv) in ac.by_ref().zip(bc.by_ref()) {
        acc[0] += av[0] * bv[0];
        acc[1] += av[1] * bv[1];
        acc[2] += av[2] * bv[2];
        acc[3] += av[3] * bv[3];
    }
    let tail: f32 = ac.remainder().iter().zip(bc.remainder()).map(|(x, y)| x * y).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `dst += scale * src`, elementwise. Written over `chunks_exact` for the
/// same reason as `dot`: it drops the per-element bounds check and gives
/// LLVM a shape it will vectorize.
#[inline]
fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let mut dc = dst.chunks_exact_mut(4);
    let mut sc = src.chunks_exact(4);
    for (d, s) in dc.by_ref().zip(sc.by_ref()) {
        d[0] += scale * s[0];
        d[1] += scale * s[1];
        d[2] += scale * s[2];
        d[3] += scale * s[3];
    }
    for (d, s) in dc.into_remainder().iter_mut().zip(sc.remainder()) {
        *d += scale * s;
    }
}

/// How many rows (or output channels) the blocked kernels below process
/// against one shared operand at a time.
///
/// The matmuls in this file are all "contract along the last axis" —
/// every output element is a dot product of one `x` row with one `w`
/// row — so the naive loop order reads the *entire* weight matrix once
/// per token. For the tied output head that matrix is `vocab * hidden`
/// floats; at an 8k BPE vocab that's tens of MB streamed from RAM per
/// token, and the arithmetic sits waiting on memory the whole time.
///
/// Processing `BLOCK` rows against each weight row before moving on cuts
/// that traffic by a factor of `BLOCK` — the weight row is loaded once
/// and used four times, and the four `x` rows it's used against stay in
/// L1. Four is deliberate: the kernels hold `BLOCK * 4` float lanes of
/// accumulator, which is 16 vector registers — the whole SSE2 register
/// file, and half of AVX's. Eight spills.
const BLOCK: usize = 4;

/// Dot one weight row against up to `BLOCK` consecutive rows of `x`,
/// reading the weight row once for all of them.
///
/// The body is `BLOCK` plain `dot` calls, not one hand-fused kernel over
/// five interleaved iterators: the fused version looks like it should be
/// faster and measures three times *slower*, because interleaving that
/// many `chunks_exact` iterators is a shape LLVM gives up vectorizing.
/// The win here is loop order, not the inner kernel — `w_row` is loaded
/// from L1 for all `n` rows instead of from RAM for one.
#[inline]
fn dot_block(x_block: &[f32], in_dim: usize, n: usize, w_row: &[f32]) -> [f32; BLOCK] {
    debug_assert!(n <= BLOCK);
    debug_assert_eq!(w_row.len(), in_dim);
    let mut out = [0.0f32; BLOCK];
    for r in 0..n {
        out[r] = dot(&x_block[r * in_dim..(r + 1) * in_dim], w_row);
    }
    out
}

/// `dst_row_j += scales[j] * src` for up to `BLOCK` consecutive rows of
/// `dst`, reading `src` once for all of them. The mirror image of
/// `dot_block`, and the reason both backward passes stream their large
/// operand a quarter as many times as the naive loop would.
#[inline]
fn axpy_block(dst_block: &mut [f32], dim: usize, n: usize, scales: &[f32; BLOCK], src: &[f32]) {
    debug_assert!(n <= BLOCK);
    debug_assert_eq!(src.len(), dim);
    for j in 0..n {
        if scales[j] == 0.0 {
            continue;
        }
        axpy(&mut dst_block[j * dim..(j + 1) * dim], src, scales[j]);
    }
}

/// `y[rows,out_dim] = x[rows,in_dim] @ w[out_dim,in_dim]^T`, no bias.
pub fn linear_fwd(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    let mut r0 = 0;
    while r0 < rows {
        let n = BLOCK.min(rows - r0);
        let x_block = &x[r0 * in_dim..(r0 + n) * in_dim];
        for o in 0..out_dim {
            let vals = dot_block(x_block, in_dim, n, &w[o * in_dim..(o + 1) * in_dim]);
            for (r, v) in vals.iter().take(n).enumerate() {
                y[(r0 + r) * out_dim + o] = *v;
            }
        }
        r0 += BLOCK;
    }
    y
}

/// Returns `(dx, dw)` for the same linear op.
///
/// `dx` and `dw` are accumulated in separate passes rather than one fused
/// loop. The fused version rereads and rewrites a whole `dw` row
/// (`out_dim * in_dim` floats, far past L1 for real layer sizes) for every
/// single token, which is what made this the most expensive op in the
/// backward pass; keeping the two passes apart lets each one stream over
/// memory it can actually keep hot. Both passes are then blocked by
/// `BLOCK` (see `dot_block`), so the operand that doesn't fit in cache is
/// streamed a quarter as many times.
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

    // dx[r] = sum_o dy[r,o] * w[o]   -- w[o] read once per block of rows.
    let mut r0 = 0;
    while r0 < rows {
        let n = BLOCK.min(rows - r0);
        let dx_block = &mut dx[r0 * in_dim..(r0 + n) * in_dim];
        for o in 0..out_dim {
            let mut scales = [0.0f32; BLOCK];
            let mut any = false;
            for (r, s) in scales.iter_mut().take(n).enumerate() {
                *s = dy[(r0 + r) * out_dim + o];
                any |= *s != 0.0;
            }
            if !any {
                continue;
            }
            axpy_block(dx_block, in_dim, n, &scales, &w[o * in_dim..(o + 1) * in_dim]);
        }
        r0 += BLOCK;
    }

    // dw[o] = sum_r dy[r,o] * x[r]   -- x[r] read once per block of
    // output channels, and the block of dw rows being accumulated into
    // (BLOCK * in_dim floats) stays in L1 across the whole row loop.
    let mut o0 = 0;
    while o0 < out_dim {
        let n = BLOCK.min(out_dim - o0);
        let dw_block = &mut dw[o0 * in_dim..(o0 + n) * in_dim];
        for r in 0..rows {
            let mut scales = [0.0f32; BLOCK];
            let mut any = false;
            for (j, s) in scales.iter_mut().take(n).enumerate() {
                *s = dy[r * out_dim + o0 + j];
                any |= *s != 0.0;
            }
            if !any {
                continue;
            }
            axpy_block(dw_block, in_dim, n, &scales, &x[r * in_dim..(r + 1) * in_dim]);
        }
        o0 += BLOCK;
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
    // The rotation angle depends only on (position, dim-pair) - not on the
    // head - so the `powf`/`sin_cos` pair is computed once per (t, k) and
    // reused across heads, instead of once per (t, h, k). Every layer runs
    // this four times per training step (q and k, forward and backward),
    // so the transcendentals dominated an otherwise trivial op.
    let inv_freq: Vec<f32> =
        (0..half).map(|k| 1.0f32 / 10000f32.powf(2.0 * k as f32 / head_dim as f32)).collect();
    for t in 0..rows {
        for k in 0..half {
            let angle = sign * t as f32 * inv_freq[k];
            let (s, c) = angle.sin_cos();
            for h in 0..heads {
                let base_idx = t * heads * head_dim + h * head_dim;
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

/// Number of stored attention-probability columns per query row: the
/// sliding window, clamped to the sequence length. See `attention_fwd`.
pub fn band_width(t_len: usize, window: usize) -> usize {
    window.max(1).min(t_len)
}

/// First key position attended by query position `t` under a `band`-wide
/// sliding window.
#[inline]
fn band_lo(t: usize, band: usize) -> usize {
    t.saturating_sub(band.saturating_sub(1))
}

/// Multi-head causal, optionally windowed, scaled dot-product attention.
/// `q`/`k`/`v` are `[T, heads*head_dim]` (`q`/`k` already RoPE'd). Returns
/// `(concat_out[T, heads*head_dim], probs[heads*T*band])`, where `band` is
/// `band_width(t_len, window)`.
///
/// `probs` is stored **banded**, not dense: row `t` holds only the
/// `band` in-window columns, with column `j` meaning key position
/// `band_lo(t, band) + j` (entries past the causal diagonal stay 0).
/// A dense `[heads, T, T]` cache would be quadratic in context length —
/// at, say, 8 heads and a 4096-token context that is 512 MB *per layer,
/// per sequence*, all of it live at once because backward needs every
/// layer's copy. It also made the whole point of sliding-window
/// attention moot: masking and re-normalizing a full-length row is O(T)
/// work per query no matter how narrow the window is. Banded storage
/// makes both the memory and the time genuinely O(T * window).
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
    let band = band_width(t_len, window);
    let mut out = vec![0.0f32; t_len * hd];
    let mut probs = vec![0.0f32; heads * t_len * band];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..heads {
        for t in 0..t_len {
            let lo = band_lo(t, band);
            let n = t - lo + 1; // in-window keys for this query
            let q_t = &q[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let base = h * t_len * band + t * band;
            let row = &mut probs[base..base + n];
            for (j, slot) in row.iter_mut().enumerate() {
                let s = lo + j;
                let k_s = &k[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                *slot = dot(q_t, k_s) * scale;
            }
            softmax_row_inplace(row);
            let out_t = &mut out[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            for j in 0..n {
                let p = probs[base + j];
                if p == 0.0 {
                    continue;
                }
                let s = lo + j;
                axpy(out_t, &v[s * hd + h * head_dim..s * hd + h * head_dim + head_dim], p);
            }
        }
    }
    (out, probs)
}

/// Returns `(dq, dk, dv)`, all `[T, heads*head_dim]`. `probs` is the
/// banded cache `attention_fwd` returned, same `window`.
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
    let band = band_width(t_len, window);
    let mut dq = vec![0.0f32; t_len * hd];
    let mut dk = vec![0.0f32; t_len * hd];
    let mut dv = vec![0.0f32; t_len * hd];
    let scale = 1.0 / (head_dim as f32).sqrt();
    // One row's worth of scratch, reused across rows and heads - the
    // gradient wrt the probabilities is only needed within the row it
    // belongs to (softmax rows are independent), so there's no reason to
    // materialize a second full-size cache alongside `probs`.
    let mut d_probs_row = vec![0.0f32; band];

    for h in 0..heads {
        let probs_h = &probs[h * t_len * band..(h + 1) * t_len * band];
        for t in 0..t_len {
            let lo = band_lo(t, band);
            let n = t - lo + 1;
            let d_out_t = &d_out[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let probs_row = &probs_h[t * band..t * band + n];

            // dv, and the gradient arriving at each probability.
            for j in 0..n {
                let s = lo + j;
                let v_s = &v[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                d_probs_row[j] = dot(d_out_t, v_s);
                let p = probs_row[j];
                if p != 0.0 {
                    axpy(&mut dv[s * hd + h * head_dim..s * hd + h * head_dim + head_dim], d_out_t, p);
                }
            }

            // Softmax backward for this row, then project into dq/dk.
            let s_sum: f32 = (0..n).map(|j| probs_row[j] * d_probs_row[j]).sum();
            let q_t = &q[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            for j in 0..n {
                let d_score = probs_row[j] * (d_probs_row[j] - s_sum) * scale;
                if d_score == 0.0 {
                    continue;
                }
                let s = lo + j;
                let k_s = &k[s * hd + h * head_dim..s * hd + h * head_dim + head_dim];
                axpy(&mut dq[t * hd + h * head_dim..t * hd + h * head_dim + head_dim], k_s, d_score);
                axpy(&mut dk[s * hd + h * head_dim..s * hd + h * head_dim + head_dim], q_t, d_score);
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
    fn banded_probs_cache_is_sized_by_the_window_not_the_context() {
        // A dense [heads, T, T] cache is what made long-context training
        // allocate gigabytes; this is the invariant that keeps it linear
        // in the window.
        let (t_len, heads, head_dim, window) = (64, 2, 4, 8);
        let x = vec![0.05f32; t_len * heads * head_dim];
        let (_, probs) = attention_fwd(&x, &x, &x, t_len, heads, head_dim, window);
        assert_eq!(probs.len(), heads * t_len * window);

        // A window at or above the context length degenerates to full
        // causal attention, and the band is then just the context.
        let (_, full) = attention_fwd(&x, &x, &x, t_len, heads, head_dim, t_len * 4);
        assert_eq!(full.len(), heads * t_len * t_len);
    }

    #[test]
    fn banded_probs_rows_are_normalized_over_the_window() {
        let (t_len, heads, head_dim, window) = (16, 2, 4, 5);
        let mut x = vec![0.0f32; t_len * heads * head_dim];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.3;
        }
        let (_, probs) = attention_fwd(&x, &x, &x, t_len, heads, head_dim, window);
        for h in 0..heads {
            for t in 0..t_len {
                let lo = t.saturating_sub(window - 1);
                let n = t - lo + 1;
                let row = &probs[h * t_len * window + t * window..h * t_len * window + t * window + window];
                let sum: f32 = row[..n].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "h={h} t={t} sum={sum}");
                // Slots past the causal diagonal are never written.
                assert!(row[n..].iter().all(|&p| p == 0.0), "h={h} t={t} tail not zero");
            }
        }
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
