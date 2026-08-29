//! Tiled matmul, forward and backward, plus the SIMD-friendly dot-product
//! micro-kernels it's built from.

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
///
/// `pub(super)` rather than private: `attention.rs` reuses this same
/// kernel directly (it isn't blocked the way the matmuls here are), and
/// both live under `ops` as siblings.
#[inline]
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
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
///
/// `pub(super)`: shared with `attention.rs`, same reason as `dot` above.
#[inline]
pub(super) fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::test_support::{assert_close, numerical_grad, seeded_vec};

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
}
