//! Grouped-query, causal, sliding-window attention — forward, single-step
//! decode, and backward.

use super::linear::{axpy, dot};

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

/// Grouped-query, causal, optionally windowed, scaled dot-product
/// attention.
///
/// `q` is `[T, heads*head_dim]`; `k`/`v` are `[T, kv_heads*head_dim]`
/// (all already RoPE'd). Query head `h` reads key/value head
/// `h / (heads / kv_heads)`, so a group of query heads shares one KV
/// head — Llama's grouped-query attention. With `kv_heads == heads` this
/// is ordinary multi-head attention.
///
/// Sharing KV heads shrinks `Wk`/`Wv`, and shrinks the KV cache that
/// decoding keeps per token by the same factor. The KV cache is what
/// bounds how long a generation can run before it stops fitting in
/// memory, so this is the difference between holding a scene and holding
/// a chapter.
///
/// Returns `(concat_out[T, heads*head_dim], probs[heads*T*band])`, where
/// `band` is `band_width(t_len, window)`.
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
#[allow(clippy::too_many_arguments)]
pub fn attention_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(heads % kv_heads, 0);
    let hd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let group = heads / kv_heads;
    let band = band_width(t_len, window);
    let mut out = vec![0.0f32; t_len * hd];
    let mut probs = vec![0.0f32; heads * t_len * band];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..heads {
        let kvh = h / group;
        for t in 0..t_len {
            let lo = band_lo(t, band);
            let n = t - lo + 1; // in-window keys for this query
            let q_t = &q[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let base = h * t_len * band + t * band;
            let row = &mut probs[base..base + n];
            for (j, slot) in row.iter_mut().enumerate() {
                let s = lo + j;
                let k_s = &k[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim];
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
                axpy(out_t, &v[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim], p);
            }
        }
    }
    (out, probs)
}

/// Attention for a single query row against an existing KV cache — the
/// decoding step.
///
/// `q_row` is `[heads*head_dim]` for the new token; `k_cache`/`v_cache`
/// are `[cached_len, kv_heads*head_dim]` holding every key/value still
/// in the attention window, oldest first, already RoPE'd. Returns the
/// `[heads*head_dim]` attention output.
///
/// No probabilities are returned: nothing backpropagates through
/// decoding, and not materializing them is most of why one decode step
/// costs a fraction of re-running the whole forward pass.
pub fn attention_step(
    q_row: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cached_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(heads % kv_heads, 0);
    let hd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; hd];
    let mut scores = vec![0.0f32; cached_len];

    for h in 0..heads {
        let kvh = h / group;
        let q_h = &q_row[h * head_dim..(h + 1) * head_dim];
        for (s, score) in scores.iter_mut().enumerate() {
            let k_s = &k_cache[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim];
            *score = dot(q_h, k_s) * scale;
        }
        softmax_row_inplace(&mut scores);
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for (s, &p) in scores.iter().enumerate() {
            if p == 0.0 {
                continue;
            }
            axpy(out_h, &v_cache[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim], p);
        }
    }
    out
}

/// Returns `(dq, dk, dv)`; `dq` is `[T, heads*head_dim]` and `dk`/`dv`
/// are `[T, kv_heads*head_dim]`. `probs` is the banded cache
/// `attention_fwd` returned, same `window`.
///
/// With grouped-query attention every query head in a group accumulates
/// into the *same* `dk`/`dv` rows, which is exactly right: a shared KV
/// head's gradient is the sum over the queries that read it.
#[allow(clippy::too_many_arguments)]
pub fn attention_bwd(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    t_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(heads % kv_heads, 0);
    let hd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let group = heads / kv_heads;
    let band = band_width(t_len, window);
    let mut dq = vec![0.0f32; t_len * hd];
    let mut dk = vec![0.0f32; t_len * kvd];
    let mut dv = vec![0.0f32; t_len * kvd];
    let scale = 1.0 / (head_dim as f32).sqrt();
    // One row's worth of scratch, reused across rows and heads - the
    // gradient wrt the probabilities is only needed within the row it
    // belongs to (softmax rows are independent), so there's no reason to
    // materialize a second full-size cache alongside `probs`.
    let mut d_probs_row = vec![0.0f32; band];

    for h in 0..heads {
        let kvh = h / group;
        let probs_h = &probs[h * t_len * band..(h + 1) * t_len * band];
        for t in 0..t_len {
            let lo = band_lo(t, band);
            let n = t - lo + 1;
            let d_out_t = &d_out[t * hd + h * head_dim..t * hd + h * head_dim + head_dim];
            let probs_row = &probs_h[t * band..t * band + n];

            // dv, and the gradient arriving at each probability.
            for j in 0..n {
                let s = lo + j;
                let v_s = &v[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim];
                d_probs_row[j] = dot(d_out_t, v_s);
                let p = probs_row[j];
                if p != 0.0 {
                    axpy(
                        &mut dv[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim],
                        d_out_t,
                        p,
                    );
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
                let k_s = &k[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim];
                axpy(&mut dq[t * hd + h * head_dim..t * hd + h * head_dim + head_dim], k_s, d_score);
                axpy(
                    &mut dk[s * kvd + kvh * head_dim..s * kvd + kvh * head_dim + head_dim],
                    q_t,
                    d_score,
                );
            }
        }
    }
    (dq, dk, dv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::test_support::{assert_close, numerical_grad, seeded_vec};

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

        let (_, probs) = attention_fwd(&q, &k, &v, t_len, heads, heads, head_dim, window);
        let (dq, dk, dv) = attention_bwd(&upstream, &q, &k, &v, &probs, t_len, heads, heads, head_dim, window);

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32]| {
            let (out, _) = attention_fwd(qq, kk, vv, t_len, heads, heads, head_dim, window);
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

        let (_, probs) = attention_fwd(&q, &k, &v, t_len, heads, heads, head_dim, window);
        let (dq, dk, dv) = attention_bwd(&upstream, &q, &k, &v, &probs, t_len, heads, heads, head_dim, window);

        let loss_of = |qq: &[f32], kk: &[f32], vv: &[f32]| {
            let (out, _) = attention_fwd(qq, kk, vv, t_len, heads, heads, head_dim, window);
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
        let (_, probs) = attention_fwd(&x, &x, &x, t_len, heads, heads, head_dim, window);
        assert_eq!(probs.len(), heads * t_len * window);

        // A window at or above the context length degenerates to full
        // causal attention, and the band is then just the context.
        let (_, full) = attention_fwd(&x, &x, &x, t_len, heads, heads, head_dim, t_len * 4);
        assert_eq!(full.len(), heads * t_len * t_len);
    }

    #[test]
    fn banded_probs_rows_are_normalized_over_the_window() {
        let (t_len, heads, head_dim, window) = (16, 2, 4, 5);
        let mut x = vec![0.0f32; t_len * heads * head_dim];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.3;
        }
        let (_, probs) = attention_fwd(&x, &x, &x, t_len, heads, heads, head_dim, window);
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
        let (out, _) = attention_fwd(&q, &k, &v, t_len, heads, heads, head_dim, 1);
        assert_close(&out, &v, 1e-5, "window=1 attention == V");
    }
}
