//! Rotary position embeddings.

/// Applies rotary position embeddings in place to a `[rows, heads*head_dim]`
/// buffer, one rotation per (row, head, dim-pair). `inverse` negates the
/// rotation angle, which is exactly the backward pass (rotation matrices
/// are orthogonal, so the transpose is the inverse rotation).
///
/// `theta` is the frequency base (10000 in the original RoPE paper). A
/// larger base makes the low-frequency dimensions turn more slowly, which
/// is what lets a model address a longer context without the position
/// signal aliasing; it's a config knob because the right value depends on
/// the context length being trained.
///
/// `pos0` is the absolute position of the first row. It's 0 for a normal
/// forward pass over a whole sequence, and the current sequence length
/// when decoding one token at a time against a KV cache — RoPE is
/// applied at the moment a key is computed and then cached, so a cached
/// key carries the rotation for the absolute position it was written at.
pub fn rope_apply_at(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
    pos0: usize,
    inverse: bool,
) {
    debug_assert_eq!(head_dim % 2, 0);
    let half = head_dim / 2;
    let sign = if inverse { -1.0 } else { 1.0 };
    // The rotation angle depends only on (position, dim-pair) - not on the
    // head - so the `powf`/`sin_cos` pair is computed once per (t, k) and
    // reused across heads, instead of once per (t, h, k). Every layer runs
    // this four times per training step (q and k, forward and backward),
    // so the transcendentals dominated an otherwise trivial op.
    let inv_freq: Vec<f32> =
        (0..half).map(|k| 1.0f32 / theta.powf(2.0 * k as f32 / head_dim as f32)).collect();
    for t in 0..rows {
        let pos = (pos0 + t) as f32;
        for k in 0..half {
            let angle = sign * pos * inv_freq[k];
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

/// `rope_apply_at` starting from position 0 — the whole-sequence case.
pub fn rope_apply(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
    inverse: bool,
) {
    rope_apply_at(x, rows, heads, head_dim, theta, 0, inverse);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_ROPE_THETA;
    use crate::ops::test_support::{assert_close, seeded_vec};

    #[test]
    fn rope_is_orthogonal_round_trip() {
        let rows = 4;
        let heads = 2;
        let head_dim = 4;
        let mut x = seeded_vec(20, rows * heads * head_dim);
        let orig = x.clone();
        rope_apply(&mut x, rows, heads, head_dim, DEFAULT_ROPE_THETA, false);
        rope_apply(&mut x, rows, heads, head_dim, DEFAULT_ROPE_THETA, true);
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
        rope_apply(&mut x, rows, heads, head_dim, DEFAULT_ROPE_THETA, false);
        let norm_after: Vec<f32> = (0..rows * heads)
            .map(|i| {
                let s = i * head_dim;
                x[s..s + head_dim].iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();
        assert_close(&norm_before, &norm_after, 1e-4, "rope norm preservation");
    }
}
