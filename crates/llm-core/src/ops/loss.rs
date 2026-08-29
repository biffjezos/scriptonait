//! The training loss and the greedy-decoding tie-breaker, grouped
//! together because both read the same `[T, vocab]` logits layout and
//! neither belongs to just the training or just the generation caller.

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
    use crate::ops::test_support::{assert_close, numerical_grad, seeded_vec};

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
