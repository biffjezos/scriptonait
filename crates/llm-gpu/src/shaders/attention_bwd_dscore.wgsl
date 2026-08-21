// First of three attention-backward passes: computes d_score[h,t,:] (the
// softmax-backward result), fusing what ops::attention_bwd does as
// "compute d_probs for this row, then reduce, then scale" - all local to
// one thread (one row), so no cross-thread accumulation/atomics needed.
const MAX_WINDOW: u32 = 256u;

struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_out: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> probs: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_score: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let h = gid.y;
    if (t >= p.t_len || h >= p.heads) {
        return;
    }
    let hd = p.heads * p.head_dim;
    let base_t = t * hd + h * p.head_dim;
    let row_base = (h * p.t_len + t) * p.t_len;

    var lo: u32 = 0u;
    if (t + 1u > p.window) {
        lo = t + 1u - p.window;
    }
    let n = t - lo + 1u;

    var dprobs: array<f32, MAX_WINDOW>;
    var s_sum: f32 = 0.0;
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        let s = lo + idx;
        let base_s = s * hd + h * p.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + d_out[base_t + d] * v[base_s + d];
        }
        dprobs[idx] = dot;
        s_sum = s_sum + probs[row_base + s] * dot;
    }

    let scale = 1.0 / sqrt(f32(p.head_dim));
    for (var s: u32 = 0u; s < p.t_len; s = s + 1u) {
        d_score[row_base + s] = 0.0;
    }
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        let s = lo + idx;
        let pr = probs[row_base + s];
        d_score[row_base + s] = pr * (dprobs[idx] - s_sum) * scale;
    }
}
