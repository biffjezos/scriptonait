// Multi-head causal, sliding-window scaled dot-product attention.
// Forward only (mirrors ops::attention_fwd, minus the probs cache since
// there's no backward pass on this backend). One thread per (row, head);
// each thread needs local scratch for its own attention row, which is why
// `window` is capped at MAX_WINDOW — see llm-gpu's `MAX_GPU_WINDOW`
// (the Rust side refuses to use this backend for a larger window instead
// of silently truncating it).
const MAX_WINDOW: u32 = 256u;

struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let h = gid.y;
    if (t >= p.t_len || h >= p.heads) {
        return;
    }
    let hd = p.heads * p.head_dim;
    let base_t = t * hd + h * p.head_dim;

    var lo: u32 = 0u;
    if (t + 1u > p.window) {
        lo = t + 1u - p.window;
    }
    // Number of valid key/value positions s in [lo, t].
    let n = t - lo + 1u;
    let scale = 1.0 / sqrt(f32(p.head_dim));

    var scores: array<f32, MAX_WINDOW>;
    var maxv: f32 = -3.0e38;
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        let s = lo + idx;
        let base_s = s * hd + h * p.head_dim;
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base_t + d] * k[base_s + d];
        }
        dot = dot * scale;
        scores[idx] = dot;
        if (dot > maxv) {
            maxv = dot;
        }
    }

    var sum: f32 = 0.0;
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        scores[idx] = exp(scores[idx] - maxv);
        sum = sum + scores[idx];
    }

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        out[base_t + d] = 0.0;
    }
    for (var idx: u32 = 0u; idx < n; idx = idx + 1u) {
        let s = lo + idx;
        let prob = scores[idx] / sum;
        let base_s = s * hd + h * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            out[base_t + d] = out[base_t + d] + prob * v[base_s + d];
        }
    }
}
