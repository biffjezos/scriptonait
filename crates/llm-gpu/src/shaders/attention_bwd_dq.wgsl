// Second attention-backward pass: dQ[t] = sum_s d_score[t,s]*K[s] over
// valid s. Thread per (row, head) - the same access pattern as the
// forward pass, so it's a gather (each dQ[t] written by exactly one
// thread), no atomics needed.
struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_score: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> dq: array<f32>;

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

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        dq[base_t + d] = 0.0;
    }
    for (var s: u32 = lo; s <= t; s = s + 1u) {
        let ds = d_score[row_base + s];
        let base_s = s * hd + h * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            dq[base_t + d] = dq[base_t + d] + ds * k[base_s + d];
        }
    }
}
