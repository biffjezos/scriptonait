// First of three attention-backward kernels: the softmax backward for
// each (row, head), leaving d_score in the same banded layout as
// `probs`. Mirrors the middle of llm_core::ops::attention_bwd.
//
// d_score doubles as the scratch for d_probs (pass one writes d_probs
// there and accumulates the row's reduction; pass two turns it into
// d_score in place), so no second banded buffer and no per-thread array
// is needed.
struct Params {
    t_len: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    band: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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
    let kvd = p.kv_heads * p.head_dim;
    let group = p.heads / p.kv_heads;
    let kvh = h / group;
    let base_t = t * hd + h * p.head_dim;
    let row = (h * p.t_len + t) * p.band;

    var lo: u32 = 0u;
    if (t + 1u > p.band) {
        lo = t + 1u - p.band;
    }
    let n = t - lo + 1u;

    var s_sum: f32 = 0.0;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let base_v = (lo + j) * kvd + kvh * p.head_dim;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            acc = acc + d_out[base_t + d] * v[base_v + d];
        }
        d_score[row + j] = acc;
        s_sum = s_sum + probs[row + j] * acc;
    }
    for (var j: u32 = n; j < p.band; j = j + 1u) {
        d_score[row + j] = 0.0;
    }

    let scale = 1.0 / sqrt(f32(p.head_dim));
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        d_score[row + j] = probs[row + j] * (d_score[row + j] - s_sum) * scale;
    }
}
