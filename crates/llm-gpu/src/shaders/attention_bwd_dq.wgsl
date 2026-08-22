// dq[t, h] = sum_j d_score[h, t, j] * k[band_lo(t) + j, h / group]. Same
// access pattern as the forward pass, so each dq row is written by
// exactly one thread: a gather, no atomics.
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

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        dq[base_t + d] = 0.0;
    }
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let ds = d_score[row + j];
        let base_k = (lo + j) * kvd + kvh * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            dq[base_t + d] = dq[base_t + d] + ds * k[base_k + d];
        }
    }
}
