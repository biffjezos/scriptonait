// dq[t, h] = sum_j d_score[h, t, j] * k[band_lo(t) + j, h / group].
//
// One workgroup of 64 threads per (row, head), each thread owning one
// feature of the head and walking the window for it: a gather, so each
// output is written by exactly one thread and no atomics are needed.
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

const THREADS: u32 = 64u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wid.x;
    let h = wid.y;
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

    for (var d: u32 = lid.x; d < p.head_dim; d = d + THREADS) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            let base_k = (lo + j) * kvd + kvh * p.head_dim;
            acc = acc + d_score[row + j] * k[base_k + d];
        }
        dq[base_t + d] = acc;
    }
}
