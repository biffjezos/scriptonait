// Grouped-query, causal, sliding-window attention over a whole sequence,
// with the banded probability cache the backward pass needs. Mirrors
// llm_core::ops::attention_fwd exactly, including its storage layout:
// `probs` is [heads, t_len, band] and entry j of row t is the key at
// absolute position `band_lo(t) + j`, not at position j.
//
// One workgroup of 64 threads per (row, head), not one thread.
//
// One thread per (row, head) is 1536 threads for a 256-token sequence
// with six heads - a few percent of what a GPU needs to stay busy, each
// thread walking the whole window scalar. Here the 64 threads of a
// workgroup split the window between them for the scores, reduce in
// workgroup memory for the softmax, and then split the head dimension
// for the value accumulation, which is 64x the parallelism for the same
// arithmetic.
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
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> probs: array<f32>;

const THREADS: u32 = 64u;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wid.x;
    let h = wid.y;
    let lane = lid.x;
    // Guarding with a return would put the barriers below in non-uniform
    // control flow; the workgroup runs and writes nothing instead.
    let live = t < p.t_len && h < p.heads;

    let hd = p.heads * p.head_dim;
    let kvd = p.kv_heads * p.head_dim;
    let group = p.heads / p.kv_heads;
    let kvh = select(0u, h / group, live);
    let base_q = select(0u, t * hd + h * p.head_dim, live);
    let row = select(0u, (h * p.t_len + t) * p.band, live);
    var lo: u32 = 0u;
    if (live && t + 1u > p.band) {
        lo = t + 1u - p.band;
    }
    let n = select(0u, t - lo + 1u, live);
    let scale = 1.0 / sqrt(f32(p.head_dim));

    // Scores, and this row's maximum, split over the window.
    var lane_max: f32 = -3.0e38;
    for (var j: u32 = lane; j < n; j = j + THREADS) {
        let base_k = (lo + j) * kvd + kvh * p.head_dim;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            acc = acc + q[base_q + d] * k[base_k + d];
        }
        let s = acc * scale;
        probs[row + j] = s;
        lane_max = max(lane_max, s);
    }
    partial[lane] = lane_max;
    workgroupBarrier();
    for (var stride: u32 = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial[lane] = max(partial[lane], partial[lane + stride]);
        }
        workgroupBarrier();
    }
    let maxv = partial[0];
    workgroupBarrier();

    // exp, and the sum, again split over the window.
    var lane_sum: f32 = 0.0;
    for (var j: u32 = lane; j < n; j = j + THREADS) {
        let e = exp(probs[row + j] - maxv);
        probs[row + j] = e;
        lane_sum = lane_sum + e;
    }
    // Entries past this row's window are never read, but a stale value
    // from an earlier sequence would be a silent wrong number if that
    // changed. Zero them.
    for (var j: u32 = n + lane; j < p.band; j = j + THREADS) {
        probs[row + j] = 0.0;
    }
    partial[lane] = lane_sum;
    workgroupBarrier();
    for (var stride: u32 = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
    }
    let sum = partial[0];
    workgroupBarrier();

    for (var j: u32 = lane; j < n; j = j + THREADS) {
        probs[row + j] = probs[row + j] / sum;
    }
    workgroupBarrier();

    // The value accumulation splits the head dimension instead: each
    // thread owns one feature and walks the window for it.
    for (var d: u32 = lane; d < p.head_dim; d = d + THREADS) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < n; j = j + 1u) {
            let base_v = (lo + j) * kvd + kvh * p.head_dim;
            acc = acc + probs[row + j] * v[base_v + d];
        }
        if (live) {
            out[base_q + d] = acc;
        }
    }
}
