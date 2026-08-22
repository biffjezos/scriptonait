// First of three attention-backward kernels: the softmax backward for
// each (row, head), leaving d_score in the same banded layout as `probs`.
// Mirrors the middle of llm_core::ops::attention_bwd.
//
// One workgroup of 64 threads per (row, head), splitting the window
// between them - see attention_fwd.wgsl for why one thread per row is
// nowhere near enough parallelism. d_score doubles as the scratch for
// d_probs: pass one writes d_probs there and reduces the row, pass two
// turns it into d_score in place.
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
    let live = t < p.t_len && h < p.heads;

    let hd = p.heads * p.head_dim;
    let kvd = p.kv_heads * p.head_dim;
    let group = p.heads / p.kv_heads;
    let kvh = select(0u, h / group, live);
    let base_t = select(0u, t * hd + h * p.head_dim, live);
    let row = select(0u, (h * p.t_len + t) * p.band, live);
    var lo: u32 = 0u;
    if (live && t + 1u > p.band) {
        lo = t + 1u - p.band;
    }
    let n = select(0u, t - lo + 1u, live);

    var lane_sum: f32 = 0.0;
    for (var j: u32 = lane; j < n; j = j + THREADS) {
        let base_v = (lo + j) * kvd + kvh * p.head_dim;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            acc = acc + d_out[base_t + d] * v[base_v + d];
        }
        d_score[row + j] = acc;
        lane_sum = lane_sum + probs[row + j] * acc;
    }
    for (var j: u32 = n + lane; j < p.band; j = j + THREADS) {
        d_score[row + j] = 0.0;
    }
    partial[lane] = lane_sum;
    workgroupBarrier();
    for (var stride: u32 = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
    }
    let s_sum = partial[0];
    workgroupBarrier();

    let scale = 1.0 / sqrt(f32(p.head_dim));
    for (var j: u32 = lane; j < n; j = j + THREADS) {
        d_score[row + j] = probs[row + j] * (d_score[row + j] - s_sum) * scale;
    }
}
