// Grouped-query, causal, sliding-window attention over a whole sequence,
// with the banded probability cache the backward pass needs. Mirrors
// llm_core::ops::attention_fwd exactly, including its storage layout:
// `probs` is [heads, t_len, band] and entry j of row t is the key at
// absolute position `band_lo(t) + j`, not at position j.
//
// One thread per (row, head). The score row is built directly in `probs`
// rather than in a private array: a private array big enough for the
// window costs a kilobyte of per-thread storage, which on a small
// integrated GPU is the difference between running and spilling.
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
    let base_q = t * hd + h * p.head_dim;
    let row = (h * p.t_len + t) * p.band;

    var lo: u32 = 0u;
    if (t + 1u > p.band) {
        lo = t + 1u - p.band;
    }
    let n = t - lo + 1u;
    let scale = 1.0 / sqrt(f32(p.head_dim));

    var maxv: f32 = -3.0e38;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let base_k = (lo + j) * kvd + kvh * p.head_dim;
        var acc: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            acc = acc + q[base_q + d] * k[base_k + d];
        }
        let s = acc * scale;
        probs[row + j] = s;
        if (s > maxv) {
            maxv = s;
        }
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let e = exp(probs[row + j] - maxv);
        probs[row + j] = e;
        sum = sum + e;
    }
    // Entries past this row's window are never read by the backward pass,
    // but a stale value from an earlier sequence in the batch would be a
    // silent wrong number if that ever changed. Zero them.
    for (var j: u32 = n; j < p.band; j = j + 1u) {
        probs[row + j] = 0.0;
    }

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        out[base_q + d] = 0.0;
    }
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let pr = probs[row + j] / sum;
        probs[row + j] = pr;
        let base_v = (lo + j) * kvd + kvh * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            out[base_q + d] = out[base_q + d] + pr * v[base_v + d];
        }
    }
}
