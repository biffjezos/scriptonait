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
/// Keys cached per pass over the window. Every thread needs the same
/// `d_score` row, and reading it from global once per feature meant 64
/// threads fetching the same value 64 times. Chunking keeps the cache
/// small enough for any window length.
const CHUNK: u32 = 256u;

var<workgroup> scores: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wid.x;
    let h = wid.y;
    // No early return: the barriers below have to stay in uniform control
    // flow. The dispatch is exactly (t_len, heads) so this never fires,
    // but an out-of-range workgroup would run with an empty window and
    // write nothing rather than skip a barrier its neighbours reach.
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

    // One thread per feature, walking the window - but the window's
    // gradients are shared by every thread, so they come through
    // workgroup memory a chunk at a time.
    for (var d: u32 = lid.x; d < p.head_dim; d = d + THREADS) {
        if (live) {
            dq[base_t + d] = 0.0;
        }
    }
    var base: u32 = 0u;
    loop {
        if (base >= n) {
            break;
        }
        let count = min(CHUNK, n - base);
        for (var i: u32 = lid.x; i < count; i = i + THREADS) {
            scores[i] = d_score[row + base + i];
        }
        workgroupBarrier();
        for (var d: u32 = lid.x; d < p.head_dim; d = d + THREADS) {
            var acc: f32 = 0.0;
            for (var j: u32 = 0u; j < count; j = j + 1u) {
                let base_k = (lo + base + j) * kvd + kvh * p.head_dim;
                acc = acc + scores[j] * k[base_k + d];
            }
            if (live) {
                dq[base_t + d] = dq[base_t + d] + acc;
            }
        }
        workgroupBarrier();
        base = base + CHUNK;
    }
}
