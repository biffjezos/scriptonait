// dk[s, kvh] = sum over the query heads sharing this kv head, and over
// every query row whose window contains s, of d_score * q[t, h];
// dv likewise with probs and d_out.
//
// The CPU reference scatters from each query row into the key rows its
// window covers. Here the loop is inverted into a gather - one workgroup
// per (key row, kv head), each of its 64 threads owning one feature and
// walking the query rows t in [s, min(T-1, s+band-1)] whose window
// contains s - so each output is written by one thread and no atomics are
// needed. Grouped-query attention is why the head loop is inside: several
// query heads feed one kv head, and their contributions sum.
//
// With one thread per (key row, kv head) this kernel ran 512 threads for
// a 256-token sequence with two kv heads, each walking a whole window
// times the group size. That left the device almost entirely idle and was
// the slowest thing in a training step.
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
@group(0) @binding(2) var<storage, read> probs: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> d_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> dk: array<f32>;
@group(0) @binding(6) var<storage, read_write> dv: array<f32>;

const THREADS: u32 = 64u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let s = wid.x;
    let kvh = wid.y;
    if (s >= p.t_len || kvh >= p.kv_heads) {
        return;
    }
    let hd = p.heads * p.head_dim;
    let kvd = p.kv_heads * p.head_dim;
    let group = p.heads / p.kv_heads;
    let base_s = s * kvd + kvh * p.head_dim;

    var hi: u32 = s + p.band - 1u;
    if (hi >= p.t_len) {
        hi = p.t_len - 1u;
    }

    for (var d: u32 = lid.x; d < p.head_dim; d = d + THREADS) {
        var acc_k: f32 = 0.0;
        var acc_v: f32 = 0.0;
        for (var g: u32 = 0u; g < group; g = g + 1u) {
            let h = kvh * group + g;
            for (var t: u32 = s; t <= hi; t = t + 1u) {
                var lo: u32 = 0u;
                if (t + 1u > p.band) {
                    lo = t + 1u - p.band;
                }
                let j = s - lo;
                let row = (h * p.t_len + t) * p.band;
                let base_t = t * hd + h * p.head_dim;
                acc_k = acc_k + d_score[row + j] * q[base_t + d];
                acc_v = acc_v + probs[row + j] * d_out[base_t + d];
            }
        }
        dk[base_s + d] = acc_k;
        dv[base_s + d] = acc_v;
    }
}
