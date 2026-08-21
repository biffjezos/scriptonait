// Third attention-backward pass: dK[s] = sum_t d_score[t,s]*Q[t] and
// dV[s] = sum_t probs[t,s]*d_out[t]. The CPU reference (ops::attention_bwd)
// computes these as a scatter from each t into overlapping s's; here the
// loop is inverted to a gather instead - thread per s, looping over
// exactly the t's whose window includes s (t in [s, min(T-1, s+window-1)],
// the causal+window mask's mirror image) - so each dK[s]/dV[s] is written
// by exactly one thread and no atomics are needed.
struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_score: array<f32>;
@group(0) @binding(2) var<storage, read> probs: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> d_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> dk: array<f32>;
@group(0) @binding(6) var<storage, read_write> dv: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let h = gid.y;
    if (s >= p.t_len || h >= p.heads) {
        return;
    }
    let hd = p.heads * p.head_dim;
    let base_s = s * hd + h * p.head_dim;

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        dk[base_s + d] = 0.0;
        dv[base_s + d] = 0.0;
    }

    var hi: u32 = s + p.window - 1u;
    if (hi >= p.t_len) {
        hi = p.t_len - 1u;
    }
    for (var t: u32 = s; t <= hi; t = t + 1u) {
        let row_base = (h * p.t_len + t) * p.t_len;
        let ds = d_score[row_base + s];
        let pr = probs[row_base + s];
        let base_t = t * hd + h * p.head_dim;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            dk[base_s + d] = dk[base_s + d] + ds * q[base_t + d];
            dv[base_s + d] = dv[base_s + d] + pr * d_out[base_t + d];
        }
    }
}
