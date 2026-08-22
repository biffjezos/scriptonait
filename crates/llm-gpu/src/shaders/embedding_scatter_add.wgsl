// table_grad[v,:] += sum_{t: ids[t]==v} d_rows[t,:]. Used for both the
// shared input embedding and every layer's PLE table gradient (same
// mechanism as the forward gather, run in reverse). ops::model's CPU
// reference does this as a scatter (loop over t, add into table_grad at
// ids[t]); here it's inverted to a gather - thread per (vocab row,
// feature), looping over all T token positions and checking for a match.
// The vocabulary is small next to the work of a training step, so being
// O(vocab * t_len) instead of O(t_len) is cheap - and, the actual point,
// it needs no atomics, unlike a literal per-token scatter would (a
// sequence longer than the vocabulary is guaranteed to repeat tokens).
struct Params {
    t_len: u32,
    hidden: u32,
    vocab: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_rows: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> table_grad: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    let d = gid.y;
    if (v >= p.vocab || d >= p.hidden) {
        return;
    }
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < p.t_len; t = t + 1u) {
        if (ids[t] == v) {
            acc = acc + d_rows[t * p.hidden + d];
        }
    }
    table_grad[v * p.hidden + d] = table_grad[v * p.hidden + d] + acc;
}
