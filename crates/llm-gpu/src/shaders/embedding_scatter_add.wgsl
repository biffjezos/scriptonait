// table_grad[v,:] += sum over the positions where token v appears of
// d_rows[t,:] — the input-embedding half of the tied embed/head gradient.
//
// Only the tokens this sequence actually contains are visited, through a
// compressed index the host builds per sequence: `row_ids[g]` is the g-th
// distinct token id, and `positions[offsets[g] .. offsets[g+1]]` are the
// sequence positions holding it. That makes this O(t_len * hidden)
// instead of O(vocab * hidden * t_len) — with a BPE-sized vocabulary the
// difference is thousands of times the work, and the slow version was
// the single most expensive dispatch in a training step, big enough on a
// small GPU to trip the driver's watchdog on its own.
//
// Each (group, feature) pair is written by exactly one thread, so no
// atomics are needed — which matters, because WGSL has no float atomics.
struct Params {
    groups: u32,
    hidden: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_rows: array<f32>;
@group(0) @binding(2) var<storage, read> row_ids: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> positions: array<u32>;
@group(0) @binding(5) var<storage, read_write> table_grad: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let d = gid.y;
    if (g >= p.groups || d >= p.hidden) {
        return;
    }
    var acc: f32 = 0.0;
    let start = offsets[g];
    let end = offsets[g + 1u];
    for (var i: u32 = start; i < end; i = i + 1u) {
        acc = acc + d_rows[positions[i] * p.hidden + d];
    }
    let row = row_ids[g];
    table_grad[row * p.hidden + d] = table_grad[row * p.hidden + d] + acc;
}
