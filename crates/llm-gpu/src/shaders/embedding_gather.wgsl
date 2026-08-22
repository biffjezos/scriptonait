// out[t,:] = table[ids[t],:]. Used for both the shared input embedding and
// each layer's per-layer embedding (PLE) lookup — a pure vector lookup,
// no matmul, so it's cheap regardless of table size.
struct Params {
    t_len: u32,
    hidden: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let d = gid.y;
    if (t >= p.t_len || d >= p.hidden) {
        return;
    }
    let id = ids[t];
    out[t * p.hidden + d] = table[id * p.hidden + d];
}
