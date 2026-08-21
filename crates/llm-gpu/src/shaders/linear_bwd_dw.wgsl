// dw[o,i] = sum_t dy[t,o]*x[t,i]. Mirrors ops::linear_bwd's dw. Thread
// per (o,i), gathering over all T rows - not a scatter, so no atomics
// needed even though this accumulates over the batch dimension the CPU
// reference loops over explicitly.
struct Params {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    let i = gid.y;
    if (o >= p.out_dim || i >= p.in_dim) {
        return;
    }
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < p.rows; t = t + 1u) {
        acc = acc + dy[t * p.out_dim + o] * x[t * p.in_dim + i];
    }
    dw[o * p.in_dim + i] = acc;
}
