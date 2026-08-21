// dgain[i] = sum_t dy[t,i]*x[t,i]*inv_rms[t]. Thread per feature - a
// gather over all T rows, not a scatter, so no atomics needed even
// though the CPU reference (ops::rmsnorm_bwd) computes this as a
// row-at-a-time accumulation into a shared buffer.
struct Params {
    rows: u32,
    dim: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> inv_rms: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgain: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.dim) {
        return;
    }
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < p.rows; t = t + 1u) {
        acc = acc + dy[t * p.dim + i] * x[t * p.dim + i] * inv_rms[t];
    }
    dgain[i] = acc;
}
