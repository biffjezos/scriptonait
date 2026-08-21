// Mirrors ops::rmsnorm_bwd's dx computation exactly. Thread per row -
// entirely local, no cross-thread accumulation, so no atomics needed.
struct Params {
    rows: u32,
    dim: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> gain: array<f32>;
@group(0) @binding(4) var<storage, read> inv_rms: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.rows) {
        return;
    }
    let n = inv_rms[t];
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < p.dim; i = i + 1u) {
        s = s + dy[t * p.dim + i] * gain[i] * x[t * p.dim + i];
    }
    let n3_over_d = n * n * n / f32(p.dim);
    for (var i: u32 = 0u; i < p.dim; i = i + 1u) {
        dx[t * p.dim + i] = n * gain[i] * dy[t * p.dim + i] - n3_over_d * x[t * p.dim + i] * s;
    }
}
