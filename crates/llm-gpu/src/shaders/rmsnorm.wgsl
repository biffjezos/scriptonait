// y[t,:] = (x[t,:] / rms(x[t,:])) * gain. Mirrors ops::rmsnorm_fwd,
// including the inv_rms cache (needed by rmsnorm_bwd_*).
struct Params {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gain: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> inv_rms_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.rows) {
        return;
    }
    var ms: f32 = 0.0;
    for (var i: u32 = 0u; i < p.dim; i = i + 1u) {
        let v = x[t * p.dim + i];
        ms = ms + v * v;
    }
    ms = ms / f32(p.dim);
    let inv = 1.0 / sqrt(ms + p.eps);
    inv_rms_out[t] = inv;
    for (var i: u32 = 0u; i < p.dim; i = i + 1u) {
        y[t * p.dim + i] = x[t * p.dim + i] * inv * gain[i];
    }
}
