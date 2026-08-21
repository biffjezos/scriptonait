// buf[i] *= scale. Used to average accumulated per-sequence gradients
// over the batch (scale = 1/batch_size) before the Adam step.
struct Params {
    len: u32,
    scale: f32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) {
        return;
    }
    buf[i] = buf[i] * p.scale;
}
