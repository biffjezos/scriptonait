// buf[i] = 0. Resets the gradient accumulators between steps.
//
// Grid-stride: each thread walks the buffer in strides of the whole grid
// rather than owning one element. A model's embedding table runs to
// millions of floats, and one-thread-per-element would ask for more
// workgroups in a dimension than WebGPU's limit allows (65535 on most
// adapters) - a validation failure, not a slowdown.
struct Params {
    len: u32,
    stride: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var i: u32 = gid.x; i < p.len; i = i + p.stride) {
        buf[i] = 0.0;
    }
}
