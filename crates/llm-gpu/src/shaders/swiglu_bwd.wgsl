// dgate[i] = d_act[i]*up[i]*silu_grad(gate[i]); dup[i] = d_act[i]*silu(gate[i]).
// Mirrors ops::swiglu_bwd.
struct Params {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d_act: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgate: array<f32>;
@group(0) @binding(5) var<storage, read_write> dup: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    return s + x * s * (1.0 - s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) {
        return;
    }
    dgate[i] = d_act[i] * up[i] * silu_grad(gate[i]);
    dup[i] = d_act[i] * silu(gate[i]);
}
