// y[rows,out_dim] = x[rows,in_dim] @ w[out_dim,in_dim]^T  (no bias)
// Direct GPU translation of llm-core's ops::linear_fwd — see that
// function for the reference this must match.
struct Params {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let o = gid.y;
    if (t >= p.rows || o >= p.out_dim) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < p.in_dim; i = i + 1u) {
        acc = acc + x[t * p.in_dim + i] * w[o * p.in_dim + i];
    }
    y[t * p.out_dim + o] = acc;
}
