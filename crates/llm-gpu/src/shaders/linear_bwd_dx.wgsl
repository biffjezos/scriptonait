// dx[t,i] = sum_o dy[t,o]*w[o,i]. Mirrors ops::linear_bwd's dx.
struct Params {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let i = gid.y;
    if (t >= p.rows || i >= p.in_dim) {
        return;
    }
    var acc: f32 = 0.0;
    for (var o: u32 = 0u; o < p.out_dim; o = o + 1u) {
        acc = acc + dy[t * p.out_dim + o] * w[o * p.in_dim + i];
    }
    dx[t * p.in_dim + i] = acc;
}
