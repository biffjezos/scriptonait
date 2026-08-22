// dx[t,i] = sum_o dy[t,o]*w[o,i]. Mirrors ops::linear_bwd's dx.
//
// Tiled through workgroup shared memory for the same reason as
// linear.wgsl — see that file's header. gid.x indexes in_dim, gid.y
// indexes rows; model.rs's dispatch_linear_bwd must match.
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

const TILE: u32 = 16u;

var<workgroup> tile_dy: array<f32, 256>; // [row_local][o_local]
var<workgroup> tile_w: array<f32, 256>;  // [o_local][in_local]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wid.y * TILE + lid.y;
    let i = wid.x * TILE + lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (p.out_dim + TILE - 1u) / TILE;
    for (var k: u32 = 0u; k < num_tiles; k = k + 1u) {
        var dyv: f32 = 0.0;
        let o_for_dy = k * TILE + lid.x;
        if (t < p.rows && o_for_dy < p.out_dim) {
            dyv = dy[t * p.out_dim + o_for_dy];
        }
        tile_dy[lid.y * TILE + lid.x] = dyv;

        var wv: f32 = 0.0;
        let o_for_w = k * TILE + lid.y;
        if (o_for_w < p.out_dim && i < p.in_dim) {
            wv = w[o_for_w * p.in_dim + i];
        }
        tile_w[lid.y * TILE + lid.x] = wv;

        workgroupBarrier();
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            acc = acc + tile_dy[lid.y * TILE + j] * tile_w[j * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (t < p.rows && i < p.in_dim) {
        dx[t * p.in_dim + i] = acc;
    }
}
