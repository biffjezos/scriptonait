// y[rows,out_dim] = x[rows,in_dim] @ w[out_dim,in_dim]^T  (no bias)
// Direct GPU translation of llm-core's ops::linear_fwd — see that
// function for the reference this must match.
//
// Tiled through workgroup shared memory. The obvious one-thread-per-output
// version (which this replaces) issued two global loads per multiply and
// read at least one of its operands with a stride of a whole row, so
// neighbouring threads never shared a cache line: it ran at a small
// fraction of the device's memory bandwidth and made a training step tens
// of times slower than the arithmetic warrants. Here each 16x16 block
// loads one tile of x and one of w cooperatively — every load contiguous
// across lanes — then reuses each loaded value 16 times out of shared
// memory.
//
// Note the dispatch convention: gid.x indexes out_dim, gid.y indexes rows
// (so the fastest-varying thread index runs along contiguous memory).
// model.rs's dispatch_linear must match.
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

const TILE: u32 = 16u;

var<workgroup> tile_x: array<f32, 256>; // [row_local][k_local]
var<workgroup> tile_w: array<f32, 256>; // [out_local][k_local]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (p.in_dim + TILE - 1u) / TILE;
    for (var k: u32 = 0u; k < num_tiles; k = k + 1u) {
        let kk = k * TILE + lid.x;

        // Both loads are contiguous across lid.x, the fastest-varying lane.
        var xv: f32 = 0.0;
        let x_row = wid.y * TILE + lid.y;
        if (x_row < p.rows && kk < p.in_dim) {
            xv = x[x_row * p.in_dim + kk];
        }
        tile_x[lid.y * TILE + lid.x] = xv;

        var wv: f32 = 0.0;
        let w_row = wid.x * TILE + lid.y;
        if (w_row < p.out_dim && kk < p.in_dim) {
            wv = w[w_row * p.in_dim + kk];
        }
        tile_w[lid.y * TILE + lid.x] = wv;

        // Barriers sit in uniform control flow: this kernel never returns
        // early, it only guards its loads and its final store.
        workgroupBarrier();
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            acc = acc + tile_x[lid.y * TILE + j] * tile_w[lid.x * TILE + j];
        }
        workgroupBarrier();
    }

    if (row < p.rows && col < p.out_dim) {
        y[row * p.out_dim + col] = acc;
    }
}
