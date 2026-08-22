// y[rows,out_dim] = x[rows,in_dim] @ w[out_dim,in_dim]^T  (no bias)
// GPU translation of llm_core's ops::linear_fwd.
//
// 64x64 output tile per workgroup, 4x4 outputs per thread, accumulated in
// registers over 16-deep k-slabs held in workgroup memory.
//
// The previous version had each thread produce one output. That reloads
// every operand from shared memory for every single multiply-add: one
// FMA per two loads, which is memory-bound at a small fraction of the
// device's arithmetic. Holding a 4x4 block in registers reuses each
// loaded value four times, an 8x cut in shared-memory traffic for the
// same arithmetic - and the matmuls are where essentially all of a
// training step's time goes.
//
// Dispatch convention: gid.x covers out_dim in blocks of 64, gid.y covers
// rows in blocks of 64. model.rs's dispatch_linear must match.
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

const TILE: u32 = 64u;   // rows and columns per workgroup
const DEPTH: u32 = 16u;  // k elements per slab
const PER: u32 = 4u;     // outputs per thread, each dimension

var<workgroup> tile_x: array<f32, 1024>; // [64][16], row-major
var<workgroup> tile_w: array<f32, 1024>; // [64][16], row-major

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row_base = wid.y * TILE;
    let col_base = wid.x * TILE;
    let tid = lid.y * 16u + lid.x;

    var acc: array<f32, 16>; // 4x4 block, [i][j]
    for (var n: u32 = 0u; n < 16u; n = n + 1u) {
        acc[n] = 0.0;
    }

    let slabs = (p.in_dim + DEPTH - 1u) / DEPTH;
    for (var s: u32 = 0u; s < slabs; s = s + 1u) {
        let k_base = s * DEPTH;
        // 256 threads cooperatively load two 64x16 tiles, four elements
        // each. Consecutive threads read consecutive k, so every load is
        // coalesced.
        for (var f: u32 = 0u; f < 4u; f = f + 1u) {
            let index = tid + f * 256u;
            let i = index / DEPTH;
            let k = index % DEPTH;

            var xv: f32 = 0.0;
            let xr = row_base + i;
            if (xr < p.rows && k_base + k < p.in_dim) {
                xv = x[xr * p.in_dim + k_base + k];
            }
            tile_x[index] = xv;

            var wv: f32 = 0.0;
            let wr = col_base + i;
            if (wr < p.out_dim && k_base + k < p.in_dim) {
                wv = w[wr * p.in_dim + k_base + k];
            }
            tile_w[index] = wv;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < DEPTH; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < PER; i = i + 1u) {
                a[i] = tile_x[(lid.y * PER + i) * DEPTH + k];
                b[i] = tile_w[(lid.x * PER + i) * DEPTH + k];
            }
            for (var i: u32 = 0u; i < PER; i = i + 1u) {
                for (var j: u32 = 0u; j < PER; j = j + 1u) {
                    acc[i * PER + j] = acc[i * PER + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < PER; i = i + 1u) {
        let row = row_base + lid.y * PER + i;
        if (row >= p.rows) {
            continue;
        }
        for (var j: u32 = 0u; j < PER; j = j + 1u) {
            let col = col_base + lid.x * PER + j;
            if (col < p.out_dim) {
                y[row * p.out_dim + col] = acc[i * PER + j];
            }
        }
    }
}
