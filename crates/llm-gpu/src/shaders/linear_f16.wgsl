// f16 variant of linear.wgsl.
//
// Identical tiling and identical indexing - only the numeric type of the
// workgroup tiles and the inner products changes. The products of one
// 16-deep slab accumulate in f16 and are added into an f32 accumulator
// at the end of each slab, which keeps the running sum in full precision
// while the inner loop gets f16's packed arithmetic and half the
// workgroup-memory traffic.
//
// Compiled only when the adapter reports `shader-f16`; llm-gpu falls
// back to the f32 kernel otherwise, so nothing here is required.
enable f16;

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

const TILE: u32 = 64u;   // output rows and columns per workgroup
const DEPTH: u32 = 16u;  // contraction elements per slab
const PER: u32 = 8u;     // outputs per thread, each dimension

var<workgroup> tile_x: array<f16, 1024>; // [64][16]
var<workgroup> tile_w: array<f16, 1024>; // [64][16]

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row_base = wid.y * TILE;
    let col_base = wid.x * TILE;
    let tid = lid.y * 8u + lid.x;

    var acc: array<f32, 64>;
    for (var n: u32 = 0u; n < 64u; n = n + 1u) {
        acc[n] = 0.0;
    }
    var slab: array<f16, 64>;

    let slabs = (p.in_dim + DEPTH - 1u) / DEPTH;
    for (var s: u32 = 0u; s < slabs; s = s + 1u) {
        let k_base = s * DEPTH;
        // 64 threads load two 64x16 tiles, sixteen elements each.
        // Consecutive threads read consecutive k, so loads coalesce.
        for (var f: u32 = 0u; f < 16u; f = f + 1u) {
            let index = tid + f * 64u;
            let i = index / DEPTH;
            let k = index % DEPTH;

            var xv: f32 = 0.0;
            let xr = row_base + i;
            if (xr < p.rows && k_base + k < p.in_dim) {
                xv = x[xr * p.in_dim + k_base + k];
            }
            tile_x[index] = f16(xv);

            var wv: f32 = 0.0;
            let wr = col_base + i;
            if (wr < p.out_dim && k_base + k < p.in_dim) {
                wv = w[wr * p.in_dim + k_base + k];
            }
            tile_w[index] = f16(wv);
        }
        workgroupBarrier();
        for (var n: u32 = 0u; n < 64u; n = n + 1u) {
            slab[n] = f16(0.0);
        }

        for (var k: u32 = 0u; k < DEPTH; k = k + 1u) {
            var a: array<f16, 8>;
            var b: array<f16, 8>;
            for (var i: u32 = 0u; i < PER; i = i + 1u) {
                a[i] = tile_x[(lid.y * PER + i) * DEPTH + k];
                b[i] = tile_w[(lid.x * PER + i) * DEPTH + k];
            }
            for (var i: u32 = 0u; i < PER; i = i + 1u) {
                for (var j: u32 = 0u; j < PER; j = j + 1u) {
                    slab[i * PER + j] = slab[i * PER + j] + a[i] * b[j];
                }
            }
        }
        for (var n: u32 = 0u; n < 64u; n = n + 1u) {
            acc[n] = acc[n] + f32(slab[n]);
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
