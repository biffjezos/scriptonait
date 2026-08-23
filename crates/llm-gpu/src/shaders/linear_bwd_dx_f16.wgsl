// f16 variant of linear_bwd_dx.wgsl.
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
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

const TILE: u32 = 64u;
const DEPTH: u32 = 16u;
const PER: u32 = 8u;

var<workgroup> tile_dy: array<f16, 1024>; // [64 rows][16 o]
var<workgroup> tile_w: array<f16, 1024>;  // [64 i][16 o], transposed on load

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

    let slabs = (p.out_dim + DEPTH - 1u) / DEPTH;
    for (var s: u32 = 0u; s < slabs; s = s + 1u) {
        let o_base = s * DEPTH;
        for (var f: u32 = 0u; f < 16u; f = f + 1u) {
            let index = tid + f * 64u;
            let a = index / DEPTH;
            let o = index % DEPTH;
            var dv: f32 = 0.0;
            let row = row_base + a;
            if (row < p.rows && o_base + o < p.out_dim) {
                dv = dy[row * p.out_dim + o_base + o];
            }
            tile_dy[index] = f16(dv);
        }
        for (var f: u32 = 0u; f < 16u; f = f + 1u) {
            let index = tid + f * 64u;
            let o = index / TILE;
            let i = index % TILE;
            var wv: f32 = 0.0;
            if (o_base + o < p.out_dim && col_base + i < p.in_dim) {
                wv = w[(o_base + o) * p.in_dim + col_base + i];
            }
            tile_w[i * DEPTH + o] = f16(wv);
        }
        workgroupBarrier();
        for (var n: u32 = 0u; n < 64u; n = n + 1u) {
            slab[n] = f16(0.0);
        }

        for (var o: u32 = 0u; o < DEPTH; o = o + 1u) {
            var a: array<f16, 8>;
            var b: array<f16, 8>;
            for (var n: u32 = 0u; n < PER; n = n + 1u) {
                a[n] = tile_dy[(lid.y * PER + n) * DEPTH + o];
                b[n] = tile_w[(lid.x * PER + n) * DEPTH + o];
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
            if (col < p.in_dim) {
                dx[row * p.in_dim + col] = acc[i * PER + j];
            }
        }
    }
}
