// f16 variant of linear_bwd_dw.wgsl.
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
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

const TILE: u32 = 64u;
const DEPTH: u32 = 16u;
const PER: u32 = 8u;

var<workgroup> tile_dy: array<f16, 1024>; // [64 o][16 t]
var<workgroup> tile_x: array<f16, 1024>;  // [64 i][16 t]

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_base = wid.y * TILE;
    let in_base = wid.x * TILE;
    let tid = lid.y * 8u + lid.x;

    var acc: array<f32, 64>;
    for (var n: u32 = 0u; n < 64u; n = n + 1u) {
        acc[n] = 0.0;
    }
    var slab: array<f16, 64>;

    let slabs = (p.rows + DEPTH - 1u) / DEPTH;
    for (var s: u32 = 0u; s < slabs; s = s + 1u) {
        let t_base = s * DEPTH;
        for (var f: u32 = 0u; f < 16u; f = f + 1u) {
            let index = tid + f * 64u;
            let t = index / TILE;
            let c = index % TILE;

            var dv: f32 = 0.0;
            if (t_base + t < p.rows && out_base + c < p.out_dim) {
                dv = dy[(t_base + t) * p.out_dim + out_base + c];
            }
            tile_dy[c * DEPTH + t] = f16(dv);

            var xv: f32 = 0.0;
            if (t_base + t < p.rows && in_base + c < p.in_dim) {
                xv = x[(t_base + t) * p.in_dim + in_base + c];
            }
            tile_x[c * DEPTH + t] = f16(xv);
        }
        workgroupBarrier();
        for (var n: u32 = 0u; n < 64u; n = n + 1u) {
            slab[n] = f16(0.0);
        }

        for (var t: u32 = 0u; t < DEPTH; t = t + 1u) {
            var a: array<f16, 8>;
            var b: array<f16, 8>;
            for (var n: u32 = 0u; n < PER; n = n + 1u) {
                a[n] = tile_dy[(lid.y * PER + n) * DEPTH + t];
                b[n] = tile_x[(lid.x * PER + n) * DEPTH + t];
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
        let o = out_base + lid.y * PER + i;
        if (o >= p.out_dim) {
            continue;
        }
        for (var j: u32 = 0u; j < PER; j = j + 1u) {
            let col = in_base + lid.x * PER + j;
            if (col < p.in_dim) {
                dw[o * p.in_dim + col] = dw[o * p.in_dim + col] + acc[i * PER + j];
            }
        }
    }
}
