// dx[t,i] = sum_o dy[t,o]*w[o,i]. Mirrors ops::linear_bwd's dx.
//
// Same 64x64 tile, 8x8 per thread structure as linear.wgsl - see that
// file for why the block size is the point. The difference here: the
// contraction runs over out_dim, and w is stored [out_dim, in_dim], so
// its tile is loaded transposed into workgroup memory (still coalesced,
// because consecutive threads take consecutive `i`).
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

var<workgroup> tile_dy: array<f32, 1024>; // [64 rows][16 o]
var<workgroup> tile_w: array<f32, 1024>;  // [64 i][16 o], transposed on load

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
            tile_dy[index] = dv;
        }
        for (var f: u32 = 0u; f < 16u; f = f + 1u) {
            let index = tid + f * 64u;
            let o = index / TILE;
            let i = index % TILE;
            var wv: f32 = 0.0;
            if (o_base + o < p.out_dim && col_base + i < p.in_dim) {
                wv = w[(o_base + o) * p.in_dim + col_base + i];
            }
            tile_w[i * DEPTH + o] = wv;
        }
        workgroupBarrier();

        for (var o: u32 = 0u; o < DEPTH; o = o + 1u) {
            var a: array<f32, 8>;
            var b: array<f32, 8>;
            for (var n: u32 = 0u; n < PER; n = n + 1u) {
                a[n] = tile_dy[(lid.y * PER + n) * DEPTH + o];
                b[n] = tile_w[(lid.x * PER + n) * DEPTH + o];
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
            if (col < p.in_dim) {
                dx[row * p.in_dim + col] = acc[i * PER + j];
            }
        }
    }
}
