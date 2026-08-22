// dw[o,i] += sum_t dy[t,o]*x[t,i]. Mirrors ops::linear_bwd's dw.
//
// Accumulates rather than overwrites, so every sequence of a batch adds
// into one gradient buffer the caller zeroes once per step - matching
// llm_core::model::backward_into.
//
// Same 64x64 tile, 4x4 per thread structure as linear.wgsl. Here the
// contraction runs over the rows (t), and both operands are stored
// [t][*], so both tiles are loaded transposed into workgroup memory.
//
// Dispatch: gid.x covers in_dim in blocks of 64, gid.y covers out_dim in
// blocks of 64.
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
const PER: u32 = 4u;

var<workgroup> tile_dy: array<f32, 1024>; // [64 o][16 t]
var<workgroup> tile_x: array<f32, 1024>;  // [64 i][16 t]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let out_base = wid.y * TILE;
    let in_base = wid.x * TILE;
    let tid = lid.y * 16u + lid.x;

    var acc: array<f32, 16>;
    for (var n: u32 = 0u; n < 16u; n = n + 1u) {
        acc[n] = 0.0;
    }

    let slabs = (p.rows + DEPTH - 1u) / DEPTH;
    for (var s: u32 = 0u; s < slabs; s = s + 1u) {
        let t_base = s * DEPTH;
        for (var f: u32 = 0u; f < 4u; f = f + 1u) {
            let index = tid + f * 256u;
            let t = index / TILE;   // 0..15
            let c = index % TILE;   // 0..63

            var dv: f32 = 0.0;
            if (t_base + t < p.rows && out_base + c < p.out_dim) {
                dv = dy[(t_base + t) * p.out_dim + out_base + c];
            }
            tile_dy[c * DEPTH + t] = dv;

            var xv: f32 = 0.0;
            if (t_base + t < p.rows && in_base + c < p.in_dim) {
                xv = x[(t_base + t) * p.in_dim + in_base + c];
            }
            tile_x[c * DEPTH + t] = xv;
        }
        workgroupBarrier();

        for (var t: u32 = 0u; t < DEPTH; t = t + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var n: u32 = 0u; n < PER; n = n + 1u) {
                a[n] = tile_dy[(lid.y * PER + n) * DEPTH + t];
                b[n] = tile_x[(lid.x * PER + n) * DEPTH + t];
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
