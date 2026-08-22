// dw[o,i] += sum_t dy[t,o]*x[t,i]. Mirrors ops::linear_bwd's dw. Thread
// per (o,i), gathering over all T rows - not a scatter, so no atomics
// needed even though this accumulates over the batch dimension the CPU
// reference loops over explicitly.
//
// Accumulates into dw rather than overwriting it, so a whole batch's
// sequences can add into one gradient buffer that the caller zeroes once
// per step — matching llm_core::model::backward_into. The caller MUST
// zero dw before the first sequence of a step.
//
// Tiled through workgroup shared memory — see linear.wgsl's header for
// why. gid.x indexes in_dim, gid.y indexes out_dim; model.rs's
// dispatch_linear_bwd must match.
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

const TILE: u32 = 16u;

var<workgroup> tile_dy: array<f32, 256>; // [t_local][out_local]
var<workgroup> tile_x: array<f32, 256>;  // [t_local][in_local]

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let o = wid.y * TILE + lid.y;
    let i = wid.x * TILE + lid.x;

    var acc: f32 = 0.0;
    let num_tiles = (p.rows + TILE - 1u) / TILE;
    for (var k: u32 = 0u; k < num_tiles; k = k + 1u) {
        let t = k * TILE + lid.y;

        var dyv: f32 = 0.0;
        let o_for_dy = wid.y * TILE + lid.x;
        if (t < p.rows && o_for_dy < p.out_dim) {
            dyv = dy[t * p.out_dim + o_for_dy];
        }
        tile_dy[lid.y * TILE + lid.x] = dyv;

        var xv: f32 = 0.0;
        if (t < p.rows && i < p.in_dim) {
            xv = x[t * p.in_dim + i];
        }
        tile_x[lid.y * TILE + lid.x] = xv;

        workgroupBarrier();
        for (var j: u32 = 0u; j < TILE; j = j + 1u) {
            acc = acc + tile_dy[j * TILE + lid.y] * tile_x[j * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (o < p.out_dim && i < p.in_dim) {
        dw[o * p.in_dim + i] = dw[o * p.in_dim + i] + acc;
    }
}
