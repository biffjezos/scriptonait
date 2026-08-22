// Sums `count` partials into one stats slot, accumulating: the second
// half of the two-stage reduction reduce.wgsl starts.
struct Params {
    base: u32,
    count: u32,
    slot: u32,
    _p0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

const THREADS: u32 = 64u;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc: f32 = 0.0;
    for (var i: u32 = lid.x; i < p.count; i = i + THREADS) {
        acc = acc + partials[p.base + i];
    }
    partial[lid.x] = acc;
    workgroupBarrier();
    for (var s: u32 = THREADS / 2u; s > 0u; s = s / 2u) {
        if (lid.x < s) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        stats[p.slot] = stats[p.slot] + partial[0];
    }
}
