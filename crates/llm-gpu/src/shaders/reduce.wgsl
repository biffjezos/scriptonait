// Sums a buffer (optionally of squares) into one slot of a small stats
// buffer, accumulating. This is how a training step gets its loss and
// its global gradient norm without reading whole tensors back: every
// tensor reduces into its own slot, and the host reads the one small
// buffer once per step.
//
// A single workgroup per dispatch: a grid-wide reduction would need
// either atomics or a second pass, and the tensors here are small enough
// that one workgroup striding over them is not the bottleneck.
struct Params {
    len: u32,
    slot: u32,
    square: u32,
    _p0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc: f32 = 0.0;
    for (var i: u32 = lid.x; i < p.len; i = i + 64u) {
        let value = src[i];
        if (p.square == 1u) {
            acc = acc + value * value;
        } else {
            acc = acc + value;
        }
    }
    partial[lid.x] = acc;
    workgroupBarrier();

    for (var stride: u32 = 32u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        stats[p.slot] = stats[p.slot] + partial[0];
    }
}
