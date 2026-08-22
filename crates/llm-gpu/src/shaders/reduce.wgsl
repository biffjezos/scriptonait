// Sums a buffer (optionally of squares) into one slot of a small stats
// buffer, accumulating. This is how a training step gets its loss and
// its global gradient norm without reading whole tensors back: every
// tensor reduces into its own slot, and the host reads the one small
// buffer once per step.
//
// A single workgroup per dispatch, 256 threads striding over the tensor:
// a grid-wide reduction would need either float atomics (which WGSL does
// not have) or a second pass. The largest tensor is the embedding table,
// which at 256 threads is a few thousand iterations each - small beside
// the matmuls of the step it belongs to.
struct Params {
    len: u32,
    slot: u32,
    square: u32,
    _p0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc: f32 = 0.0;
    for (var i: u32 = lid.x; i < p.len; i = i + THREADS) {
        let value = src[i];
        if (p.square == 1u) {
            acc = acc + value * value;
        } else {
            acc = acc + value;
        }
    }
    partial[lid.x] = acc;
    workgroupBarrier();

    for (var stride: u32 = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        stats[p.slot] = stats[p.slot] + partial[0];
    }
}
