// Partial sums (optionally of squares) of a buffer, one float per
// workgroup, for the loss and the global gradient norm.
//
// One workgroup striding over a whole tensor is a single 64- or
// 256-thread block walking millions of floats - serial time proportional
// to the largest tensor, paid once per tensor per step. Many workgroups
// each reduce their own slice into `partials[base + workgroup]`, and
// reduce_finish.wgsl sums those into the stats slot. Two dispatches
// instead of one, both parallel, and no float atomics (WGSL has none).
struct Params {
    len: u32,
    base: u32,     // first partials slot this dispatch may write
    square: u32,
    _p0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * THREADS;
    var acc: f32 = 0.0;
    for (var i: u32 = wid.x * THREADS + lid.x; i < p.len; i = i + stride) {
        let value = src[i];
        if (p.square == 1u) {
            acc = acc + value * value;
        } else {
            acc = acc + value;
        }
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
        partials[p.base + wid.x] = partial[0];
    }
}
