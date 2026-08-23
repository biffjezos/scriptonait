// Mean cross-entropy loss per row, plus its gradient wrt the logits
// already divided by t_len - mirrors llm_core::ops::cross_entropy.
//
// One workgroup of 256 threads per row, not one thread.
//
// One thread per row is `t_len` threads for the whole dispatch - 256 for
// a 256-token sequence - each walking the vocabulary three times (max,
// sum, then writing a gradient per entry). At a 4096-token vocabulary
// that was three seconds of a step, more than the forward pass, for
// arithmetic that is trivially parallel. Here the threads of a workgroup
// split the vocabulary and reduce in workgroup memory.
struct Params {
    t_len: u32,
    vocab: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss_out: array<f32>;

const THREADS: u32 = 256u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = wid.x;
    let lane = lid.x;
    // No early return: the barriers below must stay in uniform control
    // flow. An out-of-range row runs and writes nothing.
    let live = t < p.t_len;
    let base = select(0u, t * p.vocab, live);
    let count = select(0u, p.vocab, live);

    var lane_max: f32 = -3.0e38;
    for (var v: u32 = lane; v < count; v = v + THREADS) {
        lane_max = max(lane_max, logits[base + v]);
    }
    partial[lane] = lane_max;
    workgroupBarrier();
    for (var s: u32 = THREADS / 2u; s > 0u; s = s / 2u) {
        if (lane < s) {
            partial[lane] = max(partial[lane], partial[lane + s]);
        }
        workgroupBarrier();
    }
    let maxv = partial[0];
    workgroupBarrier();

    var lane_sum: f32 = 0.0;
    for (var v: u32 = lane; v < count; v = v + THREADS) {
        lane_sum = lane_sum + exp(logits[base + v] - maxv);
    }
    partial[lane] = lane_sum;
    workgroupBarrier();
    for (var s: u32 = THREADS / 2u; s > 0u; s = s / 2u) {
        if (lane < s) {
            partial[lane] = partial[lane] + partial[lane + s];
        }
        workgroupBarrier();
    }
    let sum = partial[0];
    workgroupBarrier();

    // `target` is reserved in WGSL; naming a local that fails compilation
    // for the whole module.
    let tgt = select(0u, targets[t], live);
    if (live && lane == 0u) {
        loss_out[t] = log(sum) + maxv - logits[base + tgt];
    }

    let inv_t = 1.0 / f32(p.t_len);
    for (var v: u32 = lane; v < count; v = v + THREADS) {
        let prob = exp(logits[base + v] - maxv) / sum;
        var indicator: f32 = 0.0;
        if (v == tgt) {
            indicator = 1.0;
        }
        d_logits[base + v] = (prob - indicator) * inv_t;
    }
}
