// Mean cross-entropy loss (per-row, summed on the host) plus its gradient
// wrt the logits, already divided by t_len - mirrors ops::cross_entropy
// exactly. Thread per row, looping over the vocabulary (always 259 for
// this project's byte-level tokenizer - see llm-core::tokenizer::VOCAB_SIZE,
// which this constant must match).
const VOCAB: u32 = 259u;

struct Params {
    t_len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.t_len) {
        return;
    }
    let base = t * VOCAB;
    var maxv: f32 = -3.0e38;
    for (var v: u32 = 0u; v < VOCAB; v = v + 1u) {
        if (logits[base + v] > maxv) {
            maxv = logits[base + v];
        }
    }
    var sum: f32 = 0.0;
    for (var v: u32 = 0u; v < VOCAB; v = v + 1u) {
        sum = sum + exp(logits[base + v] - maxv);
    }
    // `target` is a reserved word in WGSL (reserved for future use, even
    // though not currently a keyword) - using it as an identifier fails
    // shader compilation entirely, silently invalidating this pipeline
    // and every dispatch that uses it. Call it `tgt` instead.
    let tgt = targets[t];
    loss_out[t] = log(sum) + maxv - logits[base + tgt];

    let inv_t = 1.0 / f32(p.t_len);
    for (var v: u32 = 0u; v < VOCAB; v = v + 1u) {
        let prob = exp(logits[base + v] - maxv) / sum;
        var target_ind: f32 = 0.0;
        if (v == tgt) {
            target_ind = 1.0;
        }
        d_logits[base + v] = (prob - target_ind) * inv_t;
    }
}
