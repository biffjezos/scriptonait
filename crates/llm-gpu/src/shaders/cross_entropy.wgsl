// Mean cross-entropy loss per row, plus its gradient wrt the logits
// already divided by t_len - mirrors llm_core::ops::cross_entropy. One
// thread per row, looping over the vocabulary, which is a parameter here
// rather than a constant: the vocabulary is whatever the trained BPE
// tokenizer ended up with.
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.t_len) {
        return;
    }
    let base = t * p.vocab;
    var maxv: f32 = -3.0e38;
    for (var v: u32 = 0u; v < p.vocab; v = v + 1u) {
        if (logits[base + v] > maxv) {
            maxv = logits[base + v];
        }
    }
    var sum: f32 = 0.0;
    for (var v: u32 = 0u; v < p.vocab; v = v + 1u) {
        sum = sum + exp(logits[base + v] - maxv);
    }
    // `target` is reserved in WGSL; naming a local that fails compilation
    // for the whole module.
    let tgt = targets[t];
    loss_out[t] = log(sum) + maxv - logits[base + tgt];

    let inv_t = 1.0 / f32(p.t_len);
    for (var v: u32 = 0u; v < p.vocab; v = v + 1u) {
        let prob = exp(logits[base + v] - maxv) / sum;
        var indicator: f32 = 0.0;
        if (v == tgt) {
            indicator = 1.0;
        }
        d_logits[base + v] = (prob - indicator) * inv_t;
    }
}
