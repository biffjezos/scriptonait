// Attention for a single query row against a key/value cache — the
// decode step, and the only genuinely new kernel this backend needs.
//
// Mirrors ops::attention_step in llm-core, including grouped-query
// attention: query head h reads key/value head h / (heads / kv_heads).
//
// One thread per query head. The softmax is computed in a single
// streaming pass (the "online softmax" of FlashAttention): a running
// maximum, a running denominator, and a running weighted sum of V, each
// rescaled when a larger score turns up. That means no scratch array of
// scores, so nothing here caps how many cached positions a generation
// can attend over — the previous kernel's fixed 256-entry window array
// was the reason this backend refused long contexts outright.
//
// The cache is a ring buffer: the caller writes position p into slot
// p % capacity and passes how many slots are live. Order doesn't matter
// to the result, because softmax is order-independent and RoPE has
// already written each key's position into the key itself.
struct Params {
    cached_len: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [heads * head_dim]
@group(0) @binding(2) var<storage, read> k_cache: array<f32>;  // [capacity, kv_heads * head_dim]
@group(0) @binding(3) var<storage, read> v_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [heads * head_dim]

// Generous cap on head_dim; a model past this can't use this backend and
// the Rust side checks before ever dispatching (see `supports`).
const MAX_HEAD_DIM: u32 = 256u;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= p.heads) {
        return;
    }
    let group = p.heads / p.kv_heads;
    let kvh = h / group;
    let kv_dim = p.kv_heads * p.head_dim;
    let q_base = h * p.head_dim;
    let scale = 1.0 / sqrt(f32(p.head_dim));

    var acc: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }
    var running_max: f32 = -3.0e38;
    var denom: f32 = 0.0;

    for (var s: u32 = 0u; s < p.cached_len; s = s + 1u) {
        let base_s = s * kv_dim + kvh * p.head_dim;
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            score = score + q[q_base + d] * k_cache[base_s + d];
        }
        score = score * scale;

        // Rescale everything accumulated so far if this score is the new
        // maximum, then add this position's contribution.
        let new_max = max(running_max, score);
        let rescale = exp(running_max - new_max);
        let weight = exp(score - new_max);
        denom = denom * rescale + weight;
        for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
            acc[d] = acc[d] * rescale + weight * v_cache[base_s + d];
        }
        running_max = new_max;
    }

    for (var d: u32 = 0u; d < p.head_dim; d = d + 1u) {
        out[q_base + d] = acc[d] / denom;
    }
}
