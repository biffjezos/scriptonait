// Rotary position embedding, applied in place to Q or K, per (row, head).
// Mirrors ops::rope_apply_at (base 10000, pairs (2k, 2k+1)). Forward
// rotation only: nothing backpropagates through generation, so the
// inverse this used to carry had no caller.
struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    // Absolute position of the first row. Zero for a whole sequence;
    // the current length when decoding one token at a time, so a key
    // carries the rotation of the position it was written at.
    pos0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let h = gid.y;
    if (t >= p.t_len || h >= p.heads) {
        return;
    }
    let half = p.head_dim / 2u;
    let pos = f32(p.pos0 + t);
    let base = t * p.heads * p.head_dim + h * p.head_dim;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let freq = 1.0 / pow(10000.0, 2.0 * f32(k) / f32(p.head_dim));
        let angle = pos * freq;
        let c = cos(angle);
        let s = sin(angle);
        let a = x[base + 2u * k];
        let b = x[base + 2u * k + 1u];
        x[base + 2u * k] = a * c - b * s;
        x[base + 2u * k + 1u] = a * s + b * c;
    }
}
