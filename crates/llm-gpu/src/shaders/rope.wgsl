// Rotary position embedding, applied in place to Q or K, per (row, head).
// Mirrors ops::rope_apply_at (pairs (2k, 2k+1)), base given by `theta`
// rather than assumed — see ModelConfig::rope_theta's own doc comment.
//
// `inverse` rotates the other way, which is what the backward pass needs:
// RoPE is an orthogonal rotation, so the gradient wrt its input is the
// same rotation by the negative angle.
struct Params {
    t_len: u32,
    heads: u32,
    head_dim: u32,
    // Absolute position of the first row. Zero for a whole sequence;
    // the current length when decoding one token at a time, so a key
    // carries the rotation of the position it was written at.
    pos0: u32,
    // 1 rotates by the negative angle (the backward pass), 0 forward.
    inverse: u32,
    theta: f32,
    _p1: u32,
    _p2: u32,
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
    var sign: f32 = 1.0;
    if (p.inverse == 1u) {
        sign = -1.0;
    }
    let pos = f32(p.pos0 + t);
    let base = t * p.heads * p.head_dim + h * p.head_dim;
    for (var k: u32 = 0u; k < half; k = k + 1u) {
        let freq = 1.0 / pow(p.theta, 2.0 * f32(k) / f32(p.head_dim));
        let angle = sign * pos * freq;
        let c = cos(angle);
        let s = sin(angle);
        let a = x[base + 2u * k];
        let b = x[base + 2u * k + 1u];
        x[base + 2u * k] = a * c - b * s;
        x[base + 2u * k + 1u] = a * s + b * c;
    }
}
