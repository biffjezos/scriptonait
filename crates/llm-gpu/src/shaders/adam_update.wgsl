// One AdamW step, matching llm_core::model::AdamState::step: the same
// betas (0.9 / 0.95), the same epsilon, and decoupled weight decay
// (w -= lr * wd * w) rather than an L2 term folded into the gradient.
//
// `grad_scale` is applied to the gradient as it is read. It carries both
// the batch average (1/batch_size) and the global-norm clip factor, which
// the host computes from this step's gradient-norm readback - so neither
// needs a separate pass over every parameter.
//
// Grid-stride for the same reason as zero.wgsl: the largest tensor here
// has millions of elements, past what one-thread-per-element can dispatch.
struct Params {
    len: u32,
    lr: f32,
    bias1: f32,
    bias2: f32,
    weight_decay: f32,
    grad_scale: f32,
    stride: u32,
    _p0: u32,
};

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const EPS: f32 = 1e-8;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var i: u32 = gid.x; i < p.len; i = i + p.stride) {
        let gi = g[i] * p.grad_scale;
        let mi = BETA1 * m[i] + (1.0 - BETA1) * gi;
        let vi = BETA2 * v[i] + (1.0 - BETA2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let m_hat = mi / p.bias1;
        let v_hat = vi / p.bias2;
        w[i] = w[i] - p.lr * (m_hat / (sqrt(v_hat) + EPS) + p.weight_decay * w[i]);
    }
}
