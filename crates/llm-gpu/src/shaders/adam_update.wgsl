// Standard Adam step, matching llm-core's model::AdamState exactly
// (same beta1/beta2/eps constants). bias1/bias2 (1 - beta^step) are
// computed once on the host per step and passed in, rather than calling
// pow() per element per dispatch.
struct Params {
    len: u32,
    lr: f32,
    bias1: f32,
    bias2: f32,
};

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) {
        return;
    }
    let gi = g[i];
    let mi = BETA1 * m[i] + (1.0 - BETA1) * gi;
    let vi = BETA2 * v[i] + (1.0 - BETA2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    let m_hat = mi / p.bias1;
    let v_hat = vi / p.bias2;
    w[i] = w[i] - p.lr * m_hat / (sqrt(v_hat) + EPS);
}
