//! WebGPU device/queue setup and compute pipeline compilation. Kept
//! separate from `model.rs` so the "how do we talk to the GPU at all"
//! plumbing doesn't get lost in the "what does the model compute" logic.

use std::borrow::Cow;

/// The largest sliding-window attention span this backend's naive
/// (per-thread local-array) attention kernel supports — see
/// `shaders/attention.wgsl`'s `MAX_WINDOW` constant, which this must
/// match exactly. A config whose `effective_window()` exceeds this should
/// fall back to the CPU backend instead of using this one.
pub const MAX_GPU_WINDOW: usize = 256;

pub struct Pipelines {
    pub linear: wgpu::ComputePipeline,
    pub add_inplace: wgpu::ComputePipeline,
    pub embedding_gather: wgpu::ComputePipeline,
    pub rmsnorm: wgpu::ComputePipeline,
    pub rope: wgpu::ComputePipeline,
    pub attention: wgpu::ComputePipeline,
    pub swiglu: wgpu::ComputePipeline,
    // Training-only (backward pass + optimizer) kernels.
    pub linear_bwd_dx: wgpu::ComputePipeline,
    pub linear_bwd_dw: wgpu::ComputePipeline,
    pub rmsnorm_bwd_dx: wgpu::ComputePipeline,
    pub rmsnorm_bwd_dgain: wgpu::ComputePipeline,
    pub swiglu_bwd: wgpu::ComputePipeline,
    pub attention_bwd_dscore: wgpu::ComputePipeline,
    pub attention_bwd_dq: wgpu::ComputePipeline,
    pub attention_bwd_dkdv: wgpu::ComputePipeline,
    pub embedding_scatter_add: wgpu::ComputePipeline,
    pub cross_entropy: wgpu::ComputePipeline,
    pub zero: wgpu::ComputePipeline,
    pub scale_inplace: wgpu::ComputePipeline,
    pub adam_update: wgpu::ComputePipeline,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Pipelines,
}

fn make_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        // `None` asks wgpu to derive the bind group layout from the shader
        // itself instead of us hand-declaring one that has to be kept in
        // sync with every .wgsl file by hand.
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl GpuContext {
    /// Requests a WebGPU adapter/device from the browser (or, natively,
    /// from whatever backend `wgpu` finds — used only for
    /// `cargo check`-level type-checking in this sandbox, since there's no
    /// GPU here to actually run against) and compiles every kernel once.
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("No WebGPU adapter available in this browser: {e}"))?;

        // `adapter.limits()`, not `Limits::default()` or any other
        // Rust-side constant: those are fixed sets of limit fields baked
        // into whatever wgpu version this was built against, and browsers
        // lag/differ on which limit keys their `requestDevice` actually
        // recognizes — requesting one wgpu knows about but a given
        // browser doesn't (e.g. `maxInterStageShaderComponents` on some
        // Chrome builds) makes the whole call fail with an
        // "OperationError: ... not recognized", even though every other
        // field would've been fine. Echoing back exactly what the adapter
        // itself just reported is guaranteed to only use keys that
        // browser already understands (it's where they came from) and
        // guaranteed compute-shader-capable (real WebGPU adapters report
        // real compute limits, unlike the WebGL2 downlevel defaults).
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("scriptonait-llm-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Failed to get WebGPU device: {e}"))?;

        let pipelines = Pipelines {
            linear: make_pipeline(&device, "linear", include_str!("shaders/linear.wgsl")),
            add_inplace: make_pipeline(&device, "add_inplace", include_str!("shaders/add_inplace.wgsl")),
            embedding_gather: make_pipeline(
                &device,
                "embedding_gather",
                include_str!("shaders/embedding_gather.wgsl"),
            ),
            rmsnorm: make_pipeline(&device, "rmsnorm", include_str!("shaders/rmsnorm.wgsl")),
            rope: make_pipeline(&device, "rope", include_str!("shaders/rope.wgsl")),
            attention: make_pipeline(&device, "attention", include_str!("shaders/attention.wgsl")),
            swiglu: make_pipeline(&device, "swiglu", include_str!("shaders/swiglu.wgsl")),
            linear_bwd_dx: make_pipeline(&device, "linear_bwd_dx", include_str!("shaders/linear_bwd_dx.wgsl")),
            linear_bwd_dw: make_pipeline(&device, "linear_bwd_dw", include_str!("shaders/linear_bwd_dw.wgsl")),
            rmsnorm_bwd_dx: make_pipeline(&device, "rmsnorm_bwd_dx", include_str!("shaders/rmsnorm_bwd_dx.wgsl")),
            rmsnorm_bwd_dgain: make_pipeline(
                &device,
                "rmsnorm_bwd_dgain",
                include_str!("shaders/rmsnorm_bwd_dgain.wgsl"),
            ),
            swiglu_bwd: make_pipeline(&device, "swiglu_bwd", include_str!("shaders/swiglu_bwd.wgsl")),
            attention_bwd_dscore: make_pipeline(
                &device,
                "attention_bwd_dscore",
                include_str!("shaders/attention_bwd_dscore.wgsl"),
            ),
            attention_bwd_dq: make_pipeline(
                &device,
                "attention_bwd_dq",
                include_str!("shaders/attention_bwd_dq.wgsl"),
            ),
            attention_bwd_dkdv: make_pipeline(
                &device,
                "attention_bwd_dkdv",
                include_str!("shaders/attention_bwd_dkdv.wgsl"),
            ),
            embedding_scatter_add: make_pipeline(
                &device,
                "embedding_scatter_add",
                include_str!("shaders/embedding_scatter_add.wgsl"),
            ),
            cross_entropy: make_pipeline(&device, "cross_entropy", include_str!("shaders/cross_entropy.wgsl")),
            zero: make_pipeline(&device, "zero", include_str!("shaders/zero.wgsl")),
            scale_inplace: make_pipeline(&device, "scale_inplace", include_str!("shaders/scale_inplace.wgsl")),
            adam_update: make_pipeline(&device, "adam_update", include_str!("shaders/adam_update.wgsl")),
        };

        Ok(Self { device, queue, pipelines })
    }
}
