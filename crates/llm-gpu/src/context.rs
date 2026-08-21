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
        };

        Ok(Self { device, queue, pipelines })
    }
}
