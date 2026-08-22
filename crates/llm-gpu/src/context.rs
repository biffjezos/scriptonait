//! WebGPU device/queue setup and compute pipeline compilation. Kept
//! separate from `model.rs` so the "how do we talk to the GPU at all"
//! plumbing doesn't get lost in the "what does the model compute" logic.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};

/// A compute pipeline plus its bind group layout.
///
/// The layout is fetched once at build time rather than by calling
/// `pipeline.get_bind_group_layout(0)` on every dispatch: that call
/// allocates a fresh layout object each time, and on the web it is a
/// round trip into the browser's WebGPU implementation — thousands of
/// them per training step, for a value that never changes.
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

pub struct Pipelines {
    pub linear: Kernel,
    pub add_inplace: Kernel,
    pub embedding_gather: Kernel,
    pub rmsnorm: Kernel,
    pub rope: Kernel,
    pub attention_decode: Kernel,
    pub attention_fwd: Kernel,
    pub swiglu: Kernel,
    // Training. Generation never dispatches any of these, but they are
    // compiled with everything else: pipeline creation is where a WGSL
    // error surfaces, and finding out that the backward pass does not
    // compile at the end of the first training step is finding out too
    // late.
    pub linear_bwd_dw: Kernel,
    pub linear_bwd_dx: Kernel,
    pub rmsnorm_bwd_dx: Kernel,
    pub rmsnorm_bwd_dgain: Kernel,
    pub swiglu_bwd: Kernel,
    pub attention_bwd_dscore: Kernel,
    pub attention_bwd_dq: Kernel,
    pub attention_bwd_dkdv: Kernel,
    pub embedding_scatter_add: Kernel,
    pub cross_entropy: Kernel,
    pub adam_update: Kernel,
    pub zero: Kernel,
    pub reduce: Kernel,
    pub reduce_finish: Kernel,
}

/// A recycled pool of tiny uniform buffers, one per dispatch.
///
/// Every dispatch needs its own few-word `Params` buffer, and the buffer
/// has to stay distinct from its neighbours' because several dispatches
/// are encoded before the queue submit that runs them. Creating one per
/// dispatch (which is what this replaces) meant thousands of real GPU
/// buffer allocations per training step, each a round trip through the
/// browser's WebGPU implementation. The buffers here are created once and
/// rewritten with `queue.write_buffer`, which is ordered against the
/// submits that consume them, so a slot handed out again on a later step
/// can never disturb work already submitted.
pub struct ParamsPool {
    buffers: RefCell<Vec<wgpu::Buffer>>,
    cursor: Cell<usize>,
}

impl ParamsPool {
    fn new() -> Self {
        Self { buffers: RefCell::new(Vec::new()), cursor: Cell::new(0) }
    }

    /// Starts handing slots out from the beginning again. Call once at
    /// the start of each top-level GPU operation (a training step, one
    /// generation forward pass), never mid-encoding.
    pub fn reset(&self) {
        self.cursor.set(0);
    }

    /// The next free slot, written with `value`. `wgpu::Buffer` is a
    /// cheap reference-counted handle, so the clone is a refcount bump,
    /// not an allocation.
    pub fn alloc<T: bytemuck::Pod>(&self, device: &wgpu::Device, queue: &wgpu::Queue, value: T) -> wgpu::Buffer {
        const SLOT_BYTES: u64 = 32; // every Params struct here is <= 32 bytes
        debug_assert!(std::mem::size_of::<T>() as u64 <= SLOT_BYTES);
        let index = self.cursor.get();
        self.cursor.set(index + 1);
        let mut buffers = self.buffers.borrow_mut();
        if index == buffers.len() {
            buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params-pool-slot"),
                size: SLOT_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buffer = buffers[index].clone();
        drop(buffers);
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&value));
        buffer
    }
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Every compute dispatch issued through `model::dispatch`, counted
    /// so a step can report how many it took. Per-dispatch overhead is
    /// real on the web (each one is a call into the browser), and this
    /// is the number that says whether it dominates.
    pub dispatch_count: Cell<u32>,
    pub pipelines: Pipelines,
    pub params: ParamsPool,
    /// What the browser actually gave us. Reported to the UI because
    /// "training is slow" has very different answers depending on whether
    /// this is a real GPU or a software rasterizer.
    pub adapter_summary: String,
    /// True when the adapter is a CPU/software implementation (SwiftShader,
    /// WARP, lavapipe) rather than real hardware.
    pub is_software: bool,
    /// The adapter's own fields, kept separate from the summary string so
    /// the page can log them individually — "is this actually my GPU?"
    /// is the first question a slow training run raises.
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
    /// Limits that decide how big a dispatch can be, logged because a
    /// browser quietly capping one of them changes what the kernels can
    /// do.
    pub max_workgroups_per_dimension: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
}

fn make_pipeline(device: &wgpu::Device, label: &str, source: &str) -> Kernel {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        // `None` asks wgpu to derive the bind group layout from the shader
        // itself instead of us hand-declaring one that has to be kept in
        // sync with every .wgsl file by hand.
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    Kernel { pipeline, layout }
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
                // HighPerformance, not LowPower: on a laptop with both
                // an integrated and a discrete GPU, LowPower explicitly
                // asks the browser for the *integrated* one. That is a
                // reasonable default for drawing a UI and precisely the
                // wrong one for training a model, which is the only
                // reason this backend exists.
                power_preference: wgpu::PowerPreference::HighPerformance,
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
            attention_decode: make_pipeline(
                &device,
                "attention_decode",
                include_str!("shaders/attention_decode.wgsl"),
            ),
            attention_fwd: make_pipeline(
                &device,
                "attention_fwd",
                include_str!("shaders/attention_fwd.wgsl"),
            ),
            swiglu: make_pipeline(&device, "swiglu", include_str!("shaders/swiglu.wgsl")),
            linear_bwd_dw: make_pipeline(
                &device,
                "linear_bwd_dw",
                include_str!("shaders/linear_bwd_dw.wgsl"),
            ),
            linear_bwd_dx: make_pipeline(
                &device,
                "linear_bwd_dx",
                include_str!("shaders/linear_bwd_dx.wgsl"),
            ),
            rmsnorm_bwd_dx: make_pipeline(
                &device,
                "rmsnorm_bwd_dx",
                include_str!("shaders/rmsnorm_bwd_dx.wgsl"),
            ),
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
            cross_entropy: make_pipeline(
                &device,
                "cross_entropy",
                include_str!("shaders/cross_entropy.wgsl"),
            ),
            adam_update: make_pipeline(&device, "adam_update", include_str!("shaders/adam_update.wgsl")),
            zero: make_pipeline(&device, "zero", include_str!("shaders/zero.wgsl")),
            reduce: make_pipeline(&device, "reduce", include_str!("shaders/reduce.wgsl")),
            reduce_finish: make_pipeline(
                &device,
                "reduce_finish",
                include_str!("shaders/reduce_finish.wgsl"),
            ),
        };

        let info = adapter.get_info();
        let limits = device.limits();
        let is_software = info.device_type == wgpu::DeviceType::Cpu;
        let adapter_summary = format!(
            "{} ({:?}, {:?}){}",
            info.name,
            info.backend,
            info.device_type,
            if is_software { " — SOFTWARE renderer, not a real GPU" } else { "" }
        );

        Ok(Self {
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
            max_buffer_size: limits.max_buffer_size,
            device,
            queue,
            dispatch_count: Cell::new(0),
            pipelines,
            params: ParamsPool::new(),
            adapter_summary,
            is_software,
        })
    }
}
