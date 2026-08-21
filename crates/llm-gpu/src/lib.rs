//! WebGPU-accelerated inference for llm-core models, via `wgpu` + WGSL
//! compute shaders. **Forward pass only** — training runs on llm-core's
//! CPU backend instead (compiled to wasm, so it still runs client-side in
//! the browser, just not GPU-accelerated).
//!
//! That split is a deliberate risk trade-off, not a shortcut: this crate
//! was written without the ability to run WebGPU (or even compile for
//! wasm32) in its development sandbox, so none of it could be tested.
//! Backward-pass kernels are where gradient bugs (and GPU-specific
//! hazards like needing atomic float adds for scatter-style gradient
//! accumulation) are most likely, so keeping training on the
//! gradient-checked CPU path in `llm-core` — and using this crate only
//! for the simpler, atomics-free forward pass — keeps the untested
//! surface as small as it can be while still genuinely using WebGPU for
//! the interactive part of the app (typing a prompt, generating text).
//!
//! See the repo root README for the build steps this needs (the
//! `wasm32-unknown-unknown` target and `wasm-pack`) and for how to report
//! back if something here doesn't compile or doesn't match `llm-core`'s
//! CPU output — cross-checking the two is exactly what `wasm-app`'s
//! "compare to CPU" dev tool (see its README) is for.

mod buffers;
mod context;
mod model;

pub use context::{GpuContext, MAX_GPU_WINDOW};
pub use model::{supports, GpuModel};
