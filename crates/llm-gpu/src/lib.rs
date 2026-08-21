//! WebGPU-accelerated training and inference for llm-core models, via
//! `wgpu` + WGSL compute shaders. Forward, backward, and Adam all run
//! on-device — see `model.rs`'s module docs for how every backward
//! kernel avoids atomics (gather instead of scatter) and for
//! `GpuModel::supports`'s memory-driven size limit (the dense
//! `[heads, context_len, context_len]` attention-probs cache needed for
//! backward bounds how large a config this backend can take; larger
//! configs fall back to the CPU path).
//!
//! This crate was written without the ability to run WebGPU (or even
//! compile for wasm32) in its development sandbox — see the repo root
//! README's "what's tested and what isn't" for what that does and
//! doesn't mean for confidence in this code, and for the build steps
//! (the `wasm32-unknown-unknown` target and `wasm-pack`) needed to
//! actually run it. `wasm-app` exposes a "compare to CPU" dev tool for
//! checking this crate's forward output against `llm-core`'s
//! gradient-checked CPU reference once you can run it in a browser.

mod buffers;
mod context;
mod model;

pub use context::{GpuContext, MAX_GPU_WINDOW};
pub use model::{supports, GpuModel};
