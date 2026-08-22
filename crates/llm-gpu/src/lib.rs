//! WebGPU inference for llm-core models, via `wgpu` and WGSL compute
//! shaders.
//!
//! Generation is what runs here. The prompt is prefilled by llm-core's
//! gradient-checked CPU forward pass and its keys and values uploaded;
//! every token after that is decoded on the GPU. `model.rs`'s header
//! explains why the split falls there.
//!
//! Training is not here. It runs natively, off the user's machine
//! entirely — see `crates/llm-train`.
//!
//! ## Confidence
//!
//! This crate cannot be compiled, let alone run, in the sandbox it was
//! written in: no GPU, no wasm32 target, no route to crates.io. CI
//! compiles it and checks every shader with naga, so the WGSL parses and
//! type-checks. Whether it computes the *right numbers* needs a browser,
//! and `GpuModel::debug_compare_step` is the one-number answer to that:
//! it runs the same decode step on both backends from the same state and
//! reports the largest difference. Anything past float rounding means
//! these kernels are wrong and the CPU path should be used instead.

mod buffers;
mod context;
mod model;

pub use context::{GpuContext, Kernel, ParamsPool};
pub use model::{supports, GpuModel, MAX_HEAD_DIM};
