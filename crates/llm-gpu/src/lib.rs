//! WebGPU inference for llm-core models, via `wgpu` and WGSL compute
//! shaders.
//!
//! Both training and generation run here.
//!
//! `trainer/` is the training loop: forward, backward and AdamW as WGSL
//! kernels, with the weights, gradients and Adam moments resident in GPU
//! memory. A step reads back one small buffer — the loss and the
//! gradient norm — and nothing else.
//!
//! `model.rs` is generation. The prompt is prefilled by llm-core's
//! gradient-checked CPU forward pass and its keys and values uploaded;
//! every token after that is decoded on the GPU. That file's header
//! explains why the split falls there.
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
mod trainer;

pub use context::GpuContext;
pub(crate) use context::{Kernel, ParamsPool};
pub use model::{supports, GpuModel};
pub(crate) use model::MAX_HEAD_DIM;
pub use trainer::{supports_training, GpuStepReport, GpuTrainer};
