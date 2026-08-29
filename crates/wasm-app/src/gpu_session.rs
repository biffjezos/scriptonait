//! Opening a WebGPU device, uploading weights to it, and bringing the
//! trained copy back — the GPU session's lifecycle, independent of what
//! runs on it (training, in `training_api.rs`; generation, in
//! `inference.rs`).

use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::dto::json_string;
use crate::{js_err, GpuBackend, WasmLLM};

#[wasm_bindgen]
impl WasmLLM {
    /// Ask the browser for a WebGPU device and upload the weights to it.
    ///
    /// Call once after loading a model. Returns a short description of
    /// what the browser actually gave us — a device name, or why there
    /// isn't one. Generation survives a failure here (it finishes on the
    /// CPU); training does not, because there is no CPU training path.
    pub async fn init_gpu(&self) -> Result<String, JsValue> {
        self.acquire()?;
        let result = self.init_gpu_inner().await;
        self.release();
        result
    }

    /// Everything known about the device, as JSON, for the page to log.
    /// This is the answer to "is it actually using my GPU?" — including
    /// the case that matters most, a software adapter that reports itself
    /// as WebGPU and runs at CPU speed.
    pub fn gpu_report(&self) -> String {
        let inner = self.0.borrow();
        let Some(gpu) = inner.gpu.as_ref() else {
            return "{\"available\":false}".to_string();
        };
        let ctx = &gpu.ctx;
        let training_bytes = gpu.trainer.as_ref().map(|t| t.allocated_bytes()).unwrap_or(0);
        format!(
            "{{\"available\":true,\"adapter\":{},\"backend\":{},\"deviceType\":{},\
             \"isSoftware\":{},\"f16\":{},\"maxWorkgroupsPerDimension\":{},\
             \"maxStorageBufferBindingSize\":{},\"maxBufferSize\":{},\
             \"trainingStateBytes\":{},\"trainerReady\":{}}}",
            json_string(&ctx.adapter_name),
            json_string(&ctx.backend),
            json_string(&ctx.device_type),
            ctx.is_software,
            ctx.has_f16,
            ctx.max_workgroups_per_dimension,
            ctx.max_storage_buffer_binding_size,
            ctx.max_buffer_size,
            training_bytes,
            gpu.trainer.is_some(),
        )
    }

    /// True when the browser handed us a software rasterizer rather than
    /// real hardware — which trains at CPU speed and is worth saying out
    /// loud rather than leaving someone to wonder.
    pub fn gpu_is_software(&self) -> bool {
        self.0.borrow().gpu.as_ref().is_some_and(|gpu| gpu.is_software)
    }

    /// What generation will actually run on: a device description, or
    /// "CPU". The page reports this rather than offering it as a choice.
    pub fn device_summary(&self) -> String {
        match &self.0.borrow().gpu {
            Some(gpu) if gpu.is_software => format!("{} (software renderer)", gpu.summary),
            Some(gpu) => gpu.summary.clone(),
            None => "CPU".to_string(),
        }
    }

    pub fn using_gpu(&self) -> bool {
        self.0.borrow().gpu.is_some()
    }

    /// Bring the trained weights back from the GPU; see
    /// `sync_from_gpu_inner`.
    pub async fn sync_from_gpu(&self) -> Result<(), JsValue> {
        self.acquire()?;
        let result = self.sync_from_gpu_inner().await;
        self.release();
        result
    }

    /// The largest difference between this GPU forward pass and
    /// `llm-core`'s gradient-checked CPU one, over the same tokens and
    /// the same weights. Float rounding lands near `1e-3`; anything much
    /// larger means a kernel is wrong. Nothing calls this in normal use —
    /// it exists so a machine with a real GPU can check the kernels.
    pub async fn debug_compare_forward(&self, tokens: Vec<u32>) -> Result<f32, JsValue> {
        self.acquire()?;
        let result = self.debug_compare_forward_inner(tokens).await;
        self.release();
        result
    }
}

impl WasmLLM {
    async fn init_gpu_inner(&self) -> Result<String, JsValue> {
        let (config, weights) = {
            let inner = self.0.borrow();
            (inner.config, inner.weights.clone())
        };
        if !llm_gpu::supports(&config) {
            return Err(js_err("this model's shape is past what the GPU kernels handle"));
        }
        let ctx = Rc::new(llm_gpu::GpuContext::new().await.map_err(js_err)?);
        let model = llm_gpu::GpuModel::upload(&ctx, &weights, &config).map_err(js_err)?;
        let summary = ctx.adapter_summary.clone();
        let is_software = ctx.is_software;
        let step = self.0.borrow().step;
        self.0.borrow_mut().gpu = Some(GpuBackend {
            ctx,
            model: Some(model),
            trainer: None,
            uploaded_at_step: step,
            summary: summary.clone(),
            is_software,
        });
        Ok(summary)
    }

    /// Bring the trained weights back from the GPU and re-upload them to
    /// the generation path.
    ///
    /// Called before anything that reads the weights on this side —
    /// generating, exporting a checkpoint — rather than after every
    /// step: the weights are megabytes, a step is milliseconds, and
    /// between steps nothing here needs to see them.
    pub(crate) async fn sync_from_gpu_inner(&self) -> Result<(), JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let step = inner.step;
            let Some(gpu) = inner.gpu.as_mut() else { return Ok(()) };
            if gpu.uploaded_at_step == step || gpu.trainer.is_none() {
                return Ok(());
            }
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let trainer = trainer.expect("checked above");
        let downloaded = trainer.download_weights(&ctx).await;
        {
            let inner = &mut *self.0.borrow_mut();
            if let Some(gpu) = inner.gpu.as_mut() {
                gpu.trainer = Some(trainer);
            }
        }
        let weights = downloaded.map_err(js_err)?;

        let config = self.0.borrow().config;
        let model = llm_gpu::GpuModel::upload(&ctx, &weights, &config).map_err(js_err)?;
        let inner = &mut *self.0.borrow_mut();
        inner.weights = weights;
        inner.pretrained = true;
        let step = inner.step;
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.model = Some(model);
            gpu.uploaded_at_step = step;
        }
        Ok(())
    }

    async fn debug_compare_forward_inner(&self, tokens: Vec<u32>) -> Result<f32, JsValue> {
        let (ctx, trainer) = {
            let inner = &mut *self.0.borrow_mut();
            let Some(gpu) = inner.gpu.as_mut() else {
                return Err(js_err("no GPU device"));
            };
            (Rc::clone(&gpu.ctx), gpu.trainer.take())
        };
        let Some(mut trainer) = trainer else {
            return Err(js_err("no training state on the GPU yet — run a step first"));
        };
        let result = trainer.debug_compare_forward(&ctx, &tokens).await;
        let inner = &mut *self.0.borrow_mut();
        if let Some(gpu) = inner.gpu.as_mut() {
            gpu.trainer = Some(trainer);
        }
        result.map_err(js_err)
    }
}
