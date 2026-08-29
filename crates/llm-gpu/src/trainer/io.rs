//! Checkpoint and optimizer-state marshalling: copying a `ParamSet`
//! to and from a CPU-side `llm_core::model::ModelWeights`, and the
//! forward-pass-only comparison against the CPU reference that verifies
//! the kernels compute the same numbers.

use llm_core::model::ModelWeights;

use crate::context::GpuContext;

use super::layout::ParamSet;
use super::GpuTrainer;

impl GpuTrainer {
    /// Wait for everything submitted so far to actually finish on the
    /// device. Without this a "phase timing" is only how long encoding
    /// took, which is not the question.
    pub(super) async fn sync(&self, ctx: &GpuContext) -> Result<(), String> {
        crate::buffers::sync(&ctx.device, &ctx.queue, &self.scratch.stats).await
    }

    /// Copy the trained weights back into a CPU-side `ModelWeights` —
    /// for checkpoint export, and for handing the model to the
    /// generation backend.
    pub async fn download_weights(&self, ctx: &GpuContext) -> Result<ModelWeights, String> {
        let mut out = ModelWeights::zeros(&self.config);
        let requests: Vec<(&wgpu::Buffer, usize)> =
            self.weights.slots.iter().map(|slot| (&slot.buffer, slot.len)).collect();
        let flat = crate::buffers::read_f32_concat(&ctx.device, &ctx.queue, &requests).await?;
        let mut at = 0usize;
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(self.weights.slots.len());
        for slot in &self.weights.slots {
            values.push(flat[at..at + slot.len].to_vec());
            at += slot.len;
        }
        let mut next = values.into_iter();
        out.embed = next.next().expect("embed");
        for layer in &mut out.layers {
            layer.attn_norm_gain = next.next().expect("attn_norm_gain");
            layer.wq = next.next().expect("wq");
            layer.wk = next.next().expect("wk");
            layer.wv = next.next().expect("wv");
            layer.wo = next.next().expect("wo");
            layer.mlp_norm_gain = next.next().expect("mlp_norm_gain");
            layer.w_gate = next.next().expect("w_gate");
            layer.w_up = next.next().expect("w_up");
            layer.w_down = next.next().expect("w_down");
        }
        out.final_norm_gain = next.next().expect("final_norm_gain");
        Ok(out)
    }

    /// The Adam moment buffers and this trainer's step count.
    ///
    /// A checkpoint without these resumes with the momentum reset, and
    /// the loss visibly jumps at every restart - which, now that the page
    /// saves and restores a model on its own, would happen on every
    /// visit.
    pub async fn download_optimizer(
        &self,
        ctx: &GpuContext,
    ) -> Result<(ModelWeights, ModelWeights, i32), String> {
        let m = self.download_set(ctx, &self.m).await?;
        let v = self.download_set(ctx, &self.v).await?;
        Ok((m, v, self.step))
    }

    /// Put moment buffers back, from a saved state. Shapes must match the
    /// model this trainer holds.
    pub fn upload_optimizer(
        &mut self,
        ctx: &GpuContext,
        m: &ModelWeights,
        v: &ModelWeights,
        step: i32,
    ) -> Result<(), String> {
        self.write_set(ctx, &self.m, m)?;
        self.write_set(ctx, &self.v, v)?;
        self.step = step;
        Ok(())
    }

    /// One `ParamSet` read back into a CPU-side `ModelWeights`.
    async fn download_set(&self, ctx: &GpuContext, set: &ParamSet) -> Result<ModelWeights, String> {
        let requests: Vec<(&wgpu::Buffer, usize)> =
            set.slots.iter().map(|slot| (&slot.buffer, slot.len)).collect();
        let flat = crate::buffers::read_f32_concat(&ctx.device, &ctx.queue, &requests).await?;
        let mut out = ModelWeights::zeros(&self.config);
        let mut at = 0usize;
        let mut next = |len: usize| {
            let slice = flat[at..at + len].to_vec();
            at += len;
            slice
        };
        out.embed = next(self.config.vocab_size() * self.config.hidden_dim);
        let (h, kv, ffn) = (self.config.hidden_dim, self.config.kv_dim(), self.config.ffn_dim());
        for layer in &mut out.layers {
            layer.attn_norm_gain = next(h);
            layer.wq = next(h * h);
            layer.wk = next(kv * h);
            layer.wv = next(kv * h);
            layer.wo = next(h * h);
            layer.mlp_norm_gain = next(h);
            layer.w_gate = next(ffn * h);
            layer.w_up = next(ffn * h);
            layer.w_down = next(h * ffn);
        }
        out.final_norm_gain = next(h);
        Ok(out)
    }

    /// The reverse: a CPU-side `ModelWeights` written into a `ParamSet`.
    fn write_set(&self, ctx: &GpuContext, set: &ParamSet, source: &ModelWeights) -> Result<(), String> {
        let mut tensors: Vec<&[f32]> = vec![&source.embed];
        for layer in &source.layers {
            tensors.extend_from_slice(&[
                &layer.attn_norm_gain[..],
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.mlp_norm_gain,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ]);
        }
        tensors.push(&source.final_norm_gain);
        if tensors.len() != set.slots.len() {
            return Err("optimizer state does not match this model's shape".to_string());
        }
        for (slot, data) in set.slots.iter().zip(tensors) {
            if slot.len != data.len() {
                return Err("optimizer state does not match this model's shape".to_string());
            }
            ctx.queue.write_buffer(&slot.buffer, 0, bytemuck::cast_slice(data));
        }
        Ok(())
    }

    /// The one number that says whether these kernels are right: run the
    /// same sequence through this backend's forward pass and through
    /// `llm_core`'s gradient-checked CPU one, from the same weights, and
    /// return the largest absolute difference between their logits.
    ///
    /// Float rounding over this many accumulations lands around `1e-3`.
    /// Anything much larger means a kernel is wrong.
    pub async fn debug_compare_forward(
        &mut self,
        ctx: &GpuContext,
        tokens: &[u32],
    ) -> Result<f32, String> {
        if tokens.len() != self.t_len {
            return Err(format!("compare needs exactly {} tokens", self.t_len));
        }
        ctx.params.reset();
        ctx.dispatch_count.set(0);
        crate::buffers::write_u32(&ctx.queue, &self.scratch.tokens, tokens);
        let mut chunks = super::dispatch::Chunks::new(ctx, self.dispatches_per_submit);
        self.encode_forward(&mut chunks, ctx);
        chunks.flush();
        let vocab = self.config.vocab_size();
        let gpu_logits =
            crate::buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.logits, self.t_len * vocab).await?;

        let weights = self.download_weights(ctx).await?;
        let (cpu_logits, _) = llm_core::model::forward(&weights, &self.config, tokens);
        Ok(cpu_logits
            .iter()
            .zip(&gpu_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max))
    }

    /// Loss on a batch the model is not trained on: forward pass and
    /// cross-entropy, no backward pass and no optimizer step, so nothing
    /// about the model changes.
    ///
    /// This is the number that separates learning from memorizing.
    /// Training loss falls either way; if this one stops falling while
    /// training loss keeps going, the model has started learning the
    /// sample rather than the language.
    pub async fn eval_loss(
        &mut self,
        ctx: &GpuContext,
        inputs: &[u32],
        targets: &[u32],
    ) -> Result<f32, String> {
        let t = self.t_len;
        if inputs.len() != targets.len() || inputs.is_empty() || inputs.len() % t != 0 {
            return Err("evaluation batch does not match this trainer's sequence length".to_string());
        }
        let batch_size = inputs.len() / t;
        ctx.params.reset();
        ctx.dispatch_count.set(0);

        let mut chunks = super::dispatch::Chunks::new(ctx, self.dispatches_per_submit);
        // Only the loss slot is used, and it accumulates, so it has to
        // start at zero.
        self.dispatch_zero(&mut chunks, ctx, &self.scratch.stats, 1);
        for b in 0..batch_size {
            crate::buffers::write_u32(&ctx.queue, &self.scratch.tokens, &inputs[b * t..(b + 1) * t]);
            crate::buffers::write_u32(&ctx.queue, &self.scratch.targets, &targets[b * t..(b + 1) * t]);
            self.encode_forward(&mut chunks, ctx);
            self.encode_loss(&mut chunks, ctx);
            chunks.flush();
        }
        let stats = crate::buffers::read_f32(&ctx.device, &ctx.queue, &self.scratch.stats, 1).await?;
        Ok(stats[0] / (batch_size * t) as f32)
    }
}
