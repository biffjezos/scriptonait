//! The forward pass: embedding gather, each transformer layer, the final
//! norm and tied output head, then cross-entropy loss against the
//! targets. Mirrors `llm_core::model::forward` kernel for kernel.

use crate::context::GpuContext;
use crate::model::{dispatch, dispatch_add_inplace, dispatch_linear, dispatch_swiglu, P4};

use super::dispatch::Chunks;
use super::layout::{T_ATTN_GAIN, T_MLP_GAIN, T_WK, T_WO, T_WQ, T_WV, T_W_DOWN, T_W_GATE, T_W_UP};
use super::GpuTrainer;

use llm_core::config::LayerLayout;
use llm_core::ops;

impl GpuTrainer {
    /// `layout` is this step's depth (see `ModelConfig::layer_layout`) —
    /// `self.acts` is allocated for the model's maximum depth
    /// (`config.num_layers`), and a shorter step (a `RecurrentCore` with
    /// fewer than `core_loop_max` loops this time) just leaves the tail
    /// of it untouched, rather than needing its own smaller allocation.
    pub(super) fn encode_forward(&self, chunks: &mut Chunks, ctx: &GpuContext, layout: &LayerLayout) {
        let t = self.t_len;
        let c = &self.config;
        let (h, kv, ffn) = (c.hidden_dim, c.kv_dim(), c.ffn_dim());
        let band = ops::band_width(t, c.effective_window());

        self.dispatch_gather(chunks, ctx, self.weights.embed(), &self.scratch.tokens, &self.scratch.hidden);

        for l in 0..layout.depth() {
            let acts = &self.acts[l];
            // Resolve this depth position to which of
            // `unique_layer_count()` weight sets actually answers for it.
            // When sharing is off (today's default) this is the
            // identity, group == l.
            let g = layout.group(l);
            chunks.copy(&self.scratch.hidden, &acts.h_in, t * h);
            self.dispatch_rmsnorm(
                chunks,
                ctx,
                &self.scratch.hidden,
                self.weights.layer(g, T_ATTN_GAIN),
                &acts.normed1,
                &acts.inv_rms1,
                h,
            );
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(g, T_WQ), &acts.q, t, h, h);
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(g, T_WK), &acts.k, t, h, kv);
            dispatch_linear(chunks.enc(), ctx, &acts.normed1, self.weights.layer(g, T_WV), &acts.v, t, h, kv);
            self.dispatch_rope(chunks, ctx, &acts.q, c.num_heads, false);
            self.dispatch_rope(chunks, ctx, &acts.k, c.num_kv_heads, false);
            self.dispatch_attention_fwd(chunks, ctx, acts, band);
            dispatch_linear(
                chunks.enc(),
                ctx,
                &acts.concat,
                self.weights.layer(g, T_WO),
                &self.scratch.tmp_h,
                t,
                h,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &self.scratch.hidden, &self.scratch.tmp_h, t * h);
            chunks.copy(&self.scratch.hidden, &acts.h_after_attn, t * h);

            // FFN sublayer: rmsnorm, gate/up, SwiGLU, down, add — a fixed,
            // unconditional five-dispatch sequence, mirroring
            // `llm_core::model::layer::LayerWeights::ffn_forward` kernel
            // for kernel (see that method's doc comment for why this is
            // the seam a Mixture-of-Experts FFN would replace). A routed
            // version would add a router kernel choosing/weighting
            // experts per token here, then loop this sequence — with
            // `T_W_GATE`/`T_W_UP`/`T_W_DOWN` indexed per expert, not
            // fixed — once per expert a token is routed to; see
            // `layout.rs`'s note on `TENSORS_PER_LAYER`.
            self.dispatch_rmsnorm(
                chunks,
                ctx,
                &self.scratch.hidden,
                self.weights.layer(g, T_MLP_GAIN),
                &acts.normed2,
                &acts.inv_rms2,
                h,
            );
            dispatch_linear(chunks.enc(), ctx, &acts.normed2, self.weights.layer(g, T_W_GATE), &acts.gate, t, h, ffn);
            dispatch_linear(chunks.enc(), ctx, &acts.normed2, self.weights.layer(g, T_W_UP), &acts.up, t, h, ffn);
            dispatch_swiglu(chunks.enc(), ctx, &acts.gate, &acts.up, &self.scratch.act, t * ffn);
            dispatch_linear(
                chunks.enc(),
                ctx,
                &self.scratch.act,
                self.weights.layer(g, T_W_DOWN),
                &self.scratch.tmp_h,
                t,
                ffn,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &self.scratch.hidden, &self.scratch.tmp_h, t * h);
        }

        chunks.copy(&self.scratch.hidden, &self.scratch.h_final, t * h);
        self.dispatch_rmsnorm(
            chunks,
            ctx,
            &self.scratch.hidden,
            self.weights.final_gain(&self.config),
            &self.scratch.final_normed,
            &self.scratch.final_inv_rms,
            h,
        );
        // Weight-tied output head: logits = final_normed @ embed^T.
        dispatch_linear(
            chunks.enc(),
            ctx,
            &self.scratch.final_normed,
            self.weights.embed(),
            &self.scratch.logits,
            t,
            h,
            self.config.vocab_size(),
        );
    }

    pub(super) fn encode_loss(&self, chunks: &mut Chunks, ctx: &GpuContext) {
        let t = self.t_len;
        let params = ctx.params.alloc(
            &ctx.device,
            &ctx.queue,
            P4 { a: t as u32, b: self.config.vocab_size() as u32, c: 0, d: 0 },
        );
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.scratch.logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.scratch.targets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.scratch.d_logits.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.scratch.loss_rows.as_entire_binding() },
        ];
        // One workgroup per row; its 256 threads split the vocabulary.
        dispatch(chunks.enc(), ctx, &ctx.pipelines.cross_entropy, &entries, (t as u32, 1, 1));
        // The per-row losses reduce straight into the step's loss slot,
        // so the batch costs one readback in total rather than one per
        // sequence.
        self.dispatch_reduce(chunks, ctx, &self.scratch.loss_rows, t, 0, false);
    }

    /// Groups this sequence's token positions by token id and uploads the
    /// result for the embedding scatter. Returns how many distinct ids
    /// there are, which is the scatter's dispatch width.
    pub(super) fn upload_scatter_index(&self, ctx: &GpuContext, tokens: &[u32]) -> usize {
        let mut ids: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut positions_by_id: Vec<Vec<u32>> = Vec::with_capacity(tokens.len());
        for (position, &id) in tokens.iter().enumerate() {
            match ids.iter().position(|&existing| existing == id) {
                Some(group) => positions_by_id[group].push(position as u32),
                None => {
                    ids.push(id);
                    positions_by_id.push(vec![position as u32]);
                }
            }
        }
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        let mut positions = Vec::with_capacity(tokens.len());
        offsets.push(0u32);
        for group in &positions_by_id {
            positions.extend_from_slice(group);
            offsets.push(positions.len() as u32);
        }
        crate::buffers::write_u32(&ctx.queue, &self.scratch.row_ids, &ids);
        crate::buffers::write_u32(&ctx.queue, &self.scratch.row_offsets, &offsets);
        crate::buffers::write_u32(&ctx.queue, &self.scratch.row_positions, &positions);
        ids.len()
    }
}
