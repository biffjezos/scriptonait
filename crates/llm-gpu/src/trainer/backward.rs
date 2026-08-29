//! The backward pass: the tied output head's gradient, then each
//! transformer layer in reverse, then the other half of the tied
//! embedding gradient (the input gather's scatter-add). Mirrors
//! `llm_core::model::backward_into` kernel for kernel.

use crate::context::GpuContext;
use crate::model::{dispatch_add_inplace, dispatch_swiglu};

use super::dispatch::Chunks;
use super::layout::{T_ATTN_GAIN, T_MLP_GAIN, T_WK, T_WO, T_WQ, T_WV, T_W_DOWN, T_W_GATE, T_W_UP};
use super::GpuTrainer;

use llm_core::ops;

impl GpuTrainer {
    pub(super) fn encode_backward(&self, chunks: &mut Chunks, ctx: &GpuContext, scatter_groups: usize) {
        let t = self.t_len;
        let c = &self.config;
        let (h, kv, ffn, vocab) = (c.hidden_dim, c.kv_dim(), c.ffn_dim(), c.vocab_size());
        let band = ops::band_width(t, c.effective_window());
        let s = &self.scratch;

        // Output head, tied with the embedding table: its gradient gets
        // both this contribution and the input-gather one at the end.
        self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_logits, &s.final_normed, self.grads.embed(), t, h, vocab);
        self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_logits, self.weights.embed(), &s.d_a, t, h, vocab);
        self.dispatch_rmsnorm_bwd_dgain(
            chunks,
            ctx,
            &s.d_a,
            &s.h_final,
            &s.final_inv_rms,
            self.grads.final_gain(c),
            h,
        );
        self.dispatch_rmsnorm_bwd_dx(
            chunks,
            ctx,
            &s.d_a,
            &s.h_final,
            self.weights.final_gain(c),
            &s.final_inv_rms,
            &s.d_hidden,
            h,
        );

        for l in (0..c.num_layers).rev() {
            let acts = &self.acts[l];

            // --- MLP branch ---
            dispatch_swiglu(chunks.enc(), ctx, &acts.gate, &acts.up, &s.act, t * ffn);
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_hidden,
                &s.act,
                self.grads.layer(l, T_W_DOWN),
                t,
                ffn,
                h,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_hidden,
                self.weights.layer(l, T_W_DOWN),
                &s.d_act,
                t,
                ffn,
                h,
            );
            self.dispatch_swiglu_bwd(chunks, ctx, &s.d_act, &acts.gate, &acts.up, t * ffn);
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_gate,
                &acts.normed2,
                self.grads.layer(l, T_W_GATE),
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_up,
                &acts.normed2,
                self.grads.layer(l, T_W_UP),
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_gate,
                self.weights.layer(l, T_W_GATE),
                &s.d_a,
                t,
                h,
                ffn,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_up,
                self.weights.layer(l, T_W_UP),
                &s.d_b,
                t,
                h,
                ffn,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_b, t * h);

            self.dispatch_rmsnorm_bwd_dgain(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_after_attn,
                &acts.inv_rms2,
                self.grads.layer(l, T_MLP_GAIN),
                h,
            );
            self.dispatch_rmsnorm_bwd_dx(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_after_attn,
                self.weights.layer(l, T_MLP_GAIN),
                &acts.inv_rms2,
                &s.d_c,
                h,
            );
            // The residual splits the gradient: what arrived plus what
            // came back through the norm branch.
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_hidden, &s.d_c, t * h);

            // --- Attention branch ---
            self.dispatch_linear_bwd_dw(
                chunks,
                ctx,
                &s.d_hidden,
                &acts.concat,
                self.grads.layer(l, T_WO),
                t,
                h,
                h,
            );
            self.dispatch_linear_bwd_dx(
                chunks,
                ctx,
                &s.d_hidden,
                self.weights.layer(l, T_WO),
                &s.d_a,
                t,
                h,
                h,
            );
            self.dispatch_attention_bwd(chunks, ctx, acts, band);
            self.dispatch_rope(chunks, ctx, &s.d_q, c.num_heads, true);
            self.dispatch_rope(chunks, ctx, &s.d_k, c.num_kv_heads, true);

            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_q, &acts.normed1, self.grads.layer(l, T_WQ), t, h, h);
            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_k, &acts.normed1, self.grads.layer(l, T_WK), t, h, kv);
            self.dispatch_linear_bwd_dw(chunks, ctx, &s.d_v, &acts.normed1, self.grads.layer(l, T_WV), t, h, kv);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_q, self.weights.layer(l, T_WQ), &s.d_a, t, h, h);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_k, self.weights.layer(l, T_WK), &s.d_b, t, h, kv);
            self.dispatch_linear_bwd_dx(chunks, ctx, &s.d_v, self.weights.layer(l, T_WV), &s.d_c, t, h, kv);
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_b, t * h);
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_a, &s.d_c, t * h);

            self.dispatch_rmsnorm_bwd_dgain(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_in,
                &acts.inv_rms1,
                self.grads.layer(l, T_ATTN_GAIN),
                h,
            );
            self.dispatch_rmsnorm_bwd_dx(
                chunks,
                ctx,
                &s.d_a,
                &acts.h_in,
                self.weights.layer(l, T_ATTN_GAIN),
                &acts.inv_rms1,
                &s.d_c,
                h,
            );
            dispatch_add_inplace(chunks.enc(), ctx, &s.d_hidden, &s.d_c, t * h);
        }

        // The other half of the tied embedding gradient: the input gather.
        self.dispatch_scatter_add(chunks, ctx, &s.d_hidden, self.grads.embed(), scatter_groups, h);
    }
}
