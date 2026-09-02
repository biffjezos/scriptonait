# Implementation plan: beyond one dense model

## Where this actually went

The plan this file used to describe — move training off the user's machine, ship a
CI-pretrained checkpoint, drop `llm-gpu`'s backward/Adam kernels down to an inference-only
backend — is not what happened, and it's worth saying so plainly rather than leaving a stale
description of an abandoned direction sitting here. What shipped instead is more ambitious:
full training, from scratch, in the browser, on WebGPU. `llm-gpu` kept its backward and Adam
kernels rather than dropping them; the browser is the only place training happens, not CI.

Along the way the app grew a real feature set on top of that: a BPE tokenizer, grouped-query
attention with sliding-window attention, KV-cached generation, an instruction-format prompt
parser, AdamW with gradient clipping, a decomposed scheduler (warm-up length, stable-phase
behavior, cool-down timing — both a fixed WSD fraction and an adaptive one that pins the decay
point early on a detected plateau, and plan-length extension for a run still improving past its
planned end), per-run history with schedule-mode tracking, one-click project branching (fork a
checkpoint, corpus, and settings mid-run without disturbing the original), and a model library —
named whole-project snapshots kept in the browser (not just on disk), switchable back to instantly.

**Specialist branches are done.** Training several domain-specific models — film scripts, novels,
psychology and philosophy, storytelling guides were the four named — each forked from one seed
model via Branch, the way Branch-Train-Merge (Li et al. 2022) and Branch-Train-MiX (Sukhbaatar et
al. 2024) train domain experts (independently, embarrassingly parallel, no architecture change at
all) needed no new model code — the only missing piece was naming and switching which trained
checkpoint is active, and the model library is exactly that. Four specialist models are now a
Branch, a Save to Library, and a Switch apart, no filesystem round trip in the loop.

## What's next: different architectures and training approaches

Four directions remain, none shipped yet. Each has a genuinely different cost and a genuinely
different payoff for this app's actual scale and use case, and the honest version of both matters
more than enthusiasm for any one name:

1. **Layer sharing / recurrent depth — two phases of one job, not two directions.** The model's
   depth (`num_layers`) and how many *unique* sets of layer weights actually exist are the same
   number today; splitting them apart trades parameters for repetition — the right trade for a
   machine that's parameter-constrained (a fixed local training ceiling) more than it's
   time-constrained.

   **Phase 1 — static grouping** (Lan, Chen, Goodman, Gimpel, Sharma & Soricut, *ALBERT: A Lite
   BERT for Self-Supervised Learning of Language Representations*, 2019, arXiv:1909.11942; the
   general "recurrent depth" family traces to Dehghani, Gouws, Vinyals, Uszkoreit & Kaiser,
   *Universal Transformers*, 2018, arXiv:1807.03819). `ModelConfig` gains `unique_layers`
   (`num_layers` must be a multiple of it, same validation style as `num_kv_heads` dividing
   `num_heads`); `ModelWeights.layers` shrinks to `unique_layers` entries; the forward/backward
   loop still runs all `num_layers` depth positions (each keeps its own cache — activations differ
   per depth even when weights don't) but indexes weights by depth position, and backward
   accumulates every depth position sharing a group into that group's one gradient buffer before
   the AdamW step — the same principle already proven correct by today's input/output embedding
   weight-tying, just extended from 2 uses to `group_size` uses. Default `unique_layers =
   num_layers` (no sharing, byte-for-byte today's behavior) — new architecture variants ship
   inert, the same way per-layer embeddings did.

   Model this internally as an explicit per-depth-position group mapping, not a bare
   `num_layers / unique_layers` scalar assumption — deliberately more general than phase 1 needs,
   because it's what phase 2 needs and retrofitting it later would mean redoing this same plumbing
   twice. On the GPU side this needs no router and no per-token branching — which group answers
   for depth position 7 is a static fact decided at model-creation time, not a per-token routing
   decision the way MoE's is — so `crates/llm-gpu/src/trainer/layout.rs` only needs each depth
   position's dispatch to read weights from the right offset into a smaller buffer (and backward
   to accumulate into that shared offset), not a routed dispatch loop. Genuinely simpler GPU work
   than MoE's, with no data-dependent control flow at all.

   **Phase 2 — variable loop count** (Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson,
   Kailkhura, Bhatele & Goldstein, *Scaling up Test-Time Compute with Latent Reasoning: A
   Recurrent Depth Approach*, 2025, arXiv:2502.05171). The paper's own architecture — a few
   non-shared "prelude" layers, one shared "core" block looped many times, a few non-shared "coda"
   layers — is a *non-uniform* case of phase 1's same depth-position-to-group mapping, so the
   structural shape is already reachable once phase 1's plumbing exists. What phase 2 actually adds
   is real, separate work with real uncertainty, not a configuration change: training has to
   *sample* the loop count per example so the model learns to produce a usable state at a range of
   depths (a model trained at one fixed depth has no reason to behave well at another), which needs
   truncated backprop through the recurrence (storing every iteration's activations for a large
   loop count is its own memory problem) and carries the known stability hazards of applying the
   same weights many times in sequence. Explicitly a follow-up experiment to attempt once phase 1
   trains correctly, not a guaranteed phase — this app's small model scale hasn't been the target
   of this research, and there's no evidence yet it pays off here the way it does at the paper's
   scale.

   UI-wise, this only ever exposes what's actually built at the time — a "Layer sharing" setting
   next to Model Shape's other fields, shipping with **Off** and **Uniform groups** (an
   `unique_layers` field, phase 1) at first; phase 2, if and when it's built, adds a third mode
   with its own fields (prelude/coda size, loop count or range) rather than a mode the UI already
   shows doing nothing.

2. **Self-distillation from an earlier checkpoint.** Discussed as the answer to "there's no
   suitable teacher model" — a Branch checkpoint (or, now, a Library entry) already *is* a valid
   frozen teacher, sidestepping the usual need for a much bigger pretrained model this app has no
   path to. `ops::cross_entropy` (`crates/llm-core/src/ops/loss.rs`) is already a clean,
   single-purpose function — there's no tangled seam to extract the way FFN had. What's actually
   missing is new code: a KL-divergence-against-teacher-logits loss function, a second frozen model
   held alongside the student's, and on the GPU side a new WGSL kernel next to `cross_entropy.wgsl`
   plus the dispatch to run it. A smaller, more contained version of the same shape of work as MoE
   — fully prototypable and gradient-checked on the CPU path before any WGSL is written.

   Also the direct answer to a real question this app's own Remote training backend raises: train
   something bigger than this machine could ever train locally on `llm-server`
   (`crates/llm-server`, a native binary on a beefier GPU elsewhere), then distill it down into a
   dense student sized for whatever this machine can actually run. No new inference-time kernel
   work needed to get that win — the student is an ordinary dense model this app already knows how
   to run.

3. **Quantization.** Not one of the original three directions, but came up directly from a user
   question about weight precision and is worth planning for honestly rather than bolting on
   later. Today every checkpoint is f32 in compute and bf16 only at rest (see
   `docs/model.md`'s Precision section) — bf16 is *never* computed on directly, `Checkpoint::
   from_bytes` widens it back to f32 the instant it's loaded. Real 4-bit weight quantization would
   need two things this app has neither of: a quantized storage format (grouped scale/zero-point,
   not just a narrower float), and — the actual work — dequantize-on-the-fly WGSL kernels so
   generation never needs the full f32 footprint resident at once. Worth doing once something
   bigger than this machine's own inference ceiling actually exists to shrink (a distilled model
   still too big, or an assembled MoE); on a single dense model at this app's usual scale the size
   saved doesn't move much. An enabler for the other directions, not a standalone win yet.

4. **Mixture-of-Experts.** Every named MoE approach — including BTX's own upcycling step, which
   folds separately-trained specialist branches' FFN weights into one MoE model with a learned
   router — only ever replaces the FFN sublayer with N experts plus a router; attention, RoPE,
   and both RMSNorms are untouched. That seam is now isolated on the CPU reference path:
   `LayerWeights::ffn_forward`/`ffn_backward` in `crates/llm-core/src/model/layer.rs` are the one
   place a routed implementation would need to differ, verified unchanged by the existing
   `full_model_gradient_check` test. The real remaining work is real: the GPU path
   (`crates/llm-gpu/src/trainer/layout.rs`'s `TENSORS_PER_LAYER`, currently a compile-time
   constant baked into pointer arithmetic) would need to become a runtime value driven by an
   expert count, `forward.rs`/`backward.rs`'s fixed per-layer dispatch sequence would need a
   router kernel and a per-expert loop, and the checkpoint format
   (`crates/llm-core/src/checkpoint.rs`, additive-scalar-versioned today) would need a version
   bump to describe a variable number of per-layer tensor groups. None of that is done, or should
   be, before the actual design — expert count, shared-vs-routed experts, router shape — is
   decided. Also the smallest incremental win of the four right now: the model library already
   covers MoE's main practical benefit (several specialist personalities, pick one) for
   effectively zero risk; what MoE adds on top — automatic per-token blending — matters more at a
   production-serving scale than for this app's actual use case.

## Recommended order

**Layer sharing (phase 1, static grouping), then distillation, then quantization if something
still doesn't fit, then MoE** — phase 2 of layer sharing (variable loop count) sitting as an
explicit, uncertain-payoff follow-up to phase 1, not a step in this main sequence. Layer sharing
moved ahead of distillation on the same grounds distillation was chosen over MoE last time: it's
the lowest-risk of the four to build (no router, no data-dependent dispatch, gradient accumulation
already proven correct by today's embedding weight-tying) and it's the one direction that makes
every model this app trains cheaper in parameters immediately, not just a specific future one.
Distillation keeps its own reasoning from before: a loss-function change, CPU-prototypable end to
end before any WGSL, and the direct answer to what the Remote training backend already makes
possible — more effective capability on a small local machine. Quantization only pays off once
something distillation or a MoE assembly produces is actually too big for the machine running it;
built in isolation now, ahead of that need, it would be optimizing a size that isn't yet a problem.
MoE is real, well-scoped work whenever it's picked up, but the model library already delivers most
of its practical value today.

Whichever of these is implemented, the same practice applies: prototype on the CPU path first —
implemented in `llm-core`, verified against the gradient-check test — *before* a line of WGSL is
written for it. That's not a new rule: `WasmLLM::debug_compare_forward`'s own doc comment already
describes today's GPU kernels being checked against this same CPU implementation as the
correctness oracle. A new architecture earns that same trust the same way, rather than being
debugged for the first time in a compute shader.

## Verification

`cargo test -p llm-core --lib` (and again with `--features native-trainer`) is the gate for
anything touching `llm-core`, same as today. `llm-gpu`/`wasm-app` changes go through CI
(`.github/workflows/deploy.yml`) — this sandbox has no wasm32 target or crates.io route to
compile them locally, so a pushed commit isn't verified until that run comes back green.
