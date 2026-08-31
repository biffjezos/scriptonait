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
planned end), per-run history with schedule-mode tracking, and one-click project branching
(fork a checkpoint, corpus, and settings mid-run without disturbing the original).

## What's next: different architectures and training approaches

Three directions have come up, all variations on "more than one dense model trained the same
way." Each has a genuinely different cost, and the honest version of that cost matters more
than enthusiasm for any one name:

1. **Specialist branches.** Train several domain-specific models — film scripts, novels,
   psychology and philosophy, storytelling guides were the four named — each forked from one
   seed model via Branch, the way Branch-Train-Merge (Li et al. 2022) and Branch-Train-MiX
   (Sukhbaatar et al. 2024) train domain experts: independently, embarrassingly parallel, no
   architecture change at all. **This needs no new model code.** What's missing is a "model
   library" on the frontend/`wasm-app` side — naming and switching which trained checkpoint is
   active for inference, instead of the app's current single implicit "the model." Cheapest
   direction by a wide margin; the only real design question is that UI.

2. **Mixture-of-Experts.** Every named MoE approach — including BTX's own upcycling step, which
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
   decided.

3. **Self-distillation from an earlier checkpoint.** Discussed as the answer to "there's no
   suitable teacher model" — a Branch checkpoint already *is* a valid frozen teacher, sidestepping
   the usual need for a much bigger pretrained model this app has no path to. `ops::cross_entropy`
   (`crates/llm-core/src/ops/loss.rs`) is already a clean, single-purpose function — there's no
   tangled seam to extract the way FFN had. What's actually missing is new code: a KL-divergence-
   against-teacher-logits loss function, a second frozen model held alongside the student's, and
   on the GPU side a new WGSL kernel next to `cross_entropy.wgsl` plus the dispatch to run it. A
   smaller, more contained version of the same shape of work as MoE.

## Recommended order

**Specialist branches first.** It ships real value (four working, distinct models) with zero
architecture risk and no GPU kernel work — Branch already does the hard part. Whichever of MoE
or distillation gets picked after that should be prototyped on the CPU path first — implemented
in `llm-core`, verified against the gradient-check test — *before* a line of WGSL is written for
it. That's not a new practice: `WasmLLM::debug_compare_forward`'s own doc comment already
describes today's GPU kernels being checked against this same CPU implementation as the
correctness oracle. A new architecture earns that same trust the same way, rather than being
debugged for the first time in a compute shader.

## Verification

`cargo test -p llm-core --lib` (and again with `--features native-trainer`) is the gate for
anything touching `llm-core`, same as today. `llm-gpu`/`wasm-app` changes go through CI
(`.github/workflows/deploy.yml`) — this sandbox has no wasm32 target or crates.io route to
compile them locally, so a pushed commit isn't verified until that run comes back green.
