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

Three directions remain, none shipped yet. Each has a genuinely different cost and a genuinely
different payoff for this app's actual scale and use case, and the honest version of both matters
more than enthusiasm for any one name:

1. **Self-distillation from an earlier checkpoint.** Discussed as the answer to "there's no
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

2. **Quantization.** Not one of the original three directions, but came up directly from a user
   question about weight precision and is worth planning for honestly rather than bolting on
   later. Today every checkpoint is f32 in compute and bf16 only at rest (see
   `docs/model.md`'s Precision section) — bf16 is *never* computed on directly, `Checkpoint::
   from_bytes` widens it back to f32 the instant it's loaded. Real 4-bit weight quantization would
   need two things this app has neither of: a quantized storage format (grouped scale/zero-point,
   not just a narrower float), and — the actual work — dequantize-on-the-fly WGSL kernels so
   generation never needs the full f32 footprint resident at once. Worth doing once something
   bigger than this machine's own inference ceiling actually exists to shrink (a distilled model
   still too big, or an assembled MoE); on a single dense model at this app's usual scale the size
   saved doesn't move much. An enabler for the other two directions, not a standalone win yet.

3. **Mixture-of-Experts.** Every named MoE approach — including BTX's own upcycling step, which
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
   decided. Also the smallest incremental win of the three right now: the model library already
   covers MoE's main practical benefit (several specialist personalities, pick one) for
   effectively zero risk; what MoE adds on top — automatic per-token blending — matters more at a
   production-serving scale than for this app's actual use case.

## Recommended order

**Distillation, then quantization if something still doesn't fit, then MoE.** Not the order this
file used to name, and the reasoning is worth keeping, not just the conclusion: distillation is
the lowest-risk of the three to build (a loss-function change, CPU-prototypable end to end before
any WGSL) and the most direct answer to what this app's own Remote training backend already makes
possible — more effective capability on a small local machine. Quantization only pays off once
something a distillation or a MoE assembly produces is actually too big for the machine running
it; built in isolation now, ahead of that need, it would be optimizing a size that isn't yet a
problem. MoE is real, well-scoped work whenever it's picked up, but the model library already
delivers most of its practical value today.

Whichever of the three is implemented, the same practice applies: prototype on the CPU path first
— implemented in `llm-core`, verified against the gradient-check test — *before* a line of WGSL is
written for it. That's not a new rule: `WasmLLM::debug_compare_forward`'s own doc comment already
describes today's GPU kernels being checked against this same CPU implementation as the
correctness oracle. A new architecture earns that same trust the same way, rather than being
debugged for the first time in a compute shader.

## Verification

`cargo test -p llm-core --lib` (and again with `--features native-trainer`) is the gate for
anything touching `llm-core`, same as today. `llm-gpu`/`wasm-app` changes go through CI
(`.github/workflows/deploy.yml`) — this sandbox has no wasm32 target or crates.io route to
compile them locally, so a pushed commit isn't verified until that run comes back green.
