# Model (Settings tab → Model Shape)

Llama-style decoder-only transformer, implemented from scratch (no
pretrained weights, no third-party model code).

## Architecture

- **RMSNorm** in place of LayerNorm, no bias terms anywhere, input and
  output embeddings weight-tied.
  Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019
  (arXiv:1910.07467).
- **Rotary position embeddings (RoPE)** for queries and keys.
  Su, Lu, Pan, Murtadha, Wen & Liu, *RoFormer: Enhanced Transformer with
  Rotary Position Embedding*, 2021 (arXiv:2104.09864).
- **Grouped-query attention (GQA)**: `KV heads` must divide `Heads`;
  `KV heads` = `Heads` is ordinary multi-head attention.
  Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón & Sanghai, *GQA:
  Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints*, 2023 (arXiv:2305.13245).
- **Sliding-window attention**: `Attention window` is independent of
  `Context`; a token attends only to the most recent `window` positions,
  banded rather than dense, so attention cost scales as
  `context_len * local_window`, not `context_len²`.
  Jiang et al., *Mistral 7B*, 2023 (arXiv:2310.06825).
- **SwiGLU** feed-forward block.
  Shazeer, *GLU Variants Improve Transformer*, 2020 (arXiv:2002.05202).
- **Per-layer embeddings (PLE)**: implemented, off by default.
- **KV cache** for decoding (see [generation.md](generation.md)).
- **Layer sharing (ALBERT-style static grouping)**: implemented, off by
  default. See the section below.

## Precision

- **Compute — f32, always.** Weights, activations, gradients, and both
  AdamW moment buffers (`m`/`v`) are f32 on the GPU and in the CPU
  reference implementation alike. WGSL (WebGPU's shader language) has
  no f64 type; nothing in this app trains or generates in anything
  other than f32.
- **Checkpoint weights at rest — bf16.** Every export (Save, Auto-save,
  Export Project, Branch, the Library — see
  [project.md](project.md)) narrows weights to bf16 (the top 16 bits of
  each f32) purely to halve file size, and widens them back to f32 on
  import. bf16 over fp16 because the conversion is a plain shift/round
  with no subnormal or overflow handling to get subtly wrong, and bf16
  keeps f32's exponent range — the same format large pretrained models
  are commonly trained in natively, so a small quality cost at this
  step is well precedented.
- **Optimizer state at rest — f32, never narrowed.** Adam's momentum
  has to survive a save/resume intact, and the moment buffers are
  already 3x the size of the weights, so the size/precision trade
  isn't taken there.

## Model Shape settings

| Setting | Meaning |
|---|---|
| Layers | Transformer block count (depth positions) |
| Layer sharing | Off, or Uniform groups (see below) |
| Unique layers | Distinct weight sets, when Layer sharing is Uniform groups |
| Hidden size | Residual stream width |
| Heads | Attention heads |
| KV heads | Key/value heads (GQA); must divide Heads |
| Context | Max sequence length the model is trained/run with |
| Attention window | Sliding-window size, ≤ Context |

- Priced as you type, before anything is built: parameter count, GPU
  memory required to train the shape against the ceiling that would
  reject it, head and MLP widths, the vocabulary size the corpus
  supports, and how much a shape wastes if its dimensions fall off the
  GPU kernels' 64-wide tile grid. The estimate is computed from the same
  `ModelConfig` struct the model is actually built from, so it cannot
  drift from what it estimates.
- Changing shape starts a new model; it does not resize an existing one.

## Layer sharing

Trades parameters for repetition: fewer distinct weight sets are run at
more depth positions, so a model with the same depth (and therefore the
same per-step GPU work) costs fewer parameters to store and train.

- **Off** (default): every one of `Layers` depth positions has its own
  weights — `Unique layers` equals `Layers`, today's behavior before this
  setting existed at all.
- **Uniform groups**: `Layers` is split into `Unique layers` equal-length
  contiguous spans, in order — the first span's depth positions all run
  the first weight set, the second span the second, and so on.
  `Unique layers` must evenly divide `Layers`. Switching this on suggests
  a starting `Unique layers` value (the largest divisor of `Layers` that
  is at most half of it); typing a different value is a divisor check,
  not a retrain.

  Lan, Chen, Goodman, Gimpel, Sharma & Soricut, *ALBERT: A Lite BERT for
  Self-Supervised Learning of Language Representations*, 2019
  (arXiv:1909.11942). The general "recurrent depth" family traces to
  Dehghani, Gouws, Vinyals, Uszkoreit & Kaiser, *Universal Transformers*,
  2018 (arXiv:1807.03819), which shares one set of weights across every
  depth position (the `Unique layers = 1` case here).

- Every depth position still keeps its own activations and its own
  key/value cache during generation — two positions sharing a weight set
  still see a different residual stream (whatever came before them in the
  stack), so they compute different numbers even from identical weights.
  Only the weights, gradients, and Adam moment buffers are shared; nothing
  about attention, RoPE, or the forward/backward loop's shape changes.
- The checkpoint format stores `Unique layers` alongside the rest of the
  shape; a file from before this setting existed loads with `Unique
  layers` equal to `Layers` (no sharing), exactly its previous behavior.

## Instruction format

Every example — training and generation alike — is framed the same way:

```
BOS TASK form=novel; words=medium; about: two people in space;
         echoing: Plato's allegory of the cave STORY <text> EOS
```

- `TASK` and `STORY` are single reserved tokens, not text.
- One function renders this line for both the training and the
  generation path, so the two can't drift apart.
- Length is trained as a bucket (short/medium/long/...); the exact word
  target typed into a prompt is enforced separately, at generation time
  (see [generation.md](generation.md)).

## Where the model runs

- Training runs on the GPU only — there is no CPU training path. Forward
  pass, cross-entropy, backward pass, and AdamW are WGSL compute
  kernels; weights, gradients, and both Adam moment buffers stay in GPU
  memory between steps.
- Generation also runs on the GPU by default (see
  [generation.md](generation.md) for the CPU fallback and why it exists).
- See [architecture.md](architecture.md) for how the GPU kernels and the
  CPU reference implementation relate.
