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

## Model Shape settings

| Setting | Meaning |
|---|---|
| Layers | Transformer block count |
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
