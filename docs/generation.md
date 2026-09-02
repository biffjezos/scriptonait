# Generation (Inference tab, Settings tab)

## Prompt

- A single free-text prompt, parsed as an instruction: form (novel,
  screenplay, allegory, …), target length, subject, and an optional
  work to echo. Parsed fields are shown back as chips before
  generating, so a misread prompt is visibly misread.
- The same parse builds the instruction-format line the model was
  trained against — see [model.md](model.md).

## Sampling settings (Settings tab → Inference)

Applied in this order: vocabulary mask → repetition penalty →
temperature → top-k → (renormalize) → min-p → top-p → weighted random
draw.

| Setting | Meaning |
|---|---|
| Device | GPU (default) or CPU for this generation |
| Core loops | Depth to decode at, when the loaded model's Layer sharing is Recurrent core (see [model.md](model.md)) — any value the model was trained across, no retraining needed |
| Temperature | Logit scaling before sampling; 0 forces greedy (always the top token) |
| Top-k | Keep only the k most likely tokens |
| Top-p | Keep the smallest set of tokens whose cumulative probability reaches p (nucleus sampling). Holtzman, Buys, Du, Forbes & Choi, *The Curious Case of Neural Text Degeneration*, 2020 (arXiv:1904.09751) |
| Min-p | Keep tokens at least `min_p × (leading token's probability)` — scales with model confidence, unlike top-p's fixed mass share. Nguyen et al., *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs*, 2024 (arXiv:2407.01082) |
| Repetition penalty | Pushes down the logit of any token seen in the last 128 tokens (dividing a positive logit, multiplying a negative one — both move it down). Keskar, McCann, Varshney, Xiong & Socher, *CTRL: A Conditional Transformer Language Model for Controllable Generation*, 2019 (arXiv:1909.05858) |
| Seed | RNG seed for sampling |
| Length | Continuous (runs to the parsed/estimated target) or Limit to a hard token ceiling |

- Top-k itself: Fan, Lewis & Dauphin, *Hierarchical Neural Story
  Generation*, 2018 (arXiv:1805.04833).
- Generation never emits a token id the loaded corpus doesn't contain
  (except the end marker) — the vocabulary mask above — so an
  early-training model can't fill a sample with tokens it has had no
  reason to push down yet.

## Length control

- A word-count target parsed from the prompt sets a token budget with
  headroom; generation runs to the target, then to the next sentence or
  paragraph boundary, with a hard ceiling 40% above the target in case
  no boundary appears.
- The **Limit to** setting overrides this with a flat token ceiling
  instead.
- Ending reason is reported as one of: reached the length you asked
  for, the model ended the piece (end-of-text), or stopped (Stop
  pressed).

## Decoding

- **KV cache**: past keys/values are reused across steps rather than
  recomputed.
- GPU generation prefills the prompt on the CPU (the gradient-checked
  reference forward pass) and decodes token-by-token on the GPU; CPU
  generation runs the whole thing on the CPU. Either path stays inside
  RoPE's trained position range: a long enough prompt or generation
  triggers a KV-cache rebuild from the most recent half of the text
  before positions run past `Context`.
- CPU generation runs concurrently with GPU training rather than
  waiting for it — the two never contend for the same GPU submission.
  See [architecture.md](architecture.md).

## QA pass (after generation)

Rule-based checks on the generated text, shown as notes, not a quality
score:

- Unbalanced parentheses (often a cut-off parenthetical like `(V.O.)`).
- Length badly off the requested target.
- Repetition loops.

## Stop / Generate

- **Generate** is disabled with no model or an empty prompt; **Stop**
  interrupts a run in progress, keeping whatever text streamed so far.
- Generating with an empty corpus is refused outright (there's nothing
  trained to sample from) rather than silently producing an empty
  result.
