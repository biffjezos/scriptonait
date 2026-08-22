# scriptonait

A transformer language model in Rust, compiled to WebAssembly, that
trains and runs on the GPU of the machine that opens the page. You add
your own text, train a model on it, and generate from a prompt. Nothing
is trained on a server. The page ships no model and downloads none.

Training requires WebGPU. There is no CPU training path: without a
device the page says so and does not train.

## Features

- Add sources by file upload (plain text: `.txt`, `.md`, `.markdown`,
  `.fountain`, `.text`), by pasting text, or by URL fetch. Sources are
  stored in IndexedDB and reloaded on the next visit.
- Train in the browser on your GPU: layers, hidden size, heads, KV heads,
  context length and attention window are settable; steps, batch size,
  learning rate and a duty cycle (effort) control the run. Training can be
  stopped mid-run, and shows a live loss chart.
- Sample the model during training every N steps from a prompt you set.
  The sample is shown in one card that updates in place.
- Generate from a prompt, with temperature, top-k, top-p, repetition
  penalty and seed. Generation streams into the page and can be stopped.
- Export the model to a `.ckpt` file and import one back.
- A prompt is parsed into a form (screenplay, novel, allegory, …), a
  length target, a subject and a work to echo, all shown back as chips
  before generation.
- Screenplay structure of the loaded sources — characters, locations,
  scene count — is extracted with line-shape heuristics and can be
  prepended to the generation prompt.
- Retrieval over your own sources (TF-IDF + cosine similarity, chunked by
  scene) can add similar scenes to the prompt as few-shot context.
- A rule-based QA pass annotates generated text (e.g. characters not seen
  in the sources).
- Optional browser notification when a generation finishes.

## Model

Llama-style decoder-only transformer, implemented from scratch:

- RMSNorm, rotary position embeddings, SwiGLU MLP, no biases, input and
  output embeddings weight-tied.
- Grouped-query attention: `num_kv_heads` must divide `num_heads`.
- Sliding-window attention: `local_window` is separate from `context_len`,
  and attention is stored banded, so cost scales as
  `context_len * local_window`.
- KV cache for decoding.
- Per-layer embeddings (PLE) are implemented and off by default.
- Byte-level BPE tokenizer; base alphabet is all 256 byte values plus
  PAD/BOS/EOS, so any input encodes and there is no unknown token.
- Training is AdamW with decoupled weight decay and global-norm gradient
  clipping. It runs as WGSL kernels in `llm-gpu`; `llm-core` holds the
  CPU reference implementation those kernels are written against and
  gradient-checked by.

Instruction format the model is trained and generated against:

```
BOS TASK form=novel; words=medium; about: two people in space;
         echoing: Plato's allegory of the cave STORY <text> EOS
```

`TASK` and `STORY` are single tokens. One function renders this line for
both the training and the generation path. Length is trained as a bucket;
the exact word target is enforced in code at generation time, which runs
to the target and then to the next sentence boundary, with a ceiling 40%
above the target.

## Where things run

- **Training runs on the GPU, always.** The batch (which windows of your
  text to train on) is sampled in wasm; the forward pass, the loss, the
  backward pass and the AdamW update are WGSL kernels. The weights,
  gradients and both Adam moments stay in GPU memory between steps. One
  small buffer is read back per step: the loss and each tensor's gradient
  norm. The weights come back only when generation or a checkpoint export
  needs them.
- Generation runs on the GPU too. The prompt prefill is the one piece on
  the CPU — that forward pass is the gradient-checked reference. Sampling,
  repetition penalty and the stopping rule are on the CPU in both paths.
  Without WebGPU, generation still works on the CPU; training does not run
  at all.
- The `.wasm` is built in CI by `wasm-pack`; nothing is compiled in the
  browser. `frontend/pkg/` is generated and not in git.

## Layout

```
crates/
  llm-core/      Tokenizer, text prep, corpus, the model (forward,
                 backward, AdamW) as the CPU reference the GPU kernels
                 mirror, generation, instruction parsing, retrieval, QA,
                 screenplay parsing, checkpoints. No dependencies.
  llm-gpu/       The GPU backend: training kernels (trainer.rs) and
                 generation (model.rs), in WGSL.
  shader-check/  Compiles every WGSL shader with naga.
  wasm-app/      wasm-bindgen glue: one WasmLLM class over both crates.
frontend/
  index.html, style.css, app.js   The page.
  worker.js                       Owns the wasm module, off the main thread.
  db.js                           IndexedDB storage for sources.
```

`llm-core` has 150 tests that run without network access, including
analytic-vs-numerical gradient checks per op and for the full model, and
an equivalence test between KV-cached decoding and a full forward pass.
Those checks are what the GPU kernels are written against: each kernel's
index arithmetic mirrors a named `llm-core` function.

## Build and run

```
cargo test

rustup target add wasm32-unknown-unknown
cargo install wasm-pack
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

Serve over HTTP: module workers and `fetch` do not work from `file://`.

## Workflows

- `.github/workflows/deploy.yml` — runs the tests, compiles the shaders,
  builds the wasm, publishes to GitHub Pages on push to
  `dev`/`main`/`master`. Requires Settings → Pages → Source → "GitHub
  Actions" once. It builds the site; it does not train anything.

## Constraints

- URL fetching is subject to CORS; most sites refuse it.
- Uploads are plain text only.
- `llm-gpu` and `wasm-app` are compiled and shader-checked in CI only.
  `WasmLLM.debug_compare_forward` runs the same tokens through the GPU
  forward pass and the CPU reference from the same weights and reports the
  largest logit difference; `GpuModel::debug_compare_step` does the same
  for one decode step.
