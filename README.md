# scriptonait

Train your own language model on your own writing, in a browser tab, on
your own GPU.

**In one sentence:** you drop in the scripts or books you like, the page
teaches a small AI to write like them using your graphics card, and then
you ask it for a scene — nothing is uploaded, nothing is downloaded, and
the model is yours.

**Why:** a language model trained on a hundred thousand people's text
writes like nobody. One trained on the twenty scripts you chose writes
like those, and it does it on hardware you already own, without an
account, an API key, or your text leaving the machine.

Training requires WebGPU. There is no CPU training path: without a
device the page says so and does not train.

## What it does

**Your text**

- Add sources by file upload (plain text: `.txt`, `.md`, `.markdown`,
  `.fountain`, `.text`), by pasting, or by URL fetch. They are stored in
  IndexedDB and reloaded on the next visit.
- A BPE vocabulary is learned from your own corpus, sized to it, so the
  tokens are the phrases your text actually uses.
- Five percent of every source is held out of training, to measure the
  model against text it has never seen. The measurement uses a fixed set
  of windows, evenly spaced across that held-out text and identical on
  every measurement, so two numbers differ because the weights differ
  and not because two random draws landed on different text.

**Training**

- Runs entirely on your GPU: forward pass, cross-entropy, backward pass
  and AdamW are WGSL compute kernels. Weights, gradients and both Adam
  moment buffers stay in GPU memory between steps.
- AdamW with decoupled weight decay, global gradient-norm clipping, and a
  cosine schedule with warmup shaped to the run length you asked for and
  anchored to the step the model is already at, so a resumed run gets a
  whole schedule rather than the tail of one.
- Plateau detection: when held-out loss goes four measurements without
  improving, the learning rate is halved on top of the schedule, down to
  a floor. The cosine says where the plan expected to be; this says what
  the run actually needed, and the two are kept separate so both are
  readable. At the floor the page says so rather than cutting again —
  by then the limit is the corpus, not the rate.
- Model shape is yours to set: layers, hidden size, heads, KV heads,
  context length, attention window. So are steps, batch size, learning
  rate, and an effort setting that decides how much of the machine the
  run may take.
- Every training setting is optional. The first run benchmarks the
  machine it is on — how much work fits in one command buffer before the
  driver's watchdog takes the device away, how many sequences a batch can
  hold before a step stops being interruptible — and the settings are
  read off that measurement. The result is stored per adapter and loaded
  on the next visit, so it is measured once, not guessed and not
  hard-coded for anybody's hardware.
- Stop any time; progress is kept.
- A training plan, recomputed as the run goes: which phase it is in
  (warm-up, learning, plateau, overfitting, cooling down), what that
  phase means for the numbers you are looking at, tokens seen, how many
  times it has been over your text, an estimate of what is left — and
  what would actually help, named with the number behind it. Sources are
  classified by line shape (film scripts, novels, essays, verse), so a
  corpus that is all one thing is told so.
- Live loss chart with three curves on one axis: per-step training loss,
  held-out loss, and training loss measured on a fixed set of windows
  drawn exactly as the held-out set is. That third one exists because
  the first two cannot honestly be compared — 40% of training windows
  start at a source's opening and no held-out window ever does, so the
  two separate as soon as the model learns what an opening looks like,
  a few hundred steps in and long before anything could be memorized.
  The gap worth reading is between the dashed curve and the held-out
  one.
- Samples the model as it trains, into a single card that updates in
  place.
- The trained model and its optimizer state are saved to the browser
  after every run and restored on the next visit, so a reload costs
  nothing and training resumes with its momentum intact.

**Writing**

- Generate from a prompt, with temperature, top-k, top-p, min-p,
  repetition penalty and seed. Text streams in and can be stopped.
- Min-p truncates relative to the best token rather than to a fixed
  share of the mass, so a confident step stays sharp and an uncertain
  one stays open. At this model size the distribution is often barely
  peaked, and that is exactly where top-p behaves worst.
- The prompt is read as an instruction — form (screenplay, novel,
  allegory, …), length, subject, a work to echo — and shown back as chips
  before generating, so a misread prompt is visibly misread.
- Generation stays inside the vocabulary your text uses.
- Characters, locations and scene count are extracted from your sources
  by line shape, and can be prepended to the prompt.
- Retrieval over your own sources (TF-IDF, chunked by scene) can add
  similar scenes as few-shot context.
- A rule-based QA pass annotates the result (for example: a character who
  never appears in your sources).
- Optional browser notification when a generation finishes.

**Seeing what happened**

- Per-step console logging: loss, held-out loss, gradient norm, learning
  rate, tokens/second, dispatch and submission counts.
- Measurements loss cannot make: **bits per byte** (the held-out loss in
  a form comparable between two vocabularies, and against gzip's ~2.5 on
  English prose), the fraction of a sample's words that appear anywhere
  in your own sources, how many of its four-word runs it had already
  written, and how many of its words are distinct. Shown on the sample
  card and in the plan; `scriptonait.evaluate(text)` measures any text.
- `scriptonait.profile()` in the console times one step per phase — zero,
  forward, loss, backward, reduce, readback, AdamW — at four
  command-buffer sizes, so "it is slow" becomes a measurement.
- `scriptonait.benchmark()` re-runs the machine sweep and logs every
  candidate it timed; `scriptonait.machine()` shows what is stored.
- The device the browser handed over is named, and a software renderer
  is called out as one.
- Export the model to a `.ckpt` file, and load one back.

## The model

Llama-style decoder-only transformer, implemented from scratch:

- RMSNorm, rotary position embeddings, SwiGLU MLP, no biases, input and
  output embeddings weight-tied.
- Grouped-query attention: `num_kv_heads` must divide `num_heads`.
- Sliding-window attention: `local_window` is separate from
  `context_len`, and attention is stored banded, so cost scales as
  `context_len * local_window`.
- KV cache for decoding.
- Per-layer embeddings (PLE) are implemented and off by default.
- Byte-level BPE tokenizer: the base alphabet is all 256 byte values plus
  specials, so any input encodes and there is no unknown token.

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

- **Training runs on the GPU, always.** The batch — which windows of your
  text to train on — is chosen in wasm; everything after that is WGSL.
  One small buffer is read back per step: the loss and each tensor's
  gradient norm. The weights cross the bus only when generation or a save
  needs them.
- Generation runs on the GPU too. The prompt prefill is the one piece on
  the CPU, because that forward pass is the gradient-checked reference.
  Sampling, repetition penalty and the stopping rule are on the CPU in
  both paths. Without WebGPU, generation still works on the CPU;
  training does not run at all.
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
  db.js                           IndexedDB: your sources, your model, and
                                  what this machine benchmarked.
```

`llm-core` has 156 tests that run without network access, including
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

`.github/workflows/deploy.yml` runs the tests, compiles the shaders,
parses the frontend, builds the wasm and publishes to GitHub Pages on
push to `dev`/`main`/`master`. It builds the site; it does not train
anything.

## Constraints

- URL fetching is subject to CORS; most sites refuse it.
- Uploads are plain text only.
- `llm-gpu` and `wasm-app` are compiled and shader-checked in CI only.
  `WasmLLM.debug_compare_forward` runs the same tokens through the GPU
  forward pass and the CPU reference from the same weights and reports
  the largest logit difference; `GpuModel::debug_compare_step` does the
  same for one decode step.
