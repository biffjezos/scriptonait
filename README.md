# scriptonait

A small transformer language model, trained from scratch **in your browser**
on your own film scripts, stories, or books — written in Rust, compiled to
WebAssembly, with generation accelerated by WebGPU.

There's no pretrained base model and no server: you supply the training
text (paste it, upload files, or fetch a URL), pick a model size, and train
it yourself, entirely client-side. It's a small, educational/hobbyist-scale
model — a few hundred thousand to a few million parameters, not a
ChatGPT-scale system — good at picking up the local style, structure, and
vocabulary of whatever you feed it (scene headings, character-cue
formatting, recurring phrasing), not at long-range plot coherence.

## Architecture

- **Tokenizer**: byte-level (256 byte values + PAD/BOS/EOS = 259 tokens).
  No BPE training step, never produces an "unknown token", and keeps every
  embedding table tiny regardless of corpus size or language.
- **Model**: a small Llama/Gemma-style decoder-only transformer —
  RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP, weight-tied
  input/output embedding, no biases.
- **Per-layer embeddings (PLE)**: alongside the usual shared input
  embedding, each layer has its own small embedding table, gathered by
  token id and added straight into that layer's residual stream — a plain
  vector lookup, no matmul (Gemma 3n's PLE technique). This is what "vector
  lookups" in the original ask refers to, alongside the standard input
  embedding.
- **Sliding-window attention**: attention window is configurable
  separately from context length (Mistral-style). For long documents
  (scripts, book chapters) you can push context length up while keeping
  the window — and so the compute/memory cost — small, since local
  structure (a scene, a line of dialogue) rarely needs to look back
  thousands of tokens. The CPU backend stores attention probabilities
  **banded** (`[heads, T, window]`), so both the time and the memory are
  genuinely `context_len * local_window`, not `context_len^2`.
- **Text prep**: HTML is stripped for URL sources; all sources get
  whitespace-normalized, but — deliberately — *leading indentation is
  preserved*. Plain-text screenplay exports commonly use indentation as
  the only signal separating scene headings, character cues, dialogue, and
  action lines, so a naive "trim every line" pass would destroy exactly
  the structure worth learning.

See doc comments in `crates/llm-core/src/config.rs` and `model.rs` for the
full per-layer layout and parameter-count formula (also surfaced live in
the UI as you adjust settings).

### Memory, and why the UI refuses some shapes

The dominant memory cost of a training step is not the weights — it's the
activation cache the backward pass needs, and within that, the attention
probabilities: `num_layers * num_heads * context_len * local_window`
floats, live all at once. At the largest shape the UI's inputs allow
(16 layers, 1024 nodes, 16 heads, 4096-token full attention) that is
around 5 GB even with banded storage, against 790 MB of weights.

So `ModelConfig::memory_bytes(true)` counts activations
(`ModelConfig::activation_bytes`), the size estimate under the model
settings shows the split and flags heavy configs, and
`ModelConfig::validate` rejects anything over
`MAX_TRAINING_BYTES` (2 GB) outright — a wasm32 heap cannot hold more, so
such a config doesn't train slowly, it allocates until the tab dies and
drags the machine into swap first.

### Where training happens vs. where WebGPU is used

Both training (forward + backward + Adam) and generation can run either on
`llm-core`'s CPU implementation (compiled to wasm, works everywhere) or on
`llm-gpu`'s WebGPU backend (`wgpu` + WGSL compute shaders, needs a browser
with WebGPU) — CPU is always the default and the fallback; WebGPU is an
opt-in toggle in the Train and Generate panels, and both automatically fall
back to CPU if the model's attention window/context length is too large for
the GPU backend's fixed-size kernels (`gpu_supported()`/`MAX_GPU_WINDOW`) or
if no WebGPU device is available at all.

GPU training and CPU training keep **separate weight copies and separate
Adam optimizer state** while a session trains on the GPU — `train_step_gpu`
only updates the GPU-resident weights, so the CPU copy (and anything
derived from it: CPU generation, "Export weights", checkpoint saves) is
stale until `sync_weights_from_gpu` runs. The frontend does this
automatically whenever it's needed (before generating, exporting, or
stopping GPU training), but it does mean switching from GPU back to CPU
training resets Adam momentum, same as importing a checkpoint would (the
two optimizer states aren't compatible with each other).

See "What's tested and what isn't" below for how much confidence to place
in the WebGPU path specifically.

## Project layout

```
crates/
  llm-core/   Tokenizer, text prep, corpus/batch sampling, model
              (forward+backward+Adam), generation. Zero external
              dependencies — builds and its 100 tests run with no network
              access. This is the verified reference implementation.
  llm-gpu/    WebGPU (wgpu + WGSL) backend, mirroring llm-core's forward
              pass, backward pass, and Adam optimizer kernel-for-kernel —
              full training and generation, not forward-only.
  wasm-app/   wasm-bindgen glue exposing both as one `WasmLLM` class.
frontend/
  index.html, style.css, app.js   Main-thread UI.
  worker.js                       Owns the wasm module; runs training/
                                   generation off the main thread.
  db.js                           IndexedDB wrapper (sources + checkpoints).
  pkg/                            wasm-pack's build output — generated,
                                   not checked in (see Build below).
```

`llm-core` is the only crate in the root Cargo workspace. `llm-gpu` and
`wasm-app` are deliberately standalone crates (each has its own
`[workspace]` marker) — they depend on `wgpu`/`wasm-bindgen` and the
`wasm32-unknown-unknown` target, which the workspace root doesn't need and
which weren't available in this project's development sandbox (more
below), so keeping them out of the shared workspace keeps `cargo test` at
the repo root fully offline-buildable.

## Deploy (GitHub Pages, recommended)

`.github/workflows/deploy.yml` builds and deploys this automatically —
this is the intended way to get a working, live copy, since it runs on a
GitHub Actions runner with normal internet access (unlike this project's
dev sandbox; see "What's tested and what isn't" below). It installs the
Rust `wasm32` target and `wasm-pack`, runs `llm-core`'s test suite as a
gate, builds `crates/wasm-app` to `frontend/pkg`, and publishes
`frontend/` to Pages.

It triggers on push to `dev`, `main`, or `master`, plus manual dispatch
from the Actions tab. One manual one-time step is required and can't be
done through the GitHub API this was built with: repo **Settings → Pages
→ Build and deployment → Source → "GitHub Actions"**. After that, every
push to one of those branches deploys automatically to
`https://<owner>.github.io/<repo>/`.

## Build locally

You'll need:

```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Then, from the repo root:

```
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

`+simd128` is what lets the compiler vectorize `llm-core`'s inner loops
into real wasm SIMD; it makes a large difference to CPU training speed,
which is the default path everywhere and the only path in a browser
without WebGPU. It's what the deploy workflow builds with. (Dropping the
flag still builds and runs, just slower; WebAssembly SIMD has been
baseline in Chrome/Edge/Firefox/Safari since 2023.)

Open `http://localhost:8000` in a recent Chrome or Edge (WebGPU support;
both generation and training fall back to CPU-only in browsers without it,
or for model shapes too large for the GPU backend — see "Where training
happens vs. where WebGPU is used" above). It needs to be served over
HTTP(S) (or localhost), not opened as a `file://` URL — module workers and
`fetch` won't work otherwise.

## What's tested and what isn't

This was built in a sandboxed environment with **no network access to any
package registry** — crates.io, npmjs.org, and `static.rust-lang.org`
(needed for `rustup target add wasm32-unknown-unknown`) all returned 403.
That means:

- **`llm-core` (tokenizer, text prep, corpus, model, training, generation)
  is real, verified work**: zero external dependencies (own tiny PRNG
  instead of pulling in `rand`, no `serde`), so it built and ran fully
  offline. Its 100 tests include full gradient checks (analytic vs.
  numerical differentiation) for every op — RMSNorm, the linear layers,
  RoPE, sliding-window attention, SwiGLU, cross-entropy — plus a full-model
  gradient check across embeddings, PLE tables, attention/MLP weights, and
  norm gains, plus an end-to-end "does Adam actually reduce loss"
  training test. Run it yourself: `cargo test -p llm-core` from the repo
  root, no network needed.
- **`llm-gpu` (the WGSL/wgpu backend) and `wasm-app` (the wasm-bindgen
  glue) could not be compiled, let alone run, in that sandbox** — no GPU,
  no `wasm32` target, no way to fetch `wgpu`/`wasm-bindgen`/etc. They were
  written as carefully as I could manage by hand (the GPU kernels — forward
  *and* backward, including the Adam update — are direct, commented
  translations of the already-verified CPU ops in `llm-core/src/ops.rs`,
  cross-checked line-by-line against `llm-core::model::backward`). Every
  backward kernel is written as a *gather*, never a *scatter* (see
  `llm-gpu/src/model.rs`'s module docs), specifically so nothing needs
  atomic float adds — the highest-bug-risk shortcut this crate deliberately
  avoids. The GitHub Actions deploy workflow *does* build both crates
  (that's the point of it — a runner with normal internet access) and that
  build has succeeded, so the code is at least known to compile cleanly
  against real dependencies; what's still unverified is *runtime*
  correctness (does the WGSL actually compute the right numbers, does it
  run at all on a given GPU/driver) — that needs an actual browser, which
  is what `debug_compare_gpu_cpu` and `debug_compare_gpu_cpu_gradient`
  below are for.
- **The frontend JS** (`app.js`, `worker.js`, `db.js`) was syntax-checked
  with Node and carefully reviewed, but never run in an actual browser.

**First thing to do after building**: open the browser console, create a
small model, and click "Compare GPU vs CPU (debug)" in the Generate panel
and "Compare GPU vs CPU gradient (debug)" in the Train panel (both appear
once a WebGPU device initializes). The first calls `debug_compare_gpu_cpu`,
comparing one forward pass's logits between backends; the second calls
`debug_compare_gpu_cpu_gradient`, which runs forward + cross-entropy +
backward on both backends (without touching any real weights) and compares
their embedding-table gradients — since that gradient depends on nearly the
entire backward pass (every layer's attention/MLP backward, every layer's
PLE scatter, and the input embedding scatter all feed into it), a small
diff there is a strong end-to-end signal the WGSL backward kernels are
correct. Both should report a tiny difference (well under `1e-2`, just
float rounding). If either doesn't, or if the WebGPU path doesn't work at
all, that's expected risk materializing, not a mystery: the bug is almost
certainly in `crates/llm-gpu` (compare its WGSL/Rust against the matching
function in `llm-core/src/ops.rs` or `model.rs`, which you can trust).
Please report back what you find — a follow-up session with real
WebGPU/wasm32 access should be able to fix it quickly once the actual
failure is known.

## Using the app

1. **Add sources** — paste text, upload `.txt`/`.md`/`.fountain`/etc.
   files, or fetch a URL (see the CORS caveat below). Each source is
   stored in IndexedDB and can be edited or deleted later; edits and
   deletes immediately update the live training corpus if a model exists.
2. **Pick a model shape** — layers, nodes (hidden size), attention heads,
   context length, attention window. A live parameter-count/memory
   estimate updates as you adjust these. Click "Create model".
3. **Train** — pick a batch size and learning rate, optionally check
   "Train on WebGPU" (falls back to CPU if the model's window is too large
   for the GPU backend or no WebGPU device is available), click "Start
   training". Loss is plotted live; training runs in a background worker
   so the UI stays responsive.
4. **Generate** — type a prompt, optionally enable WebGPU acceleration,
   optionally tag it with a genre/tone and/or a target word count, and
   optionally have the model reminded of characters/locations so far or
   given similar scenes from your sources as context (see "Story-aware
   features" below). Click "Generate"; a QA pass runs automatically and
   any notes appear under the output.
5. **Save/load** — checkpoint weights to IndexedDB (with a name, for
   later reloading), or download/import raw weight bytes as a file.

### Story-aware features (non-neural scaffolding)

A few features sit *around* the model rather than inside it — plain
heuristics and orchestration, not anything learned or trained. They're
cheap, don't need a bigger model, and were prioritized over deeper
architecture changes precisely because they're low-risk:

- **Story state** (`crates/llm-core/src/screenplay.rs`): a plain
  line-shape scan (no ML) over your sources for scene headings
  (`INT./EXT. ...`) and ALL-CAPS character cues, aggregated into a running
  character/location list shown in the Sources panel. This is genuinely
  "free" auto-tagging — it's what answers "who are the characters in my
  corpus" without training a classifier for it. It's a heuristic, not a
  parser: unusual formatting can fool it (that's noted in the UI too).
- **Control tags**: optional genre/tone text prepended to a source's text
  (`[GENRE: sci-fi] [TONE: dark]`) or to a generation prompt. For a tag to
  actually influence training rather than getting buried mid-window,
  `Corpus::sample_batch` deliberately samples 40% of training windows
  starting exactly at a source boundary (see its doc comment) — a
  preamble only teaches the model anything if the model consistently sees
  it positioned at the start.
- **Retrieval** (`crates/llm-core/src/retrieval.rs`): TF-IDF + cosine
  similarity over your sources' scenes, no embedding model needed at this
  scale. "Use similar scenes from your sources as context" in the
  Generate panel prepends the best-matching scenes to the prompt; "Preview
  retrieved context" shows what would be retrieved without generating.
- **QA notes** (`crates/llm-core/src/qa.rs`): a rule-based pass over
  freshly generated text — unbalanced parentheses, degenerate
  line-repetition loops (a common small-model failure mode), characters
  introduced that don't appear in any source yet, and a rough
  length-vs-target check. Not a quality judgement, just things worth a
  glance.

These are all covered by `llm-core`'s offline test suite like everything
else in that crate.

### Known limitations

- **URL fetching is subject to CORS** — most sites don't send the headers
  that allow a browser page to `fetch()` them cross-origin. When it fails,
  copy the text and use "Paste text" instead; this isn't a bug in the app.
- **Only plain-text file formats** are supported for upload (`.txt`,
  `.md`, `.fountain`, ...) — not `.pdf`/`.docx`, which need a real parser
  this project doesn't include.
- **No KV cache**: generation re-runs the full forward pass over the
  current context window for every new token, so it's O(n²) in the number
  of generated tokens. Fine at "simple hobbyist model" scale; a real
  optimization opportunity if you extend this.
- **Byte-level tokenization** means a generation window boundary can
  occasionally cut a multi-byte UTF-8 character in half, showing up as a
  stray `�` in the output — a known, minor artifact of this tokenizer
  choice, not a crash.
- This is a **from-scratch, small model** trained only on what you feed
  it — expect it to pick up local style and formatting quickly, not to
  write a coherent multi-page story.

### Possible extensions

- Grouped-query attention (fewer KV heads than Q heads) to shrink the
  KV-cache once one exists.
- A KV cache for generation, to drop the O(n²) re-forward cost.
- Raising `MAX_GPU_WINDOW` (currently 256) once someone can verify a larger
  dense per-layer attention-probs cache still fits comfortably in GPU
  memory for the context lengths people actually want to train.
- A real BPE tokenizer if you want a larger effective context per
  character (trades away the byte-level tokenizer's simplicity and small,
  fixed vocab size).
