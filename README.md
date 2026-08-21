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
  thousands of tokens.
- **Text prep**: HTML is stripped for URL sources; all sources get
  whitespace-normalized, but — deliberately — *leading indentation is
  preserved*. Plain-text screenplay exports commonly use indentation as
  the only signal separating scene headings, character cues, dialogue, and
  action lines, so a naive "trim every line" pass would destroy exactly
  the structure worth learning.

See doc comments in `crates/llm-core/src/config.rs` and `model.rs` for the
full per-layer layout and parameter-count formula (also surfaced live in
the UI as you adjust settings).

### Where training happens vs. where WebGPU is used

Training (forward + backward + Adam) runs on `llm-core`'s CPU
implementation, compiled to wasm — still entirely client-side, just not
GPU-accelerated. **Generation** runs on WebGPU by default (`llm-gpu`),
with automatic fallback to the same CPU code if WebGPU isn't available.

This split is deliberate, not a shortcut — see "What's tested and what
isn't" below.

## Project layout

```
crates/
  llm-core/   Tokenizer, text prep, corpus/batch sampling, model
              (forward+backward+Adam), generation. Zero external
              dependencies — builds and its 56 tests run with no network
              access. This is the verified reference implementation.
  llm-gpu/    WebGPU (wgpu + WGSL) forward-pass backend, mirroring
              llm-core's forward pass kernel-for-kernel. Forward only.
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

## Build

You'll need:

```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Then, from the repo root:

```
wasm-pack build crates/wasm-app --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

Open `http://localhost:8000` in a recent Chrome or Edge (WebGPU support;
generation falls back to CPU-only in browsers without it — training works
everywhere since it's CPU/wasm regardless). It needs to be served over
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
  offline. Its 56 tests include full gradient checks (analytic vs.
  numerical differentiation) for every op — RMSNorm, the linear layers,
  RoPE, sliding-window attention, SwiGLU, cross-entropy — plus a full-model
  gradient check across embeddings, PLE tables, attention/MLP weights, and
  norm gains, plus an end-to-end "does Adam actually reduce loss"
  training test. Run it yourself: `cargo test -p llm-core` from the repo
  root, no network needed.
- **`llm-gpu` (the WGSL/wgpu backend) and `wasm-app` (the wasm-bindgen
  glue) could not be compiled, let alone run, in that sandbox** — no GPU,
  no `wasm32` target, no way to fetch `wgpu`/`wasm-bindgen`/etc. They were
  written as carefully as I could manage by hand (the GPU kernels are
  direct, commented translations of the already-verified CPU ops in
  `llm-core/src/ops.rs`), but they are genuinely unverified. Training was
  deliberately kept off the GPU path specifically to avoid writing
  backward-pass/gradient-accumulation shaders (the highest-bug-risk code,
  especially anything needing atomic float adds) with zero ability to test
  them.
- **The frontend JS** (`app.js`, `worker.js`, `db.js`) was syntax-checked
  with Node and carefully reviewed, but never run in an actual browser.

**First thing to do after building**: open the browser console, create a
small model, and click "Compare GPU vs CPU (debug)" in the Generate panel
(it appears once a WebGPU device initializes). That calls
`debug_compare_gpu_cpu`, which runs the same forward pass on both backends
and reports the largest logit difference — it should be tiny (well under
`1e-2`, just float rounding). If it's not, or if the WebGPU path doesn't
work at all, that's expected risk materializing, not a mystery: the bug is
almost certainly in `crates/llm-gpu` (compare its WGSL/Rust against the
matching function in `llm-core/src/ops.rs`, which you can trust). Please
report back what you find — a follow-up session with real WebGPU/wasm32
access should be able to fix it quickly once the actual failure is known.

## Using the app

1. **Add sources** — paste text, upload `.txt`/`.md`/`.fountain`/etc.
   files, or fetch a URL (see the CORS caveat below). Each source is
   stored in IndexedDB and can be edited or deleted later; edits and
   deletes immediately update the live training corpus if a model exists.
2. **Pick a model shape** — layers, nodes (hidden size), attention heads,
   context length, attention window. A live parameter-count/memory
   estimate updates as you adjust these. Click "Create model".
3. **Train** — pick a batch size and learning rate, click "Start
   training". Loss is plotted live; training runs in a background worker
   so the UI stays responsive.
4. **Generate** — type a prompt, optionally enable WebGPU acceleration,
   click "Generate".
5. **Save/load** — checkpoint weights to IndexedDB (with a name, for
   later reloading), or download/import raw weight bytes as a file.

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
- GPU backward-pass kernels, once someone can actually test them, so
  training itself could move to WebGPU too.
- A real BPE tokenizer if you want a larger effective context per
  character (trades away the byte-level tokenizer's simplicity and small,
  fixed vocab size).
