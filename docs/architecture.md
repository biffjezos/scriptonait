# Backend architecture (brief)

## Crate layout

```
crates/
  llm-core/    Tokenizer, corpus, the model (forward, backward, AdamW)
               as the CPU reference the GPU kernels mirror, generation,
               instruction parsing, checkpoints. No external dependencies;
               builds and tests offline.
  llm-gpu/     The GPU backend: training kernels and generation, in WGSL.
  llm-server/  Native binary: a companion GPU training server for the
               "Remote" training backend (see training.md).
  shader-check/ Compiles every WGSL shader with naga, as a build check.
  wasm-app/    wasm-bindgen glue: one WasmLLM class wrapping both
               llm-core and llm-gpu for the browser.
frontend/
  index.html, style.css, app.js   The page.
  worker.js                       Owns the wasm module, off the main thread.
  db.js                           IndexedDB: sources, models, the model
                                   library, machine benchmarks, settings,
                                   run history.
  project.js                      The whole-project (.snp) file format.
```

`llm-core` is a normal Cargo workspace member and builds without network
access. `llm-gpu`, `wasm-app`, and `llm-server` each declare their own
`[workspace]` (need `wgpu`/`wasm-bindgen`/a real GPU adapter from
crates.io, unavailable in every offline sandbox this repo is developed
in) and are compiled and tested only in CI.

## Correctness

- `llm-core` has 185+ tests with no network dependency, including
  analytic-vs-numerical gradient checks per operation and for the full
  model, and an equivalence test between KV-cached decoding and a full
  forward pass.
- The GPU kernels are written to mirror named `llm-core` functions
  index-for-index. `WasmLLM::debug_compare_forward` runs the same
  tokens through the GPU forward pass and the CPU reference from the
  same weights and reports the largest logit difference;
  `GpuModel::debug_compare_step` does the same for one decode step —
  the CPU implementation is the oracle a GPU kernel is checked against,
  not the other way around.

## Concurrency

- One Web Worker owns the wasm module and everything GPU-related;
  `app.js` (the main thread) never touches it directly.
- Training and CPU-preferred inference share that one worker but don't
  block each other: CPU generation deliberately skips the "GPU busy"
  guard and yields to the event loop via a macrotask after every token,
  which is what lets an in-flight GPU training step's buffer readback
  run in between. GPU-preferred generation does take the same guard
  training uses, and is refused once (not queued) if the GPU is busy.

## Checkpoint format

- Custom binary format (no external serialization dependency): magic
  bytes, a version number, model shape, then weights — additive-only
  versioning, so an older checkpoint still loads with missing fields
  defaulted rather than rejected. Current version: 4.
- The whole-project `.snp` format (see [project.md](project.md)) wraps
  a checkpoint and an optimizer-state blob alongside a JSON header for
  corpus/history/settings, using the same hand-rolled framing.

## Build and run

```
cargo test                                          # llm-core only

rustup target add wasm32-unknown-unknown
cargo install wasm-pack
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

Serve over HTTP — module workers and `fetch` don't work from `file://`.

`.github/workflows/deploy.yml` runs the tests, compiles the shaders,
builds the wasm module, and publishes to GitHub Pages on push to
`dev`/`main`/`master`; every push (any branch) still runs the build-and-
test job. It builds the site — it does not train or ship a model.
Nothing is fetched or compiled inside the browser.

## Constraints

- Uploads are plain text only — no PDF/DOCX parsing.
- `llm-gpu`, `wasm-app`, and `llm-server` are compiled, shader-checked,
  and (for `llm-server`) run only in CI or on a real machine with a GPU
  and network access to crates.io — not in every development sandbox.
