# scriptonait

Train a language model on your own writing, entirely in a browser tab,
on your own GPU. No account, no API key, no upload — the corpus, the
weights, and the trained model never leave the machine.

Requires WebGPU. There is no CPU training path: without a compatible
GPU/browser, the page says so and does not train.

## Contents

- [Corpus](docs/corpus.md) — adding text, the tokenizer, the held-out
  split, corpus stats.
- [Model](docs/model.md) — architecture (RoPE, GQA, SwiGLU, RMSNorm,
  sliding-window attention), the Model Shape settings, the instruction
  format.
- [Training](docs/training.md) — AdamW and gradient clipping, Training
  Mode, the machine benchmark, the Training Plan settings, Progress /
  Metrics / Samples, the remote training backend, console tools.
- [Scheduler](docs/scheduler.md) — the five-axis learning-rate schedule
  (warm-up, stable phase, cool-down timing, decay start, plan length),
  plateau detection, Auto mode's own decisions.
- [Generation](docs/generation.md) — the Inference tab, sampling
  settings, length control, decoding, the QA pass.
- [Project persistence](docs/project.md) — Save/Load, Export/Import
  Project, Branch, the model Library, Auto-save.
- [Backend architecture](docs/architecture.md) — crate layout,
  correctness (CPU reference vs. GPU kernels), concurrency, the
  checkpoint format, build and CI.

## Build and run

```
cargo test

rustup target add wasm32-unknown-unknown
cargo install wasm-pack
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

See [docs/architecture.md](docs/architecture.md) for what each build
step does and why.
