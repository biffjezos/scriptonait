# scriptonait

A small transformer that writes screenplays, novels and philosophical
allegories from a prompt, running in your browser. Written in Rust,
compiled to WebAssembly ahead of time, pretrained on public-domain text.

Type this:

> Write a 700 word novel about two people in space related to Plato's
> allegory of the cave

and it writes it — reading the prompt as an instruction (form, length,
subject, what to echo) rather than as text to continue.

## What it actually is, and isn't

It's a few-million-parameter model trained for hours on a CI runner over
a few tens of MB of public-domain books and plays. Expect it to hold a
voice, honour the form you asked for, keep screenplay formatting
plausible, and land near the length you wanted. Do not expect a coherent
700-word story with a real arc, characters who stay themselves, or an
ending that answers its beginning. Nothing trainable inside a GitHub
Actions budget does that, and no amount of frontend polish changes it.

What it's good for: style, structure, format, and having something on the
page to push against.

## How it's put together

The short version: **training happens on a server, generation happens in
your browser, and nothing is compiled while you wait.**

```
crates/
  llm-core/    The whole engine. Tokenizer, text prep, corpus, model
               (forward, backward, AdamW), generation, instruction
               parsing, checkpoints. Zero dependencies — 150+ tests,
               including analytic-vs-numerical gradient checks for every
               op, run with no network access.
  llm-data/    Cleans downloaded public-domain text, learns the BPE
               vocabulary, and writes a pre-tokenized dataset.
  llm-train/   Native pretraining: threaded, time-budgeted, resumable,
               and loud about its progress.
  llm-bench/   Throughput numbers, so performance claims are reproducible.
  wasm-app/    wasm-bindgen glue: one WasmLLM class over llm-core.
frontend/
  index.html, style.css, app.js   The page.
  worker.js                       Owns the wasm module, off the main thread.
  db.js                           IndexedDB, for your own sources.
  pkg/                            wasm-pack output — built by CI, not in git.
  model/                          The shipped checkpoint — fetched by CI
                                  from the latest release, not in git.
corpus/
  manifest.tsv, fetch.sh          What the model is trained on.
```

### The model

A Llama-style decoder-only transformer: RMSNorm, rotary position
embeddings, SwiGLU MLP, weight-tied input/output embedding, no biases.
On top of that:

- **Byte-level BPE tokenizer.** The base alphabet is all 256 byte values,
  so anything encodes and there is no unknown token; learned merges sit
  on top. A 700-word story is ~900 tokens instead of ~4,000, which is a
  4x saving on generation time, on training time, and on how much story
  fits in the attention window, all at once. The pre-tokenizer keeps
  whitespace runs as their own chunk on purpose — in a plain-text
  screenplay the indent is the only thing separating a character cue from
  an action line, so `"\n\n     "` becoming one token is the model
  learning what a character cue looks like.
- **Grouped-query attention.** Several query heads share one key/value
  head, which shrinks Wk/Wv and — the part that matters — shrinks the KV
  cache per generated token, which is what bounds how long a generation
  can run.
- **Sliding-window attention.** The window is configurable separately
  from context length, and attention probabilities are stored *banded*,
  so both time and memory are `context * window` rather than `context²`.
- **A KV cache.** The prompt is processed once; each new token costs one
  row plus attention over the window. Generating a 900-token story used
  to cost ~460,000 token-forwards and now costs ~1,400.
- **Per-layer embeddings** (Gemma 3n's PLE) are implemented and off by
  default. At a byte vocabulary they were free; at an 8k BPE vocabulary
  each table is the size of the whole input embedding, so a 6-layer model
  would spend more parameters on them than on all its attention and MLP
  combined.

### The instruction format

A plain language model continues text. Given an instruction it continues
*the instruction*. So the model is trained on a format instead:

```
BOS TASK form=novel; words=medium; about: two people in space;
         echoing: Plato's allegory of the cave STORY <the text> EOS
```

`TASK` and `STORY` are single tokens, so the boundary between the ask and
the answer costs two tokens rather than a paragraph of scaffolding. One
function renders that line, called by both the training path and the
generation path — any divergence between them would be a silent quality
bug.

Nobody hand-writes training examples. `llm_core::instruct::synthesize_examples`
cuts a real book or play at paragraph boundaries and pairs each chunk with
the instruction that would have asked for it: the form its own shape says
it is, its own length bucket, and a subject drawn from its most
distinctive words.

Lengths are **buckets**, not numbers. The model can't count, so training
it against "700" teaches a number it can't honour. The counting is done by
the code, which can: generation runs to your target and then to the next
sentence boundary, so it ends on a finished sentence, with a hard ceiling
40% over in case the model never produces one.

### Where training happens

Pretraining runs natively on a GitHub Actions runner
(`.github/workflows/pretrain.yml` → `crates/llm-train`). A browser tab
gets one thread — wasm threads need `SharedArrayBuffer`, which needs
cross-origin isolation headers, which GitHub Pages does not serve — and
it competes with the compositor for the machine, and it loses everything
when you close it. A runner has four cores, native optimization flags,
and six hours.

Six hours is also the cap, and useful training is longer than that, so a
run is one *round*: resume from the published checkpoint, train for the
budget, publish, and optionally dispatch the next round. The model
improves run over run.

Fine-tuning in the browser is still there, demoted to an optional panel,
running on a duty cycle you choose so it can't take the machine.

### Nothing is compiled in the browser

The `.wasm` file is built by `wasm-pack` in CI and served as a compiled
artifact; the page fetches and instantiates it. There is no toolchain on
the page and no client-side compilation step. `frontend/pkg/` is
generated in CI and deliberately not in git.

The model is likewise not built on demand: it's a release asset the
deploy workflow packages with the site.

## The corpus

`corpus/manifest.tsv` lists the works, one per line, with the form each is
labelled as. `corpus/fetch.sh` downloads them and **verifies each against
its expected title** — a wrong Project Gutenberg id doesn't 404, it
quietly returns a different book, so the check is what keeps a typo cheap.
Gutenberg's ~500-line licence wrapper is stripped before training; left
in, the model would see it dozens of times and learn it better than any of
the prose.

**On film scripts.** Film scripts are almost never public domain. What
stands in for them is public-domain *drama* — stage plays, in the same
character-cue-and-dialogue shape a screenplay uses. That teaches the
structure without teaching film formatting, and the README should say so
rather than let the `form=screenplay` label imply otherwise. To train on
real film scripts, add them as sources in the app and fine-tune; that's
what that path is for. Adding more public-domain works is one line in the
manifest.

## Running it

### The deployed site

`.github/workflows/deploy.yml` builds and publishes on every push to
`dev`/`main`/`master`. One manual step is required once: repo **Settings →
Pages → Build and deployment → Source → "GitHub Actions"**.

### Training a model

From the Actions tab, run **Pretrain the shipped model**. `minutes` is the
budget for one round; `rounds` chains several. The first run starts from
scratch; later runs resume. Each round publishes to the `model-latest`
release and redeploys the site.

### Locally

```
cargo test                      # the engine, no network needed
cargo run --release -p llm-bench   # throughput numbers

corpus/fetch.sh
cargo run --release -p llm-data  -- --raw corpus/raw --out corpus/build
cargo run --release -p llm-train -- --data corpus/build \
    --out model/scriptonait.ckpt --web-out frontend/model/scriptonait.ckpt \
    --minutes 30

rustup target add wasm32-unknown-unknown
cargo install wasm-pack
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg
cd frontend && python3 -m http.server 8000
```

`+simd128` is what lets the compiler vectorize the inner loops into real
wasm SIMD; it makes a large difference to how fast text appears. Serve
over HTTP — module workers and `fetch` don't work from a `file://` URL.

## Performance

Measured by `cargo run --release -p llm-bench`, at a ~5M-parameter shape
with 512-token sequences, on four cores:

| | before | after |
|---|---|---|
| training | 94 tok/s | 892 tok/s |
| generation | 12.7 tok/s | 373 tok/s |

Training: the workspace release profile had been set to `opt-level = "z"`
— a size setting meant for the wasm bundle, applied to native builds
nobody ships (2.4x); blocked matmuls, which matter most for the tied
output head at a BPE-sized vocabulary (1.1x); and splitting a batch across
threads (3.6x).

Generation: almost entirely the KV cache. Then the tokenizer change puts
roughly 4x more *text* behind each of those tokens.

## What's tested, and what isn't

`llm-core` has zero dependencies and 150+ tests that run with no network
access, including analytic-vs-numerical gradient checks for every op and
for the full model, an end-to-end "does AdamW reduce loss" test, and an
equivalence test between KV-cached decoding and a full forward pass.
(That last one earned its place immediately: it caught the cache attending
over one key too many.) Run them with `cargo test`.

`wasm-app` cannot be compiled in this project's development sandbox at all
— no wasm32 target, no route to crates.io — so the Actions build is its
only compiler, which is why every push runs it. The frontend JavaScript is
syntax-checked as ES modules and reviewed, but has not been run in a
browser here.

There used to be a WebGPU backend (`crates/llm-gpu`). It has been removed:
it was never verified to compute the right numbers, it couldn't express
grouped-query attention or PLE-off models without a rewrite that also
couldn't be verified here, and its CPU/GPU toggle kept two independent
sets of weights and optimizer state — flipping it mid-run silently reset
Adam's momentum. With the KV cache and a BPE vocabulary the CPU path is
fast enough at this model size. It's in the git history if someone with a
real WebGPU setup wants to bring it back.

## Known limitations

- **URL fetching is subject to CORS.** Most sites don't allow it. Paste
  the text instead; that isn't a bug in this page.
- **Plain text only** for uploads — no PDF or DOCX.
- **Byte-level BPE** means a generation cut short can end mid-character.
  The streaming path only ever emits complete characters, so this shows up
  as a slightly short ending rather than a `�`.
- **The model is small.** See the top of this file.
