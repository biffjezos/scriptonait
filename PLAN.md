# Implementation plan: fast, pretrained, prompt-driven generation

## What's wrong today

Three complaints, one root cause between them.

1. **"The network is very slow. It even slows down my entire computer."**
   The app trains a transformer *from scratch, in a browser tab, on your
   CPU*, and nothing useful comes out until it has. A browser tab is the
   slowest place on the machine to run a training loop, and a training
   loop that saturates a core (or four) is exactly what makes the rest of
   the machine crawl. Even on the fast path the tokenizer is byte-level
   (one token per byte, so a 700-word story is ~4000 tokens instead of
   ~900) and generation has no KV cache (each new token re-runs the whole
   forward pass — O(n²)). So the two things a user actually waits on,
   training and generating, are both paying a large, avoidable multiple.

2. **"There are no notifications and I don't know what it's doing."**
   Long jobs report a loss number and nothing else: no throughput, no
   ETA, no phase, no OS-level notification when a job finishes. If it
   takes twenty minutes, you have to sit and watch it.

3. **"Remove the Train on CPU button and the toggles."**
   CPU-vs-WebGPU is not a decision a user should be asked to make. Worse,
   the two backends keep *separate weight copies and separate optimizer
   state*, so flipping the toggle mid-run silently resets Adam momentum.
   The toggle exposes an implementation detail whose only honest setting
   is "pick the fast one that works".

And the actual goal — "write a 700 word novel about two people in space
related to Plato's allegory of the cave" — needs three things the app
doesn't have: a model that has already read a lot of prose, an
understanding that the prompt is an *instruction* (form = novel, length =
700 words, subject = X related to Y) rather than text to continue, and
length control.

## The shape of the fix

**Move training off the user's machine.** Pretraining runs natively on a
GitHub Actions runner — real threads, real CPU flags, no browser — and the
finished tokenizer and weights are committed into the repo and served with
the page. The browser opens with a model that already works and only ever
runs inference. In-browser training stays, demoted to an optional,
throttled "fine-tune on your own text" path for people who want it.

**Precompile everything.** The wasm bundle is built by CI with
`wasm-pack` and committed to `frontend/pkg`, exactly like the weights.
Nothing is compiled in the browser; the page fetches a `.wasm` file and
instantiates it.

**Then make the model itself worth running:** BPE tokenizer, grouped-query
attention, KV-cached generation, AdamW with warmup/cosine decay and
gradient clipping, and an instruction format the prompt parser targets.

### Honest ceiling

This stays a from-scratch model trained for hours, not GPU-months, on a
few tens of MB of public-domain text. Expect locally fluent, stylistically
on-genre prose that holds a scene and honours the requested form and
length. Do not expect a coherent 700-word story with a real arc; nothing
trainable on a CI runner does that. Every step below is aimed at getting
as far up that curve as the budget allows, and the README will say the
same thing in the same words.

## Steps

Each step compiles, keeps `cargo test -p llm-core` green, and is pushed on
its own. CI (`.github/workflows/deploy.yml`) is the compile gate for the
crates this sandbox can't build (`llm-gpu`, `wasm-app` — no wasm32 target,
no crates.io route here), so every push is checked against a green run
before the next step starts.

1. **PLAN.md** — this file.
2. **Core speed.** Register-tiled matmuls in `ops::linear_fwd`/`linear_bwd`
   (they dominate every forward and backward pass), tighter attention
   loops, and a `llm-bench` binary that reports tokens/s so the gain is a
   measured number and not a claim.
3. **BPE tokenizer.** Trainable byte-level BPE with serialization and a
   byte fallback (so it still never emits an unknown token). `vocab_size`
   moves from a constant into `ModelConfig`. ~4x fewer tokens per word:
   4x less compute per word of output, 4x more text per context window.
4. **Architecture.** Grouped-query attention, per-layer embeddings made
   optional (they triple parameter count once the vocab is 8k), configurable
   RoPE theta, AdamW + gradient clipping + warmup/cosine LR schedule, a KV
   cache for generation, and top-k/top-p/repetition-penalty sampling.
5. **Instructions.** A `<|task|>…<|story|>…` training format, a parser that
   turns "Write a 700 word novel about two people in space related to
   Plato's allegory of the cave" into form/length/subject fields, dataset
   synthesis that builds matching instruction examples out of the corpus,
   and length-aware stopping.
6. **Native CLIs.** `llm-data` (clean and chunk downloaded public-domain
   text, build the instruction dataset, train the BPE vocab) and
   `llm-train` (multithreaded data-parallel training, resumable
   checkpoints, progress/throughput/ETA logging).
7. **Pretraining in CI.** A workflow that fetches public-domain film
   scripts, novels, and philosophical allegories (Plato's cave included),
   trains for the runner's budget, commits the checkpoint, and re-dispatches
   itself to continue — so the shipped model improves run over run.
8. **GPU/wasm layer.** `llm-gpu` drops its backward and Adam kernels and
   becomes a GQA-aware inference backend, selected automatically. `wasm-app`
   exposes the new API and loads the shipped checkpoint.
9. **Frontend.** No backend toggles. Opens with the pretrained model. Live
   status: phase, tokens/s, ETA, progress. OS notifications when a long job
   finishes. The optional fine-tune path yields between steps against a
   configurable time budget, so it can't monopolise the machine.
10. **README + final CI verification.**
