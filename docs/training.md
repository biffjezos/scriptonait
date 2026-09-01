# Training (Training tab, Settings tab)

## Train / Stop

- **Train** — disabled with no model shape priced yet or too little
  corpus text; starts (or resumes) a run against the current model,
  corpus, and settings.
- **Stop** — ends the run after its current step; progress, the
  optimizer state, and the schedule's position are all kept, so
  pressing Train again continues rather than restarting.
- A model too big for this machine's GPU, or a browser with no WebGPU
  at all, disables Train with the reason stated inline.

## Where it runs

Training runs on the GPU only — forward pass, cross-entropy, backward
pass, and the optimizer step are WGSL compute kernels; there is no CPU
training path. Without WebGPU, the page says so and refuses to train.
See [architecture.md](architecture.md).

## Optimizer

- **AdamW**: Adam with decoupled weight decay.
  Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2019
  (arXiv:1711.05101).
- **Global gradient-norm clipping**, default max norm 1.0.
- Learning-rate schedule: see [scheduler.md](scheduler.md).
- Optimizer moment buffers are checkpointed with the model, so a
  resumed run continues with Adam's momentum intact rather than
  restarting it from zero.

## Training Mode (Settings tab)

- **Auto** — reads batch size and effort off this machine's own
  benchmark (below); resets both to their measured defaults whenever
  switched on.
- **Manual** exposes:
  - **Effort** — fraction of wall-clock time a training step is allowed
    to occupy the GPU for, before yielding it back: Gentle (0.25),
    Balanced (0.5), Full speed (1), or Auto (full speed, but every step
    stays interruptible).
  - **Batch size** — sequences per training step. 0 defers to the
    machine benchmark. Shown alongside the resulting tokens-per-step
    (batch × context length), marked "measured" or "default" depending
    on whether a benchmark for this machine exists.
  - **Learning rate** — overrides the scheduler's peak rate entirely.

## Machine Benchmark (Settings tab)

- Runs once automatically before the first training step on a given
  adapter, unless **Auto-benchmark** is turned off.
- Measures how much work fits in one GPU command buffer before the
  driver's watchdog would reclaim the device, and how large a batch can
  be while keeping a step interruptible.
- Stored per GPU adapter in IndexedDB and reused on later visits — a
  machine is measured once, not guessed at or hard-coded.
- Turning Auto-benchmark off stops *re*-measuring automatically; it does
  not discard an existing measurement, which keeps being used for
  batch-size defaults.
- Console-only: `scriptonait.benchmark()` re-runs the sweep and logs
  every candidate timed; `scriptonait.machine()` returns what's stored.

## Training Plan (Settings tab, live during a run)

| Setting | Meaning |
|---|---|
| Planned steps | Total steps this run's schedule is shaped against |
| Metrics every | Steps between held-out loss measurements |
| Source-opening windows | % of training windows drawn from a source's opening (default 40%) rather than a random offset, so the model sees openings often enough to learn them |
| Show training window | Whether the actual text of the current training batch is displayed live |
| Max characters | Truncation length for the displayed training window |
| Sample every | Steps between generating a training sample (checkbox to enable/disable) |

## Progress (Training tab)

- **Training plan**: which phase the run is in (warm-up, learning,
  plateau, overfitting, cooling down), what that phase implies, tokens
  seen, corpus pass count, an estimate of what's left, and a
  ranked list of what would actually help next — each tied to a number,
  not just a suggestion.
- **Loss chart**: three curves on one axis — per-step training loss,
  held-out loss, and training loss measured on the same fixed kind of
  window the held-out set uses. The third curve exists because the
  first two aren't directly comparable (40% of training windows start
  at a source's opening; no held-out window ever does) — the gap worth
  reading is between the third curve and the held-out one.

## Metrics (Training tab)

- Run history: one row per measurement, with the settings that produced
  it, kept across runs and reloads. Training events (a run starting, a
  rate cut, a phase change) sit on the same timeline.
- **Copy as Markdown** / **Copy as JSON** / **Clear History**.

## Samples (Training tab)

- One card, updated in place as training samples are generated, with
  every past sample kept — **Earlier** / **Later** (or a typed index)
  page back through them, each labeled with the step it was taken at.
- Each sample is scored: **bits per byte** (held-out loss in a form
  comparable across vocabularies, and against gzip's ≈2.5 on English
  prose), the fraction of its words that appear anywhere in the corpus,
  how many of its four-word runs are repeats of ones it already wrote,
  and how many of its words are distinct.

## Remote training (Settings tab → Training → Location)

- **Local** (default) trains in this browser tab, on this machine's GPU.
- **Remote** sends training to `llm-server`, a native binary the user
  runs on their own GPU machine (`crates/llm-server`) — job-based, not
  per-step: the browser uploads a checkpoint snapshot and the corpus
  once, the server runs its own training loop, and progress streams
  back over Server-Sent Events. Only training is ever remote; generation
  always runs locally, in this browser.
- **Remote server URL** / **Remote server token** / **Test Connection**
  configure and check the connection. The server defaults to port 8420
  and requires a bearer token (`--token`/`LLM_SERVER_TOKEN`) unless
  explicitly run without one.
- `llm-server`'s own schedule is a simpler flat-then-cosine curve with
  no held-out-loss/plateau detection and no autosave-to-file — those
  stay client-side concerns; the browser still periodically pulls a
  checkpoint back to keep a local copy current. Branching a remote run
  is not supported.

## Console

- Per-step logging: loss, held-out loss, gradient norm, learning rate,
  tokens/second, dispatch and submission counts.
- `scriptonait.profile()` — times one step per phase (zero, forward,
  loss, backward, reduce, readback, AdamW) at four command-buffer sizes.
- `scriptonait.kernels()` — times each kernel at the current model's
  shapes, logged sorted by cost.
- `scriptonait.evaluate(text, loss)` — scores arbitrary text the same
  way a training sample is scored.
