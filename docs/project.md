# Project persistence (Overview tab, Settings tab → Auto-save)

## Overview tab, at a glance

- Model status and details: parameters, layers, hidden size, heads (and
  KV heads), context and attention window, vocabulary size, training
  steps, and the device it trains/generates on.
- A one-line training-plan summary (phase and progress), mirroring the
  Training tab's own plan (see [training.md](training.md)) without
  switching tabs.
- Corpus summary: per-source train tokens, held-out tokens, and how
  many times each has been sampled.
- Guidance text for what to do next, before anything's been trained.

## Saving and loading

Four tiers, from lightest to heaviest:

| Action | Contains | Where |
|---|---|---|
| Save / Load Model | Checkpoint only: weights, tokenizer, model shape | Overview |
| Export Project / Import Project | Checkpoint + optimizer state + corpus + history + settings | Overview |
| Branch | Same whole-project bundle, written to a **new** file; the original project keeps training/autosaving undisturbed | Training tab |
| Library: Save / Switch / Delete | Same whole-project bundle, kept in this browser instead of a file | Overview |
| Auto-save | The whole-project bundle, written automatically | Settings |

## Save / Load Model

- **Save** exports just the trained checkpoint (`.ckpt`): weights,
  tokenizer, and model shape. No optimizer momentum, no corpus, no
  history, no settings.
- **Load Model** imports a `.ckpt` (or `.bin`) file as the current
  model, leaving the current corpus/history/settings as they are.

## Export Project / Import Project

- **Export Project** writes a `.snp` file: checkpoint, optimizer state,
  every corpus source, the full run history, and settings (auto-save
  config, device preference, benchmark, training plan, remote server
  config, inference options).
- **Import Project** replaces the live model, corpus, history, and
  settings with what's in a chosen `.snp` file, after confirmation.
- Custom container format (no zip dependency): magic bytes, a version,
  a JSON header (sources/history/settings, small enough as text), then
  the checkpoint and optimizer bytes back to back.

## New Project

- Clears the current model, corpus, and history, and forgets any
  connected auto-save file — a clean slate. Offers to pick a new
  auto-save file/folder first.

## Branch (Training tab)

- One click for: stop training if running, wait for it to actually
  stop, export the current whole-project bundle to a new file (or the
  browser's download), then resume the original run exactly where it
  left off. The original project's own auto-save target is never
  touched by a branch.
- This is what turns one seed model into several: train, Branch,
  keep training the original; open the branch later (or use the
  Library, below) to train it toward a different corpus. The same
  approach as training separate domain-specialist models from one
  starting point and keeping them all, rather than fine-tuning one
  model repeatedly and losing the earlier state — Li et al.,
  *Branch-Train-Merge: Embarrassingly Parallel Training of Expert
  Language Models*, 2022 (arXiv:2208.03306); Sukhbaatar et al.,
  *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM*,
  2024 (arXiv:2403.07816).
- Not available for a training run on the remote backend.

## Library (Overview tab)

- **Save to Library** — names and stores the current whole-project
  bundle in this browser (IndexedDB), without leaving the page or
  touching a file. Defaults to the project's own name plus the current
  step. Stops training first the same way Branch does, and resumes it
  after.
- **Switch** — loads a saved entry back as the live project (same
  effect, and the same confirmation, as Import Project).
- **Delete** — removes an entry permanently.
- What makes several trained models — one per specialist corpus — a
  click apart instead of a file picker apart.

## Auto-save (Settings tab)

- **On/off**, **frequency** (steps between writes), **mode**:
  - **Overwrite** — one browser-storage copy, replaced each time.
  - **Add** — a new file per save, into a chosen folder, named after
    the project and the step.
- **File name** / **Choose…** connects a real file (or, in Add mode, a
  folder) that auto-save writes into as well as the browser copy — the
  copy that survives cleared site storage or the page itself failing.
- Every auto-save (browser copy or file) writes the whole project, the
  same bundle Export Project does — a single file left behind after a
  crash has to be enough on its own to get back to where things were.
