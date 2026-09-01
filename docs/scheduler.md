# Learning rate scheduler (Settings tab → Scheduler)

The schedule is decomposed into five independent axes rather than one
named preset, so each part of "how the learning rate moves" can be
reasoned about (and tested) on its own.

**Scheduler mode**: Auto uses this app's chosen defaults (Plan-based
warm-up, Flat stable phase, Deferred/WSD cool-down, Fixed decay start,
Fixed plan length) and disables the axis controls below. Manual exposes
all five.

| Axis | Options | Meaning |
|---|---|---|
| Warm-up length | Plan-based (2% of steps) / Fixed-length (~40 steps) | How many steps ramp from 0 to the peak rate. Fixed-length is sized from AdamW's own β₂ time constant, not from the run length. |
| Stable phase | Flat / Reactive rate cuts | Whether the rate can be cut mid-run on a detected plateau (see below), independent of the cool-down curve. |
| Cool-down timing | Deferred (WSD) / Immediate (Cosine) | *When* decay starts. Deferred holds the peak rate flat until late in the run, then decays — Warmup-Stable-Decay, as introduced in Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, 2024 (arXiv:2404.06395). Immediate decays continuously from the end of warm-up — the classic cosine schedule, Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2017 (arXiv:1608.03983), used here without the restarts. |
| Decay start | Fixed fraction (80% of plan) / Adaptive | Only meaningful under a Deferred cool-down. Adaptive pins the decay window to start the first time held-out loss stops clearing its own noise floor, instead of waiting for 80% of the plan regardless of progress. |
| Plan length | Fixed / Adaptive — extends while still improving | Adaptive stretches the plan by a fifth (capped at five extensions) each time the run reaches its planned end while still genuinely improving, instead of parking at the floor rate with headroom left. |

## Compatibility constraints

- **Reactive rate cuts** forces **Cool-down timing** to Immediate and
  **Plan length** to Fixed, and disables both controls. Reactive cuts
  combined with a deferred cool-down or an adaptively-extended plan is a
  known bad interaction (compounding rate reductions); this app doesn't
  let the combination be selected.
- **Decay start** is only enabled when Cool-down timing is Deferred and
  Stable phase is not Reactive — it has no effect otherwise, and is
  disabled and reset to Fixed the rest of the time.

## Reactive rate cuts (plateau detection)

- Checked every 25 steps' worth of held-out measurement.
- Four consecutive measurements with no improvement beyond a noise
  threshold → the learning rate is halved (`plateau_scale ×= 0.5`).
- Won't cut below 5% of the pre-cut rate (`plateau_scale` floor of 0.05)
  — past that point the page says the limit is the corpus, not the
  rate, rather than cutting again.
- Independent of the schedule's own floor: a Deferred/Immediate cool-down
  already decays to 10% of the peak rate (`min_lr_ratio`) by the end of
  its own curve; plateau cuts multiply on top of that, not instead of it.

## Numbers this app picks for you

Two scheduler-adjacent decisions are never a raw setting anywhere,
Manual mode included — both re-evaluated live rather than fixed at
training start:

- **Auto mode's peak learning rate**: 6e-4 while the model is below its
  compute-optimal token budget (20 tokens per parameter — Hoffmann et
  al., *Training Compute-Optimal Large Language Models*, 2022,
  arXiv:2203.15556, the same ratio the training plan's corpus-size
  advice uses), 5e-5 once past it. Judged fresh against live tokens-seen
  on every settings change, not decided once at the start of a run.
- **How many passes over the corpus are worth planning for**: roughly
  four epochs' worth of tokens — repeated data holds up well to about
  four passes and adds little beyond sixteen. Muennighoff et al.,
  *Scaling Data-Constrained Language Models*, 2023 (arXiv:2305.16264).

## Live controls (Training tab, during a run)

- **Learning rate override** — type a rate and Apply; also switches this
  run out of Auto's own rate decision so it isn't silently recomputed on
  the next settings push.
- **Undo Plateau Cut** — reverts the most recent reactive-cuts halving.
