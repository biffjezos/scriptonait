// Owns the wasm module and everything expensive.
//
// Two things happen off the main thread here, and they have opposite
// pacing needs.
//
// Generating text is what the user is waiting on, so it runs flat out and
// streams every piece back as it appears. Fine-tuning is not — it's a
// background job that can take minutes — so it runs on a duty cycle: work
// for a slice, then yield for a slice, at a ratio the user picks. That is
// the whole answer to "it slows down my entire computer". A worker thread
// that never yields still competes with the compositor, the network stack,
// and every other tab for a core; one that works 25% of the time simply
// takes four times as long and leaves the machine usable.
//
// Every long operation reports progress on a fixed interval. Not because
// progress bars are nice, but because a job that prints nothing is
// indistinguishable from a hung one, which is exactly the complaint this
// rewrite started from.

// The wasm glue is imported dynamically, inside `ensureWasm`, not with a
// static top-level `import`.
//
// A static import that fails - a missing or stale `pkg/`, a renamed
// export, a 404 on the .wasm - aborts the whole worker module before
// `self.onmessage` is ever assigned. The page then gets no reply, no
// error and no log for anything it asks: every call just times out after
// a minute. Loading it here instead turns that into an error message
// naming the file.
let wasm = null;
let llm = null;
let wasmReady = null;
/// True while a training run owns the GPU. Anything else that wants the
/// device waits for it - the wasm side refuses overlapping GPU work, and
/// finding that out as an error beats finding it out as a panic.
let training = false;

// Set by a `stop` message. Checked by the generation callback and between
// training steps — both cooperative, since neither can be interrupted
// mid-step.
let stopRequested = false;

/// Which device user-initiated generation prefers, set from the Settings
/// tab and applied to the next Generate call — never retried or polled
/// when it can't be honored (see `llm.generate`'s own `prefer_gpu`
/// argument). 'cpu' is the only choice that is guaranteed to work while
/// training holds the GPU; 'gpu' can be refused if training is running.
let inferenceDevice = 'gpu';

const PROGRESS_INTERVAL_MS = 250;

/// Bumped whenever the benchmark measures something different, so a
/// stored profile from an older build is re-measured instead of trusted.
const BENCH_VERSION = 1;

/// How many held-out measurements may pass with no new best before the
/// learning rate is cut. Four, at one measurement every twenty-five
/// steps, is a hundred steps of no progress — long enough not to react
/// to the noise in a single measurement, short enough not to spend an
/// afternoon on a rate that is too large.
const PLATEAU_PATIENCE = 4;
/// Tokens that must have gone through the model before the rate may be
/// cut for a plateau.
///
/// A cut at 0.8 passes is a cut made on a curve that is still falling
/// steeply, and halving the rate there costs the run real progress for
/// a plateau that was never there. The same gate the corpus advice
/// uses, for the same reason.
const TOKENS_BEFORE_CUTTING_THE_RATE = 2e6;

/// Default cadence for an unattended run to write itself down, used only
/// until the page sends its own Settings-tab frequency in the train
/// payload — a few minutes of work at risk instead of a night's.
const AUTOSAVE_EVERY_STEPS = 1000;
/// What a cut multiplies the rate by, and how far the cuts may go in
/// total. Halving is the standard move; a run that has fallen to a
/// twentieth of its schedule has a problem another cut will not fix.
const PLATEAU_FACTOR = 0.5;
const PLATEAU_FLOOR = 0.05;
/// The smallest movement that counts as movement at all, when there is
/// not enough curve yet to measure the noise from.
const PLATEAU_MIN_DELTA = 0.005;

/// How much longer an adaptively-extended plan grows each time — a
/// fifth more steps, the same fraction WSD_DECAY_FRACTION reserves for
/// decaying, so an extension gives roughly another decay window's worth
/// of room rather than an arbitrary bump.
const PLAN_EXTEND_FRACTION = 0.2;
/// How many times one run may extend itself. Every extension requires
/// fresh evidence of real improvement (see heldOutTrend), so this is a
/// backstop against a curve that keeps barely clearing the noise floor
/// forever, not the expected case.
const PLAN_EXTEND_MAX_TIMES = 5;

/// The smallest movement that counts as movement, as a fraction of the
/// loss level itself.
///
/// A curve that is smoothly, monotonically improving has almost no
/// spread around its own mean by construction — that is what "smooth"
/// means, not evidence there is nothing left to improve. A noise floor
/// built only from that spread shrinks fastest exactly when a run is
/// behaving best, and starts mistaking ordinary deceleration (a cosine
/// schedule slowing down is expected, not a plateau) for the real
/// thing. That is what a real run's own history showed: a noise floor
/// that had shrunk to 0.0036 called four still-falling measurements
/// "no improvement," cut the rate, which slowed the fall further,
/// shrank the floor further, and cut again — five cuts in 15,000 of a
/// 100,000-step plan, landing the schedule on its floor (5% of the
/// planned rate) before a fifth of the run had happened. This relative
/// floor is the fix: it can't shrink below a fixed fraction of the
/// loss's own current level, so a curve has to actually go flat at that
/// level, not just get smooth, before its own noise counts as evidence.
const PLATEAU_RELATIVE_FLOOR = 0.01;

/// How much the held-out curve moves on its own, measured from the
/// curve rather than assumed.
///
/// Everything that asks "has this stopped improving" has to ask it the
/// same way, or the page contradicts itself — which it did: the plateau
/// detector cut the learning rate in half at step 3,800 while the phase
/// beside it said "held-out loss still improving". Both were reading
/// the same numbers through different rules, and one of them was a
/// constant somebody picked.
function heldOutNoise(series, window) {
  const recent = series.slice(-window * 2);
  if (recent.length < 4) return PLATEAU_MIN_DELTA;
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const spread = Math.sqrt(
    recent.reduce((a, b) => a + (b - mean) ** 2, 0) / (recent.length - 1),
  );
  return Math.max(spread, PLATEAU_MIN_DELTA, mean * PLATEAU_RELATIVE_FLOOR);
}

/// The next step at which an `interval`-spaced cadence (metrics, samples,
/// autosave) is due, aligned to absolute step 0 rather than to whenever
/// this happened to be called from.
///
/// A run measuring every 500 steps that gets stopped and restarted at
/// step 34,334 should still land its next measurement at 34,500 — the
/// same grid an uninterrupted run would have used — not at 34,834
/// (34,334 + 500), which is what "call again in one interval" produces
/// instead of "call again on the grid this cadence has always been on."
/// Used both to seed a fresh run's first deadline and to reschedule the
/// next one after each firing, so a slice's overshoot past the exact
/// aligned step can never knock the grid itself out of alignment.
function nextOnGrid(step, interval) {
  if (!(interval > 0)) return Infinity;
  return (Math.floor(step / interval) + 1) * interval;
}

/// How many held-out windows the validation set holds, and how often it
/// is measured.
///
/// These trade against each other at fixed cost. Sixteen windows every
/// fifty steps is the same GPU time as six every twenty-five, and it is
/// a far better measurement: nearly three times the text, and — because
/// the set is fixed rather than resampled — no sampling term at all.
/// Two consecutive numbers now differ only because the weights differ,
/// which is the property every rule built on their difference assumed
/// and did not have.
const VALIDATION_WINDOWS = 16;
/// Every hundred steps rather than every fifty, because two fixed sets
/// are measured now instead of one and the pair has to cost what the
/// single noisy measurement did. At a five-thousand-step run that is
/// still fifty points on the curve.
///
/// The default, not the only value — the Settings tab's own "Metrics
/// every" field (see `train()`'s `metricsEvery` parameter) overrides it
/// per run; this is what a run falls back to when that field is left at
/// 0 or the setting predates this default's existence.
const VALIDATE_EVERY = 100;

/// Tokens that must have gone through the model before anything is said
/// about the corpus or about overfitting.
///
/// Held-out loss measured at step 250 of a run says nothing: the model
/// has seen a fraction of a percent of its text, both curves are falling
/// off a cliff together, and the differences the rules look at are
/// dominated by how fast that fall happens to be at two nearby moments.
/// A warning that fires there is a warning about nothing, and a page
/// that cries wolf at step 250 has taught the user to ignore it by the
/// time it is right.
const TOKENS_BEFORE_JUDGING_HELD_OUT = 2e6;

function post(type, payload = {}, transfer) {
  if (transfer) self.postMessage({ type, ...payload }, transfer);
  else self.postMessage({ type, ...payload });
}

// Anything that escapes a handler - or happens outside one - reaches the
// page instead of dying in a console nobody has open.
self.addEventListener('error', (event) => {
  post('worker-error', {
    message: (event && (event.message || String(event.error))) || 'unknown worker error',
    stack: (event && event.error && event.error.stack) || '',
  });
});
self.addEventListener('unhandledrejection', (event) => {
  const reason = event && event.reason;
  post('worker-error', {
    message: `unhandled rejection: ${(reason && reason.message) || String(reason)}`,
    stack: (reason && reason.stack) || '',
  });
});

function fail(rid, error) {
  // The stack travels with the message. "Cannot read properties of
  // undefined" is useless on its own; the same error with a frame in it
  // names the call that did it, and the page shows both.
  post('error', {
    rid,
    message: error && error.message ? error.message : String(error),
    stack: (error && error.stack) || '',
  });
}

/// Coerce a value that must reach wasm as a string.
///
/// wasm-bindgen reads `.length` off whatever it's handed for a `String`
/// parameter, so `undefined` there throws "Cannot read properties of
/// undefined (reading 'length')" from inside generated glue, with no clue
/// which call it came from. A stored source with no text — an old record,
/// a file that read as nothing — reaches `upsert_source` exactly that
/// way. Coercing at the boundary turns a mystery into either working
/// code or an honest error message.
function text(value, what) {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) {
    throw new Error(`${what} was ${value === null ? 'null' : 'missing'}`);
  }
  return String(value);
}

async function ensureWasm() {
  if (!wasmReady) {
    wasmReady = (async () => {
      log('loading ./pkg/wasm_app.js');
      const module = await import('./pkg/wasm_app.js');
      await module.default();
      wasm = module;
      log('wasm module ready');
    })().catch((error) => {
      // A failed load must not be cached as a pending promise, or every
      // later call waits on something that will never resolve.
      wasmReady = null;
      throw new Error(
        `could not load the wasm module (./pkg/wasm_app.js): ${(error && error.message) || error}. ` +
          'The build in frontend/pkg is missing or does not match this page.',
      );
    });
  }
  await wasmReady;
}

/// Load the checkpoint the site ships with, from bytes the page already
/// downloaded.
///
/// The download itself deliberately happens on the main thread. It used
/// to happen here, and that was a bad idea twice over: a worker's fetch
/// is harder to observe when it misbehaves (a hung one produced no
/// error, no reply, and an app that sat on "Loading the model..."
/// forever), and progress reporting has to be relayed back to the page
/// anyway. Fetching where the progress bar lives is simpler and one
/// failure mode shorter.
async function loadModelBytes(bytes) {
  await ensureWasm();
  llm = wasm.WasmLLM.from_checkpoint(new Uint8Array(bytes));
  await initGpu();
  return describeModel();
}

/// Ask for a WebGPU device and upload the weights to it.
///
/// This is what training runs on, and there is no second path: without a
/// device the page can still generate (on the CPU) but cannot train.
/// A failure is reported rather than thrown, so the page can say which
/// of the two it is looking at.
/// Verbose logging, on by default: this page's whole performance story
/// is "which device ran it, and how long did a step take", and that has
/// to be readable in the console rather than inferred from a progress
/// bar. Logs go to the worker's console and are mirrored to the page's.
function log(message, data) {
  if (data === undefined) {
    console.log(`[scriptonait] ${message}`);
  } else {
    console.log(`[scriptonait] ${message}`, data);
  }
  post('log', { message, data: data === undefined ? null : data });
}

async function initGpu() {
  if (!llm) return;
  const startedAt = performance.now();
  try {
    const summary = await llm.init_gpu();
    const report = JSON.parse(llm.gpu_report());
    log(
      `WebGPU device acquired in ${(performance.now() - startedAt).toFixed(0)} ms` +
        ` (f16 available: ${report.f16 ? 'yes' : 'no'}; matmuls run in f32)`,
      report,
    );
    if (report.isSoftware) {
      log(
        'WARNING: this is a SOFTWARE renderer, not your GPU. Training will run at ' +
          'roughly CPU speed. Check chrome://gpu (or your browser\'s equivalent) for why ' +
          'hardware acceleration is off.',
      );
    }
    post('gpu-status', { available: true, device: summary, report });
  } catch (error) {
    const reason = (error && error.message) || String(error);
    log(`no WebGPU device: ${reason}. Generation will run on the CPU; training cannot run.`);
    post('gpu-status', { available: false, device: 'CPU', reason });
  }
}

function describeModel() {
  const info = llm.info();
  return {
    layers: info.layers,
    uniqueLayers: info.unique_layers,
    hidden: info.hidden,
    heads: info.heads,
    kvHeads: info.kv_heads,
    contextLen: info.context_len,
    window: info.window,
    vocabSize: info.vocab_size,
    params: info.params,
    step: info.step,
    pretrained: info.pretrained,
    device: llm.device_summary(),
    usingGpu: llm.using_gpu(),
    sources: llm.num_sources(),
    corpusTokens: llm.total_tokens(),
  };
}

function describePrompt(prompt) {
  const parsed = llm ? llm.parse_prompt(prompt) : wasm.parse_prompt_standalone(prompt);
  return {
    form: parsed.form,
    targetWords: parsed.target_words,
    subject: parsed.subject,
    reference: parsed.reference,
    instruction: parsed.instruction,
  };
}

async function generate({ prompt, extraContext, temperature, topK, topP, minP, repetitionPenalty, seed, maxTokens }) {
  stopRequested = false;
  const startedAt = performance.now();
  let lastPost = 0;
  let tokens = 0;

  const result = await llm.generate(
    prompt,
    extraContext || '',
    temperature,
    topK,
    topP,
    minP || 0,
    repetitionPenalty,
    seed,
    inferenceDevice !== 'cpu',
    // 0 = continuous: length stays whatever the prompt itself asks for
    // (or the default budget, if it asks for nothing).
    maxTokens || 0,
    (piece, words) => {
      tokens += 1;
      // Text goes back immediately — that's what makes the page feel
      // alive — but the *statistics* are throttled, because posting a
      // structured message per token costs more than generating one.
      post('generate-piece', { piece });
      const now = performance.now();
      if (now - lastPost >= PROGRESS_INTERVAL_MS) {
        lastPost = now;
        const elapsed = (now - startedAt) / 1000;
        post('generate-progress', {
          words,
          tokens,
          elapsedSeconds: elapsed,
          tokensPerSecond: elapsed > 0 ? tokens / elapsed : 0,
        });
      }
      return !stopRequested;
    },
  );

  const elapsed = (performance.now() - startedAt) / 1000;
  if (result.stop_reason === 'busy') {
    // A single refusal, not a retry: the GPU is doing something else
    // (training, a save, another generation) right now. Nothing here
    // waits or polls for it to finish — switch Inference to CPU, or try
    // again once whatever's running has finished.
    throw new Error('the GPU is busy (training or another save) — switch to CPU, or try again once it finishes');
  }
  return {
    text: result.text,
    wordCount: result.word_count,
    tokensGenerated: result.tokens_generated,
    stopReason: stopRequested ? 'stopped' : result.stop_reason,
    elapsedSeconds: elapsed,
    tokensPerSecond: elapsed > 0 ? result.tokens_generated / elapsed : 0,
  };
}

/// Fine-tune until stopped or until `maxSteps`, yielding between slices.
///
/// `effort` is the fraction of wall-clock time this is allowed to use
/// (0.25 = work a quarter of the time). The slice length is fixed and the
/// pause is derived from it, rather than the other way around, so a
/// single step never gets interrupted — it can't be — and the machine
/// gets a predictable gap even if one step turns out slow.
/// Generate a short sample from the weights as they are right now.
///
/// Uses the ordinary generation path, stopped by its own callback once
/// enough words have arrived — the model is mid-training, so this is
/// about hearing where it has got to, not producing anything finished.
/// `generate` pulls the freshly trained weights off the GPU first, so a
/// sample shows the model as it is at this step rather than as it was
/// when training started.
/// `sampling` is the Inference tab's own settings — the same values
/// Generate uses, not a second hidden set tucked away in here.
/// `maxTokens` is that tab's own length setting too (0 = Continuous).
async function trainingSample(prompt, maxTokens, sampling) {
  const result = await llm.generate(
    prompt,
    '',
    sampling.temperature,
    sampling.topK,
    sampling.topP,
    sampling.minP,
    sampling.repetitionPenalty,
    Math.floor(Math.random() * 1e9),
    // The Inference tab's own device choice. GPU here runs on the same
    // device training does, so it still serializes with it (see the
    // caller); CPU is the race-free path and runs alongside training
    // without pausing it.
    inferenceDevice !== 'cpu',
    maxTokens || 0,
    () => true,
  );
  return result.text;
}

// --- The training plan -------------------------------------------------
//
// A loss number and a step count do not tell anybody what to do next.
// The plan does: which phase the run is in, what that phase means, and
// which of the things a person can actually change would help.
//
// Everything here is arithmetic over numbers the wasm side already
// keeps. It is recomputed on the validation cadence rather than every
// step, because none of it moves faster than that.

/// Two different questions get confused constantly, and confusing them
/// is how a training page ends up giving advice that contradicts itself
/// from one visit to the next. They are:
///
///   1. **Has this model been trained enough yet?** Answered by tokens
///      *seen* — how many tokens have gone through it, counting repeats.
///      This is the number that governs early on, and it is almost
///      always the binding one, because a run has to process tens of
///      millions of tokens before anything reads like language.
///   2. **Is there enough text for a model this size?** Answered by
///      tokens *available*. This only binds once the first question is
///      answered — a corpus cannot be the limit on a model that has
///      barely been trained on it.
///
/// The 20-tokens-per-parameter figure below is from Hoffmann et al.
/// (2022), and it is *not* a threshold below which a model cannot learn
/// English. It says how to split a fixed compute budget between model
/// size and data for one pass over that data. Below it you are not
/// compute-optimal — you will reach diminishing returns sooner and
/// overfit earlier — which is a different statement from "this will not
/// work".
const TOKENS_PER_PARAM = 20;

/// The Settings tab's Training Mode "Auto" peak learning rate, before
/// this model has seen its own compute-optimal token budget
/// (`TOKENS_PER_PARAM * params`, the same "wanted" figure `planActions`
/// already advises against) — large enough to actually learn a language
/// from a random start. 6e-4 is what nanoGPT uses for a 768-wide GPT-2;
/// a narrower model tolerates more rather than less, so it is
/// conservative at the widths this page builds.
const AUTO_LR_FROM_SCRATCH = 6e-4;
/// Auto's peak rate once a model is past that budget: small enough not
/// to undo what it already knows. Whether a model is "past" its budget
/// is judged fresh every settings push (see `applyLiveSettings`) against
/// live `tokensSeen` — not against whether the model was ever reloaded
/// from storage, which `model.pretrained` used to stand in for and got
/// this permanently wrong for any model that had survived one page
/// reload, regardless of how little of its budget it had actually used.
const AUTO_LR_TRAINED = 5e-5;

/// How many passes over the same text are worth making. Muennighoff et
/// al. (2023) found repeated data holds up well to about four epochs and
/// decays after; past roughly sixteen it adds nearly nothing. Four is
/// the number to plan a run against: a corpus is worth about four times
/// its token count in useful training.
const USEFUL_EPOCHS = 4;

/// Below this many tokens seen, nothing about the corpus can be
/// concluded — the model has not been trained enough for the corpus to
/// be what is limiting it. Ten million is where a model this size
/// starts producing sentences rather than words.
const TOKENS_BEFORE_JUDGING_THE_CORPUS = 10e6;

/// How much the held-out curve moved between its two most recent
/// windows of measurements, and the noise floor to judge that move
/// against: `trend > noise` is still meaningfully improving, `trend <
/// -noise` has risen (memorizing), anything in between is a plateau.
/// `{ trend: null, noise: 0 }` before there is enough history or enough
/// of the corpus has gone through the model for either answer to mean
/// anything (see TOKENS_BEFORE_JUDGING_HELD_OUT) — every caller must
/// treat `null` as "no verdict yet," not as zero.
///
/// Shared by `trainingPhase` and the adaptive scheduler axes (cool-down
/// timing, plan length) below, so anything that asks "is this curve
/// still improving" asks it through the exact same math. They used to
/// each keep their own copy: the plateau detector cut the learning rate
/// in half at step 3,800 while the phase display beside it said
/// "held-out loss still improving", because each read the curve's own
/// spread through a different formula. heldOutNoise's relative floor
/// also matters here specifically — a curve that has gone smooth rather
/// than actually flat has almost no spread of its own, and a noise
/// floor built only from that spread is what mistook a still-falling
/// curve for a plateau before.
function heldOutTrend(heldOut, tokensSeen) {
  const WINDOW = 5;
  if (heldOut.length < WINDOW * 2 || tokensSeen < TOKENS_BEFORE_JUDGING_HELD_OUT) {
    return { trend: null, noise: 0 };
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const recent = heldOut.slice(-WINDOW * 2);
  return {
    trend: mean(recent.slice(0, WINDOW)) - mean(recent.slice(WINDOW)),
    noise: heldOutNoise(heldOut, WINDOW),
  };
}

/// Which phase of a run this is, and what that means for what you are
/// looking at.
///
/// The phases are not decoration. "Loss is barely moving" means one
/// thing in the first fifty steps, when the learning rate is still a
/// fraction of its peak, and the opposite thing two thousand steps
/// later. Saying which is which is the difference between waiting and
/// wasting an afternoon.
function trainingPhase(plan, { heldOut, trainingLoss }) {
  const { step, plannedSteps, warmupSteps, peakLr, lrNow, minLrRatio } = plan;
  if (step < warmupSteps) {
    return {
      key: 'warm-up',
      title: 'Warm-up',
      detail:
        `Rate ${lrNow.toExponential(1)} of ${peakLr.toExponential(1)} peak, step ` +
        `${step.toLocaleString()} of ${warmupSteps.toLocaleString()}.`,
    };
  }

  // Both gated on the model having actually been trained. A curve 250
  // steps into a run is falling off a cliff; the difference between two
  // adjacent windows of it is about how steep the cliff is at two nearby
  // moments, and calling that "overfitting" is how a page ends up
  // warning about memorization before the model can spell.
  const { trend, noise } = heldOutTrend(heldOut, plan.tokensSeen);
  const gap = trainingLoss === null || heldOut.length === 0
    ? null
    : heldOut[heldOut.length - 1] - trainingLoss;

  // "Overfitting" means the model is favoring memorized training
  // examples over generalizing, which cannot yet be true of examples the
  // run has not been through once — below one pass over the corpus, the
  // same statistical signal (held-out rising faster than its own noise
  // floor) falls through to the plateau branch below instead, which
  // already gives the right answer for it: capacity, not data.
  const epochs = plan.trainingTokens > 0 ? plan.tokensSeen / plan.trainingTokens : 0;
  if (trend !== null && trend < -noise && epochs >= 1) {
    return {
      key: 'overfitting',
      title: 'Overfitting',
      detail: 'Held-out loss rising while training loss falls. Best model so far is saved.',
    };
  }
  if (trend !== null && trend < noise) {
    return {
      key: 'plateau',
      title: 'Plateau',
      detail:
        'Held-out loss has stopped improving' +
        (gap === null ? '' : `, ${gap.toFixed(2)} above training loss`) + '.',
    };
  }
  // Past the end of the plan the cosine has nothing left to decay: the
  // rate is parked on its floor and stays there however long training
  // continues. That is a different situation from the tail of a run, and
  // calling both "cooling down" produced the sentence "down to 5.0e-5,
  // near its floor of 5.0e-5", which says nothing at all.
  if (plannedSteps > 0 && step >= plannedSteps) {
    return {
      key: 'past-plan',
      title: 'Past the planned run',
      detail:
        `Step ${step.toLocaleString()} of a planned ${plannedSteps.toLocaleString()}. Rate ` +
        `parked at its floor, ${lrNow.toExponential(1)}. Set Steps to the length you intend.`,
    };
  }
  // The cosine tail: the last stretch of a planned run, where the rate
  // is close to its floor and the model is settling rather than moving.
  if (plannedSteps > 0 && lrNow <= peakLr * (minLrRatio + 0.05)) {
    return {
      key: 'cooling',
      title: 'Cooling down',
      detail:
        `Rate ${lrNow.toExponential(1)} of ${peakLr.toExponential(1)} peak, approaching floor ` +
        `${(peakLr * minLrRatio).toExponential(1)} at step ${plannedSteps.toLocaleString()}.`,
    };
  }
  return {
    key: 'learning',
    title: 'Learning',
    detail:
      `Step ${step.toLocaleString()} of ${plannedSteps.toLocaleString()}, rate ` +
      `${lrNow.toExponential(1)} of ${peakLr.toExponential(1)} peak` +
      (plan.plateauScale < 1 ? ` (cut to ${plan.plateauScale.toFixed(2)}x)` : '') +
      '. Held-out loss improving.',
  };
}

/// What a person could actually do about it, most useful first.
///
/// The order here is the whole point, and getting it wrong is worse than
/// saying nothing. An earlier version led with "your corpus is 43x too
/// small for this model" at step 4,573 — which is a statement about
/// compute-optimal allocation, presented as though it were a verdict on
/// whether the thing can work, to somebody whose model had processed
/// 0.3 passes of its text and was already producing English words.
///
/// So: training progress first, because until a model has actually been
/// trained the corpus cannot be what is limiting it. The corpus second,
/// and only once there is enough training behind it for the claim to
/// mean anything.
function planActions(plan, phase, { heldOut, trainingLoss, tokensSeen, tokensPerSecond, tokensPerStep }) {
  const actions = [];
  const { params, trainingTokens, validationTokens, contextLen } = plan;
  const round = (n) => Math.round(n).toLocaleString();
  const duration = (seconds) => {
    if (!isFinite(seconds) || seconds <= 0) return null;
    if (seconds < 3600) return `${Math.round(seconds / 60)} minutes`;
    if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} hours`;
    return `${(seconds / 86400).toFixed(1)} days`;
  };

  // What this corpus is worth in total: its tokens, times the number of
  // passes over the same text that still teach something.
  const budget = trainingTokens * USEFUL_EPOCHS;
  const epochs = trainingTokens > 0 ? tokensSeen / trainingTokens : 0;

  // 1. Is it trained yet? Almost always the answer early on, and the
  //    only one that can be acted on by waiting.
  if (tokensSeen < TOKENS_BEFORE_JUDGING_THE_CORPUS && tokensSeen < budget) {
    const target = Math.min(TOKENS_BEFORE_JUDGING_THE_CORPUS, budget);
    const remaining = target - tokensSeen;
    const eta = tokensPerSecond > 0 ? duration(remaining / tokensPerSecond) : null;
    actions.push({
      key: 'keep-training',
      urgency: 'high',
      text:
        `${round(tokensSeen)} tokens seen (${epochs.toFixed(2)} passes). Keep training to about ` +
        `${round(target)} tokens seen${eta ? ` (~${eta})` : ''}. Corpus can't be judged before then.`,
    });
  }

  // 1b. Is the run about to ask for more passes than repeated text is
  //     worth? This has to be said before the run, not forty hours into
  //     it: the existing epoch warning only fires once the passes have
  //     actually been made, which is too late to be advice.
  if (plan.plannedSteps > 0 && trainingTokens > 0 && tokensPerStep > 0) {
    const planned = plan.plannedSteps * tokensPerStep;
    const plannedEpochs = planned / trainingTokens;
    if (plannedEpochs > USEFUL_EPOCHS * 1.2) {
      const enough = Math.round((budget / tokensPerStep) / 100) * 100;
      const hours = tokensPerSecond > 0 ? planned / tokensPerSecond / 3600 : null;
      const enoughHours = tokensPerSecond > 0 ? budget / tokensPerSecond / 3600 : null;
      actions.push({
        key: 'run-too-long',
        urgency: 'normal',
        text:
          `${plan.plannedSteps.toLocaleString()} steps planned = ${round(planned)} tokens, ` +
          `${plannedEpochs.toFixed(1)} passes${hours ? `, ~${hours.toFixed(0)}h` : ''}. About ` +
          `${round(enough)} steps${enoughHours ? ` (~${enoughHours.toFixed(0)}h)` : ''} covers ` +
          'what this corpus is worth — set Steps to that.',
      });
    }
  }

  // 2. Only now: is there enough text for a model this size? Judged
  //    against what repeated passes are worth, not against a single
  //    pass, because this run makes several.
  const judged = tokensSeen >= TOKENS_BEFORE_JUDGING_THE_CORPUS ||
    phase.key === 'overfitting' ||
    phase.key === 'plateau';
  const wanted = params * TOKENS_PER_PARAM;
  if (judged && trainingTokens > 0 && budget < wanted) {
    const perSource = plan.sources > 0 ? plan.corpusTokens / plan.sources : 0;
    const moreSources = perSource > 0 ? Math.ceil((wanted - budget) / (perSource * USEFUL_EPOCHS)) : 0;
    // The model size this corpus supports, at the same rule — stated as
    // the trade it is, not as an instruction. A smaller model reaches
    // its best sooner and that best is lower; which one is wanted is
    // not something a page can decide.
    const supported = budget / TOKENS_PER_PARAM;
    actions.push({
      key: 'corpus-size',
      urgency: phase.key === 'overfitting' ? 'high' : 'normal',
      text:
        `${round(trainingTokens)} training tokens, worth about ${round(budget)} over ` +
        `${USEFUL_EPOCHS} passes. A ${round(params)}-parameter model wants about ` +
        `${round(wanted)}. Add about ${round(moreSources)} more sources this size, or use a ` +
        `model nearer ${round(supported)} parameters.`,
    });
  }

  if (validationTokens < contextLen + 1) {
    actions.push({
      key: 'no-validation',
      urgency: 'high',
      text:
        'Not enough text held out to measure overfitting. Add sources past about ' +
        `${round((contextLen * 20) / 0.05)} tokens.`,
    });
  }

  // What kind of text is missing. A corpus that is all one thing teaches
  // the shape of that thing: thirty scripts teach a model that every
  // paragraph is one line of dialogue long.
  const mix = plan.mix;
  if (judged && Array.isArray(mix) && mix.length > 0 && plan.corpusTokens > 0) {
    const largest = mix[0];
    const share = largest.tokens / plan.corpusTokens;
    if (share > 0.8) {
      const missing = ['film scripts', 'novels and prose fiction', 'essays and philosophy',
        'verse and lyrics']
        .filter((label) => !mix.some((m) => m.label === label && m.tokens / plan.corpusTokens > 0.05));
      if (missing.length > 0) {
        actions.push({
          key: 'corpus-mix',
          urgency: 'normal',
          text:
            `${Math.round(share * 100)}% of your corpus is ${largest.label}. Add ` +
            `${missing.slice(0, 2).join(' or ')} to widen it.`,
        });
      }
    }
  }

  if (phase.key === 'past-plan') {
    actions.push({
      key: 'set-a-plan',
      urgency: 'normal',
      text: 'Past the planned steps; rate is flat at its floor. Set Steps to the length you intend.',
    });
  }

  if (phase.key === 'overfitting') {
    actions.push({
      key: 'stop-here',
      urgency: 'high',
      text: 'Held-out loss is getting worse — stop here, or add more text and keep training.',
    });
  }
  if (phase.key === 'plateau') {
    const gap = trainingLoss === null || heldOut.length === 0
      ? 0
      : heldOut[heldOut.length - 1] - trainingLoss;
    actions.push({
      key: 'plateau-what-next',
      urgency: 'normal',
      text: gap > 0.3
        ? 'Held-out sits well above training loss — add more text.'
        : 'Both curves have flattened — a larger hidden size or another layer would help more than more steps.',
    });
  }

  if (plan.plateauScale <= 0.06) {
    actions.push({
      key: 'plateau-floor',
      urgency: 'high',
      text:
        `Rate cut to ${plan.plateauScale.toFixed(2)}x and held-out loss still flat. The limit ` +
        'is the corpus or model shape, not the rate.',
    });
  }

  // What the samples themselves say. Loss can fall for a long time
  // while the output is still not words.
  const q = plan.quality;
  if (q && q.words >= 20) {
    if (q.knownWordRate < 0.75) {
      actions.push({
        key: 'not-words-yet',
        urgency: 'normal',
        text:
          `${Math.round(q.knownWordRate * 100)}% of the last sample's words appear in your ` +
          'corpus' +
          (q.unknownExamples && q.unknownExamples.length
            ? ` (${q.unknownExamples.slice(0, 4).join(', ')})`
            : '') + '.',
      });
    }
    if (q.repeated4gramRate > 0.25) {
      actions.push({
        key: 'repeating',
        urgency: 'normal',
        text:
          `${Math.round(q.repeated4gramRate * 100)}% of four-word runs in the last sample ` +
          'repeat. Raise repetition penalty or min-p in Inference settings.',
      });
    }
  }

  // Repeated passes stop paying somewhere past four. Said once the run
  // is actually there, not as a warning about a run that has done 0.3.
  if (epochs >= USEFUL_EPOCHS) {
    actions.push({
      key: 'epochs',
      urgency: epochs >= USEFUL_EPOCHS * 4 ? 'high' : 'normal',
      text:
        `${epochs.toFixed(1)} passes over your text (useful passes: about ${USEFUL_EPOCHS}). ` +
        'More text would help more than more steps.',
    });
  }

  return actions;
}

/// The whole plan: the phase, the numbers behind it, and what to do.
function buildPlan(state) {
  const plan = JSON.parse(llm.training_plan());
  plan.quality = state.quality || null;
  plan.bitsPerByte = state.bitsPerByte || 0;
  const phase = trainingPhase(plan, state);
  const tokensPerStep = state.tokensPerStep || plan.contextLen;
  // Counted as it happened, on the wasm side, and carried in the
  // checkpoint. It used to be `step * whatever the batch size is now`,
  // which is wrong twice: the batch size changes between runs, and the
  // page's current setting has nothing to do with what earlier steps
  // actually processed. A model trained at batch 4 and then looked at
  // with the box set to 1 reported a quarter of its real training.
  const tokensSeen = plan.tokensSeen;
  const runStep = Math.max(0, plan.step - plan.startStep);
  const remaining = plan.plannedSteps > runStep ? plan.plannedSteps - runStep : 0;
  const tokensPerSecond = state.msPerStep > 0 ? tokensPerStep / (state.msPerStep / 1000) : 0;
  return {
    phase,
    actions: planActions(plan, phase, { ...state, tokensPerStep, tokensSeen, tokensPerSecond }),
    numbers: {
      step: plan.step,
      // Steps into this run, which is the frame the schedule works in.
      runStep: Math.max(0, plan.step - plan.startStep),
      plannedSteps: plan.plannedSteps,
      warmupSteps: plan.warmupSteps,
      lrNow: plan.lrNow,
      peakLr: plan.peakLr,
      params: plan.params,
      tokensPerStep,
      tokensSeen,
      trainingTokens: plan.trainingTokens,
      validationTokens: plan.validationTokens,
      tokensPerParam: plan.params > 0 ? plan.trainingTokens / plan.params : 0,
      wantedTokens: Math.round(plan.params * TOKENS_PER_PARAM),
      epochs: plan.trainingTokens > 0 ? tokensSeen / plan.trainingTokens : 0,
      corpusChars: plan.corpusChars,
      tokensPerSecond,
      etaSeconds: state.msPerStep > 0 ? (remaining * state.msPerStep) / 1000 : null,
      mix: plan.mix,
      sources: plan.sources,
      plateauScale: plan.plateauScale,
      bitsPerByte: plan.bitsPerByte,
      quality: plan.quality,
    },
  };
}

/// Measure this machine, once, and let the measurement pick the
/// settings.
///
/// Two things about a training step are properties of the GPU and its
/// driver rather than of the model, and neither can be reasoned out from
/// here:
///
///   * How much work belongs in one command buffer. Too little pays the
///     submission cost on every dispatch; too much hands the driver a
///     buffer long enough to trip its watchdog and lose the device. The
///     best value differs by adapter, by backend and by driver version.
///   * How many sequences a batch should hold. A larger batch is a
///     steadier gradient and usually more tokens per second, up to the
///     point where a single step takes long enough that stopping feels
///     broken and the watchdog gets interested.
///
/// So both are timed here, on this machine, with the model that is
/// actually loaded, and the winner is stored. `bench_step` runs at
/// learning rate zero and restores the step counter, so this costs time
/// and changes nothing else.
async function benchmark({ ceilingMs = 1500, budgetMs = 60000, repeats = 3 } = {}) {
  const device = JSON.parse(llm.gpu_report());
  if (!device.available) throw new Error('benchmarking needs a GPU device');
  const info = llm.info();
  const contextLen = info.context_len;
  const startedAt = performance.now();
  const overBudget = () => performance.now() - startedAt > budgetMs;
  const tokensPerSecond = (batch, ms) => (ms > 0 ? (batch * contextLen) / (ms / 1000) : 0);

  log('benchmarking this machine — one timed sweep, nothing is learned from it', {
    adapter: device.adapter,
    backend: device.backend,
    deviceType: device.deviceType,
    contextLen,
    params: info.params,
  });

  // The first step allocates every training buffer and compiles every
  // pipeline. Timing that would measure the driver's lazy work, not the
  // step.
  const warmupStart = performance.now();
  await llm.bench_step(1, 32);
  log(`benchmark warmup ${(performance.now() - warmupStart).toFixed(0)} ms ` +
      '(allocating training state and compiling pipelines)');

  /// Fastest of `repeats` runs, not the mean: a slow run is another
  /// process getting the GPU, and the fastest is the one this
  /// configuration is capable of.
  async function timeStep(batch, chunk, runs) {
    let best = Infinity;
    for (let i = 0; i < runs; i += 1) {
      const started = performance.now();
      await llm.bench_step(batch, chunk);
      best = Math.min(best, performance.now() - started);
    }
    return best;
  }

  // --- How much work per command buffer, at one sequence -------------
  const chunkSweep = [];
  let chunk = 32;
  let chunkMs = Infinity;
  for (const candidate of [8, 16, 32, 64, 128, 256]) {
    if (overBudget()) {
      log(`benchmark: out of time budget, stopping the command-buffer sweep at ${candidate}`);
      break;
    }
    let ms;
    try {
      ms = await timeStep(1, candidate, repeats);
    } catch (error) {
      // A device lost to the watchdog takes everything with it, so a
      // failure here ends the sweep rather than continuing past it.
      log(`benchmark: ${candidate} dispatches/submit failed (${error && error.message || error}) ` +
          '— keeping the best value measured before it');
      break;
    }
    chunkSweep.push({ dispatchesPerSubmit: candidate, msPerStep: ms });
    log(`benchmark: ${candidate} dispatches/submit -> ${ms.toFixed(1)} ms/step ` +
        `(${tokensPerSecond(1, ms).toFixed(0)} tok/s at batch 1)`);
    if (ms < chunkMs) {
      chunkMs = ms;
      chunk = candidate;
    }
  }
  if (chunkSweep.length === 0) {
    throw new Error('the benchmark could not time a single step on this device');
  }
  post('bench-progress', { stage: 'chunk', dispatchesPerSubmit: chunk });

  // --- How many sequences per batch ----------------------------------
  //
  // Ascending, and stopping at the first candidate that is slower per
  // token or takes longer than the ceiling: past that point a batch buys
  // a marginally steadier gradient with a step nobody can interrupt.
  const batchSweep = [];
  let batchSize = 1;
  let bestRate = 0;
  let bestMs = chunkMs;
  for (const candidate of [1, 2, 4, 8, 16]) {
    if (overBudget()) {
      log(`benchmark: out of time budget, stopping the batch sweep at ${candidate}`);
      break;
    }
    let ms;
    try {
      ms = await timeStep(candidate, chunk, candidate >= 8 ? 1 : 2);
    } catch (error) {
      log(`benchmark: batch ${candidate} failed (${error && error.message || error}) ` +
          '— keeping the largest batch that worked');
      break;
    }
    const rate = tokensPerSecond(candidate, ms);
    batchSweep.push({ batchSize: candidate, msPerStep: ms, tokensPerSecond: rate });
    log(`benchmark: batch ${candidate} -> ${ms.toFixed(0)} ms/step, ${rate.toFixed(0)} tok/s`);
    // The smallest batch is taken whatever it costs — there is nothing
    // below it — but a larger one has to earn its step time.
    const first = bestRate === 0;
    if (!first && ms > ceilingMs) {
      log(`benchmark: batch ${candidate} takes ${ms.toFixed(0)} ms, past the ${ceilingMs} ms ` +
          `ceiling that keeps a step interruptible — staying at ${batchSize}`);
      break;
    }
    // Three percent, because anything smaller is inside the noise of two
    // runs and not worth a step that takes twice as long to interrupt.
    if (!first && rate <= bestRate * 1.03) {
      log(`benchmark: batch ${candidate} is no faster per token than ${batchSize} — stopping here`);
      break;
    }
    bestRate = rate;
    bestMs = ms;
    batchSize = candidate;
  }

  const profile = {
    version: BENCH_VERSION,
    adapter: device.adapter,
    backend: device.backend,
    deviceType: device.deviceType,
    isSoftware: device.isSoftware,
    dispatchesPerSubmit: chunk,
    batchSize,
    msPerStep: bestMs,
    tokensPerSecond: bestRate,
    // The batch ceiling depends on the model's shape, so a profile
    // measured against a different model says nothing about this one.
    shape: {
      layers: info.layers,
      uniqueLayers: info.unique_layers,
      hidden: info.hidden,
      heads: info.heads,
      kvHeads: info.kv_heads,
      contextLen: info.context_len,
      window: info.window,
      vocabSize: info.vocab_size,
    },
    chunkSweep,
    batchSweep,
    elapsedSeconds: (performance.now() - startedAt) / 1000,
  };
  llm.set_dispatches_per_submit(chunk);
  log('machine profile measured', profile);
  return profile;
}

/// Recompute the plan, announce a change of phase, and hand it to the
/// page. Called at the start of a run and on the validation cadence,
/// never per step: nothing in it moves faster than that.
function reportPlan(state) {
  let plan;
  try {
    plan = buildPlan(state);
  } catch (error) {
    log(`could not build the training plan: ${(error && error.message) || error}`);
    return state.lastPhase;
  }
  if (plan.phase.key !== state.lastPhase) {
    log(`phase: ${plan.phase.title} — ${plan.phase.detail}`, plan.numbers);
    for (const action of plan.actions) {
      log(`plan (${action.urgency}): ${action.text}`);
    }
    if (state.lastPhase !== null) {
      recordEvent(plan.numbers.step, 'phase', `${plan.phase.title}: ${plan.phase.detail}`);
    }
  }
  post('train-plan', plan);
  return plan.phase.key;
}

/// The run currently in flight, so every record can say which run it
/// belongs to. A model is trained across many runs with different
/// settings, and "which run was that" is the first question anybody asks
/// of a number six hours old.
let runId = null;

/// Something worth remembering that is not a measurement: a run
/// starting, a rate being cut, a piece of advice. Recorded on the same
/// timeline as the numbers so the two can be read together — a loss
/// curve with no note of what changed halfway through is a curve nobody
/// can explain.
function recordEvent(step, kind, text, extra = {}) {
  post('train-record', { runId, step, kind, text, at: Date.now(), ...extra });
}

/// How long one slice of training runs before the worker checks in
/// (progress, autosave, samples) and, on Auto effort, hands the GPU
/// back for a beat. Fixed — the thing effort actually tunes is the
/// pause between slices, not the slice length itself.
const SLICE_MS = 120;

/// Every setting a run in flight can react to without a fresh model —
/// the same ground `train()` starts from and `update-training-settings`
/// changes mid-run, through the one function below that both call. Model
/// shape (layers/hidden/heads/context/window) is deliberately not here:
/// changing that only ever takes effect on a newly built model, there is
/// no "live" version of it.
///
/// Two kinds of field live in here. Most (`autosaveEvery`, `sampleEvery`,
/// the sample prompt/length/sampling, `batchSize`, `maxSteps`, `pauseMs`)
/// are plain data the training loop rereads every cycle — a change just
/// lands on the next check. `peakLearningRate` and `boundarySampleRate`
/// are different: the model itself owns that state (the schedule, the
/// corpus), so applying one of those means calling into `llm` right
/// here rather than only writing a JS-side variable — see
/// `applyLiveSettings` below.
const live = {
  batchSize: 1,
  maxSteps: 0,
  effort: 1,
  pauseMs: 0,
  autosaveEvery: AUTOSAVE_EVERY_STEPS,
  validateEvery: VALIDATE_EVERY,
  sampleEvery: 0,
  samplePrompt: '',
  sampleMaxTokens: 0,
  sampling: null,
  // 'wsd' | 'cosine-cuts' | 'cosine' — see the Settings tab's own
  // Schedule control. Only 'cosine-cuts' ever calls decay_on_plateau;
  // the model itself only distinguishes 'wsd' from everything else
  // (llm.set_schedule_kind), since "cosine" and "cosine-cuts" are the
  // same shape and differ only in whether this file reacts to a
  // plateau on top of it.
  scheduleMode: 'wsd',
  // Which formula `set_project_plan` uses for warmup length — see the
  // Settings tab's Warm-up control and `set_warmup_strategy`. `false` is
  // the existing 2%-of-plan heuristic every plan has used until now.
  warmupVariance: false,
  // The Training tab's own Mode control (not the Scheduler's) — true
  // picks AUTO_LR_FROM_SCRATCH/AUTO_LR_TRAINED by live token budget on
  // every settings push instead of using whatever `peakLearningRate`
  // was sent; see `applyLiveSettings`.
  autoLearningRate: true,
  // The Settings tab's "Decay start" and "Plan length" controls — only
  // reachable when Cool-down timing is Deferred (WSD) and Stable phase
  // is Flat, same guard `applySchedulerCompatibility` already enforces
  // for reactive-cuts. Both false is every plan's behavior until now:
  // decay starts at the fixed WSD_DECAY_FRACTION point, and a plan never
  // grows itself.
  decayStartAdaptive: false,
  planLengthAdaptive: false,
};

/// One place both `train()` (seeding from the settings a run started
/// with) and the `update-training-settings` handler (applying a change
/// mid-run) go through, so the two can never disagree about what "0" or
/// "missing" falls back to, or drift into two different ideas of what a
/// setting means. Every field is optional — only what's present is
/// touched, so a partial push from one changed control never resets
/// anything else.
function applyLiveSettings({
  batchSize, maxSteps, effort, peakLearningRate, autoLearningRate, scheduleMode, boundarySampleRate,
  autosaveFrequencySteps, metricsEvery, sampleEvery, samplePrompt, sampleMaxTokens, sampling,
  warmupVariance, decayStartAdaptive, planLengthAdaptive,
} = {}) {
  if (typeof batchSize === 'number' && batchSize > 0) live.batchSize = batchSize;
  // Before `maxSteps` below: set_project_plan reads this same strategy
  // flag to compute warmup_steps, so the plan has to see the flag's new
  // value, not the one it's about to replace.
  if (typeof warmupVariance === 'boolean' && llm) {
    live.warmupVariance = warmupVariance;
    llm.set_warmup_strategy(warmupVariance);
  }
  if (typeof maxSteps === 'number') {
    live.maxSteps = maxSteps;
    // The whole project's planned length, not just this sitting —
    // idempotent and a no-op at 0, so this is safe to call on every
    // settings push, not just the first.
    if (llm) llm.set_project_plan(maxSteps);
  }
  if (typeof effort === 'number') {
    live.effort = effort;
    live.pauseMs = Math.max(0, Math.round(SLICE_MS * (1 - effort) / Math.max(effort, 0.05)));
  }
  if (typeof scheduleMode === 'string' && llm) {
    live.scheduleMode = scheduleMode;
    llm.set_schedule_kind(scheduleMode === 'wsd' ? 'wsd' : 'cosine');
    // A model saved while stuck under an old plateau cut carries that
    // cut in its checkpoint regardless of what schedule it resumes
    // under — switching away from "cosine-cuts" specifically is someone
    // trying to get out from under exactly that, so clear it here
    // rather than leaving them to separately find "Undo Plateau Cut."
    if (scheduleMode !== 'cosine-cuts' && llm.plateau_scale() < 1) {
      const before = llm.plateau_scale();
      llm.reset_plateau_scale();
      log(`schedule changed to ${scheduleMode}; cleared a plateau cut left at ${before.toFixed(2)}x`);
      recordEvent(llm.step(), 'schedule-restored',
        `switched to ${scheduleMode}; an earlier plateau cut (${before.toFixed(2)}x) was cleared`);
    }
  }
  if (typeof autoLearningRate === 'boolean') live.autoLearningRate = autoLearningRate;
  // Auto mode picks its own rate, judged fresh against this model's
  // *live* token count every time settings are pushed — not decided
  // once at Train and never revisited, and not standing in a fact about
  // the model (whether it was ever reloaded from storage, which is what
  // `pretrained` used to mean here and got permanently wrong for any
  // model that had survived one page reload). `peakLearningRate` is
  // still what Manual mode's typed rate arrives as.
  let effectiveLr = peakLearningRate;
  if (live.autoLearningRate && llm) {
    const plan = JSON.parse(llm.training_plan());
    effectiveLr = plan.tokensSeen < plan.params * TOKENS_PER_PARAM ? AUTO_LR_FROM_SCRATCH : AUTO_LR_TRAINED;
  }
  if (typeof effectiveLr === 'number' && effectiveLr > 0 && llm) {
    const before = JSON.parse(llm.training_plan()).peakLr;
    llm.set_learning_rate(effectiveLr);
    const plan = JSON.parse(llm.training_plan());
    if (plan.peakLr !== before) {
      // A rate change — auto-selected or typed by hand, either is the
      // schedule's own judgment being overridden — clears a plateau cut
      // still sitting underneath it from whatever the rate used to be,
      // which would otherwise keep suppressing it. This was the actual
      // shape of "I can't change the learning rate": a new peak set
      // while plateau_scale was already down at its 0.05 floor changed
      // nothing about the rate actually in force.
      const hadCut = plan.plateauScale < 1;
      if (hadCut) llm.reset_plateau_scale();
      const inForce = JSON.parse(llm.training_plan()).lrNow;
      const how = live.autoLearningRate ? 'auto-selected' : 'set by hand';
      log(
        `peak learning rate ${how}: ${before.toExponential(2)} to ` +
          `${plan.peakLr.toExponential(2)}; the rate in force is now ${inForce.toExponential(2)}` +
          (hadCut ? ' (an earlier plateau cut was cleared)' : ''),
      );
      recordEvent(plan.step, 'rate-changed',
        `peak learning rate ${how} from ${before.toExponential(2)} to ` +
        `${plan.peakLr.toExponential(2)} (in force: ${inForce.toExponential(2)})` +
        (hadCut ? ' — cleared an earlier plateau cut' : ''));
    }
  }
  if (typeof boundarySampleRate === 'number' && boundarySampleRate >= 0 && llm) {
    llm.set_boundary_sample_rate(boundarySampleRate);
  }
  if (typeof autosaveFrequencySteps === 'number') {
    live.autosaveEvery = autosaveFrequencySteps > 0 ? autosaveFrequencySteps : AUTOSAVE_EVERY_STEPS;
  }
  if (typeof metricsEvery === 'number') {
    live.validateEvery = metricsEvery > 0 ? metricsEvery : VALIDATE_EVERY;
  }
  if (typeof sampleEvery === 'number') live.sampleEvery = sampleEvery;
  if (typeof samplePrompt === 'string') live.samplePrompt = samplePrompt;
  if (typeof sampleMaxTokens === 'number') live.sampleMaxTokens = sampleMaxTokens;
  if (sampling) live.sampling = sampling;
  if (typeof decayStartAdaptive === 'boolean') live.decayStartAdaptive = decayStartAdaptive;
  if (typeof planLengthAdaptive === 'boolean') live.planLengthAdaptive = planLengthAdaptive;
}

async function train({
  batchSize, peakLearningRate, autoLearningRate, maxSteps, effort, scheduleMode, sampleEvery,
  samplePrompt, sampleMaxTokens, sampling, autosaveFrequencySteps, metricsEvery, boundarySampleRate,
  warmupVariance, decayStartAdaptive, planLengthAdaptive,
}) {
  // The exact same call a mid-run settings change makes — see
  // `applyLiveSettings`'s own doc comment. Starting a run is that
  // function's seed case, not a second code path.
  applyLiveSettings({
    batchSize, maxSteps, effort, peakLearningRate, autoLearningRate, scheduleMode, boundarySampleRate,
    autosaveFrequencySteps, metricsEvery, sampleEvery, samplePrompt, sampleMaxTokens, sampling,
    warmupVariance, decayStartAdaptive, planLengthAdaptive,
  });
  stopRequested = false;
  training = true;
  runId = `run-${Date.now().toString(36)}`;
  // Steps, learning rate, batch size and effort were all just applied
  // above, the same way a change to any of them mid-run is — this is
  // the "start" case of that one path, not a second one.

  const info = llm.info();
  const startingPlan = JSON.parse(llm.training_plan());
  if (live.batchSize <= 1) {
    log(
      `training at batch size ${live.batchSize}: ${live.batchSize * info.context_len} tokens per ` +
        'step. That is the smallest batch there is — if this was not deliberate, the machine ' +
        'benchmark has not run for this model shape and the page fell back to it.',
    );
  }
  log('training run starting', {
    device: llm.device_summary(),
    softwareRenderer: llm.gpu_is_software(),
    batchSize: live.batchSize,
    contextLen: info.context_len,
    tokensPerStep: live.batchSize * info.context_len,
    dispatchesPerSubmit: llm.dispatches_per_submit(),
    maxSteps: live.maxSteps || `until stopped (schedule shaped for ${startingPlan.plannedSteps})`,
    startingAtStep: llm.step(),
    peakLearningRate: startingPlan.peakLr,
    effort: live.effort,
    learningRate: live.autoLearningRate ? 'automatic' : peakLearningRate,
    plateauScale: llm.plateau_scale(),
    params: info.params,
  });

  const startedAt = performance.now();
  let steps = 0;
  let tokens = 0;
  let lastPost = 0;
  let smoothedLoss = null;
  // Reread every slice — a batch-size change mid-run should show up in
  // the plan/ETA math within one slice, not only on the next Train click.
  let tokensPerStep = live.batchSize * info.context_len;
  // First sample after the first interval, not immediately: a sample at
  // step 0 is noise from an untouched model.
  let nextSampleAt = nextOnGrid(llm.step(), live.sampleEvery);
  // Held-out loss on the same cadence as the loss chart's own reporting:
  // often enough to see the two curves separate, rare enough that its
  // forward passes are a rounding error against training. A Settings-tab
  // value overrides the default; 0 or missing falls back to it rather
  // than turning measurement off — there is no "never" for this, since
  // the plateau detector depends on it.
  log(
    `held-out loss will be measured every ${live.validateEvery} steps on a fixed set of ` +
      `${VALIDATION_WINDOWS} windows (${VALIDATION_WINDOWS * info.context_len} tokens), the ` +
      'same windows each time — so two measurements differ only because the weights differ',
  );
  let nextValidateAt = nextOnGrid(llm.step(), live.validateEvery);
  // Counted from where this run starts, so a model at step 4,441 does
  // not save on its very first progress report.
  let nextAutosaveAt = nextOnGrid(llm.step(), live.autosaveEvery);
  let validationLoss = null;
  // Held-out losses in order, so the run can say when more text would
  // help more than more steps.
  const heldOut = [];
  // The phase the run was last seen in, so a change of phase is
  // announced once instead of on every recomputation.
  let lastPhase = null;
  /// The last quality measurement taken from a generated sample, so the
  /// plan can carry it between samples instead of only at the moment one
  /// is produced.
  let lastQuality = null;
  let lastBitsPerByte = 0;
  // Loss on a fixed set of training windows, drawn exactly as the
  // held-out set is. The only training number held-out loss can honestly
  // be compared with.
  let trainingProbe = null;
  let lastGradNorm = 0;
  // Held-out measurements since the last one that was actually better.
  let sinceImprovement = 0;
  let bestSeen = null;
  // Median-ish step cost, for the estimate of how long the rest takes.
  let recentStepMs = null;
  // How many times "Plan length: Adaptive, extends" has grown this run's
  // plan — see PLAN_EXTEND_MAX_TIMES.
  let planExtensions = 0;

  // Say where the run is starting from before it starts: which phase,
  // how much text there is for a model this size, what would help.
  lastPhase = reportPlan({
    heldOut, trainingLoss: null, stepsDone: 0, tokensPerStep, msPerStep: null, lastPhase: null,
  });

  // The settings this run was started with, on the record. Without this
  // a history is a list of numbers with no note of what produced them.
  {
    const plan = JSON.parse(llm.training_plan());
    recordEvent(llm.step(), 'run-started',
      `run started: ${plan.plannedSteps.toLocaleString()} steps, batch ${live.batchSize}, ` +
      `${tokensPerStep.toLocaleString()} tokens/step, peak rate ${plan.peakLr.toExponential(2)}, ` +
      `warm-up ${plan.warmupSteps}, effort ${live.effort}`,
      {
        settings: {
          plannedSteps: plan.plannedSteps, batchSize: live.batchSize, tokensPerStep, effort: live.effort,
          peakLr: plan.peakLr, warmupSteps: plan.warmupSteps,
          minLrRatio: plan.minLrRatio, weightDecay: plan.weightDecay,
          gradClip: plan.gradClip, plateauScale: plan.plateauScale,
        },
        model: {
          layers: info.layers, hidden: info.hidden, heads: info.heads,
          kvHeads: info.kv_heads, contextLen: info.context_len, window: info.window,
          vocabSize: info.vocab_size, params: info.params,
        },
        corpus: {
          sources: plan.sources, chars: plan.corpusChars,
          trainingTokens: plan.trainingTokens, validationTokens: plan.validationTokens,
        },
        device: llm.device_summary(),
        dispatchesPerSubmit: llm.dispatches_per_submit(),
      });
  }

  /// One periodic training sample: generate, score it against the
  /// corpus, record it. Closes over this run's own `runId`/`smoothedLoss`/
  /// `validationLoss`/`lastQuality` rather than taking them as
  /// parameters, since it only ever runs as part of this training run.
  async function runTrainingSample(prompt, maxTokens, samplingSettings) {
    const sampleStart = performance.now();
    const text = await trainingSample(prompt, maxTokens, samplingSettings);
    // What the loss curve cannot say: is this English, and is it still
    // saying anything new. Measured against the user's own corpus, so
    // no word list has to ship with the page.
    const quality = JSON.parse(llm.evaluate(text, validationLoss === null ? -1 : validationLoss));
    lastQuality = quality;
    // The Samples panel (backed by history) is the only place a sample
    // is shown — every one kept, not just the latest.
    post('train-record', {
      runId, kind: 'sample', at: Date.now(), step: llm.step(),
      text, quality, loss: smoothedLoss, prompt,
    });
    log(
      `sample at step ${llm.step().toLocaleString()}: ` +
        `${(quality.knownWordRate * 100).toFixed(0)}% of its words are in your corpus, ` +
        `${(quality.repeated4gramRate * 100).toFixed(0)}% of its four-word runs are repeats, ` +
        `${(quality.distinctWordRate * 100).toFixed(0)}% of its words are distinct` +
        (quality.unknownExamples.length > 0
          ? ` — words it invented: ${quality.unknownExamples.slice(0, 5).join(', ')}`
          : ''),
      quality,
    );
    log(`sample generated in ${(performance.now() - sampleStart).toFixed(0)} ms`);
  }

  while (!stopRequested && (live.maxSteps <= 0 || steps < live.maxSteps)) {
    const sliceStart = performance.now();
    // Reread every slice, same as tokensPerStep above.
    tokensPerStep = live.batchSize * info.context_len;
    // Work for a slice...
    while (performance.now() - sliceStart < SLICE_MS) {
      const stepStart = performance.now();
      let report;
      // Everything through the progress post below shares one guard: a
      // step that is refused, or anything that throws while reporting on
      // it (a bad JSON payload, say), must not end the run in silence.
      //
      // It used to: the guard that stops two GPU operations overlapping
      // returns an error, nothing caught it, the promise rejected, and
      // the loop simply stopped — leaving the page showing the last
      // sample it had produced, with no message anywhere. A narrower
      // try/catch around just `train_step` fixed that call, but left the
      // reporting code after it (which can itself throw) able to
      // reproduce the exact same silent death. A run that ends has to
      // say so, whichever part of the iteration is what ended it.
      try {
        report = await llm.train_step(live.batchSize);
        if (!report) {
          training = false;
          return { steps, stopReason: 'no-data', elapsedSeconds: (performance.now() - startedAt) / 1000 };
        }
        const stepMs = performance.now() - stepStart;
        // Exponential average, and not from the first step: that one pays
        // for allocating every training buffer on the device.
        if (steps > 0) {
          recentStepMs = recentStepMs === null ? stepMs : recentStepMs * 0.9 + stepMs * 0.1;
        }
        steps += 1;
        tokens += report.tokens;
        smoothedLoss = smoothedLoss === null ? report.loss : smoothedLoss * 0.9 + report.loss * 0.1;
        lastGradNorm = report.grad_norm;

        // The first step pays for allocating every training buffer on the
        // device, so it is logged on its own rather than averaged in.
        if (steps === 1) {
          log(`first step ${stepMs.toFixed(0)} ms (includes allocating GPU training state)`,
            JSON.parse(llm.gpu_report()));
        }
        if (steps <= 5 || steps % 25 === 0) {
          log(
            `step ${report.step.toLocaleString()}: ${stepMs.toFixed(1)} ms, ` +
              `${(report.tokens / (stepMs / 1000)).toFixed(0)} tok/s, ` +
              `loss ${report.loss.toFixed(4)}, |grad| ${report.grad_norm.toFixed(3)}, ` +
              `lr ${report.lr.toExponential(2)}, ` +
              `${report.dispatches} dispatches in ${report.submits} submits ` +
              `(${(stepMs * 1000 / Math.max(report.dispatches, 1)).toFixed(1)} us each)`,
          );
        }

        const now = performance.now();
        if (now - lastPost >= PROGRESS_INTERVAL_MS) {
          lastPost = now;
          const elapsed = (now - startedAt) / 1000;
          post('train-progress', {
            step: report.step,
            steps,
            loss: report.loss,
            smoothedLoss,
            validationLoss,
            trainingProbe,
            lr: report.lr,
            gradNorm: report.grad_norm,
            elapsedSeconds: elapsed,
            tokensPerSecond: elapsed > 0 ? tokens / elapsed : 0,
            fractionDone: live.maxSteps > 0 ? steps / live.maxSteps : 0,
            // This step's batch, in draw order — {id, excerpt} per window.
            // `id`, not a title: the page (not the worker) knows those.
            sources: JSON.parse(report.sources),
          });
        }
      } catch (error) {
        // A CPU-preferred sample's own GPU sync (worker.js's
        // sync_from_gpu call below, fire-and-forget so it can never
        // block training) briefly holds the same guard a training step
        // needs — an expected, occasional collision now that sampling
        // and training genuinely run concurrently, not the kind of
        // refusal that means something is actually wrong. Nothing else
        // in this block touches the GPU, so "already busy" can only
        // have come from the train_step() call above; skip this attempt
        // and let the slice after next retry, rather than ending the
        // run over a lock a training sample was always going to let go
        // of on its own.
        const message = (error && error.message) || String(error);
        if (/already busy/i.test(message)) {
          log(`step refused (GPU busy with a training sample's sync) — retrying next slice`);
          break;
        }
        training = false;
        const explained = describeTrainingFailure(error);
        log(`training stopped at step ${llm.step().toLocaleString()}: ${explained}`);
        recordEvent(llm.step(), 'run-failed', `training stopped: ${explained}`);
        post('train-stopped', { step: llm.step(), reason: explained });
        throw error;
      }
      // A slice runs a fixed time budget's worth of steps, not a fixed
      // number of them — on a fast machine that's a couple dozen steps,
      // and the sample/validate/autosave cadences are only ever checked
      // between slices. Left alone, "every 20 steps" quietly became
      // "every 20 steps, rounded up to the next slice boundary" — a
      // sample requested at step 20 landing at step 80 because that's
      // where the first slice happened to end. Breaking out as soon as a
      // sample is due keeps the overshoot to at most one step instead of
      // one whole slice; the sample itself still runs after the loop,
      // between slices, exactly as before.
      if (
        stopRequested ||
        (live.maxSteps > 0 && steps >= live.maxSteps) ||
        (live.sampleEvery > 0 && llm.step() >= nextSampleAt)
      ) break;
    }

    // Held-out loss, between slices for the same reason sampling is.
    if (llm.step() >= nextValidateAt) {
      nextValidateAt = nextOnGrid(llm.step(), live.validateEvery);
      try {
        const measured = await llm.validation_loss(VALIDATION_WINDOWS);
        // The comparable training number: the same number of windows,
        // drawn the same way, from the text the model does train on.
        // The per-step loss cannot play this role — 40% of training
        // windows start at a source's opening, no held-out window ever
        // does, and the model learns openings within a few hundred
        // steps. The gap that opens there is a sampling difference, not
        // memorization, and it is what made the overfitting warning fire
        // at a tenth of an epoch.
        const probe = await llm.training_probe_loss(VALIDATION_WINDOWS);
        if (measured >= 0) {
          validationLoss = measured;
          heldOut.push(measured);
          // Measured against the probe, not against the per-step loss.
          const gap = probe >= 0 ? measured - probe : null;
          if (probe >= 0) trainingProbe = probe;
          // Bits per byte rather than nats per token: comparable
          // between two vocabularies, and against a reference anybody
          // can check — gzip lands around 2.5 on English prose.
          const bpb = JSON.parse(llm.evaluate('', measured)).bitsPerByte;
          log(
            `step ${llm.step().toLocaleString()}: held-out ${measured.toFixed(4)}` +
              (probe >= 0 ? `, same-shaped training windows ${probe.toFixed(4)}` : '') +
              (gap === null ? '' : `, gap ${gap.toFixed(4)}`) +
              (smoothedLoss === null ? '' : `, per-step loss ${smoothedLoss.toFixed(4)}`) +
              (bpb > 0 ? `, ${bpb.toFixed(3)} bits per byte` : ''),
          );
          lastBitsPerByte = bpb;

          // Plateau detection — only when the Schedule setting actually
          // asks for it (see live.scheduleMode's own comment). Warmup-
          // Stable-Decay and plain Cosine already have a principled
          // answer for "when does the rate come down"; layering a
          // reactive cut on top of either is how a run's own history
          // showed the death-spiral this guards against: a shrinking
          // noise floor keeps mistaking a decelerating-but-real descent
          // for a plateau, cuts the rate, which slows the descent
          // further, shrinks the floor further, and cuts again.
          if (live.scheduleMode === 'cosine-cuts') {
            // Judged against how much this curve moves anyway, not
            // against a constant — and not at all until the model has
            // had enough training for a plateau to be a real thing
            // rather than the shape of a steep descent seen through
            // noise.
            const noise = heldOutNoise(heldOut, PLATEAU_PATIENCE);
            const trained = JSON.parse(llm.training_plan()).tokensSeen;
            if (bestSeen === null || measured < bestSeen - noise) {
              bestSeen = measured;
              sinceImprovement = 0;
            } else if (trained < TOKENS_BEFORE_CUTTING_THE_RATE) {
              // Counted, but not acted on: a run that crosses the
              // threshold mid-plateau should not have to start over.
              sinceImprovement += 1;
            } else {
              sinceImprovement += 1;
              if (sinceImprovement >= PLATEAU_PATIENCE) {
                sinceImprovement = 0;
                const before = llm.plateau_scale();
                const after = llm.decay_on_plateau(PLATEAU_FACTOR, PLATEAU_FLOOR);
                if (after < before) {
                  log(
                    `plateau: ${PLATEAU_PATIENCE} held-out measurements with no improvement past ` +
                      `${bestSeen.toFixed(4)} — cutting the learning rate to ${after.toFixed(2)}x ` +
                      `the schedule (was ${before.toFixed(2)}x)`,
                  );
                  post('train-advice', {
                    step: llm.step(),
                    advice:
                      `Held-out loss flat for ${PLATEAU_PATIENCE} measurements. Rate cut to ` +
                      `${after.toFixed(2)}x the schedule.`,
                  });
                  recordEvent(llm.step(), 'rate-cut',
                    `learning rate cut to ${after.toFixed(2)}x the schedule after ` +
                    `${PLATEAU_PATIENCE} measurements with no improvement past ` +
                    `${bestSeen.toFixed(4)} (noise floor ${noise.toFixed(4)})`);
                } else {
                  log(
                    `plateau: the learning rate is already at its floor (${after.toFixed(2)}x the ` +
                      'schedule) and held-out loss is still not improving. More text, or a ' +
                      'different model shape — not a smaller rate.',
                  );
                }
              }
            }
          }

          // The two adaptive scheduler axes ("Decay start", "Plan
          // length") — both read the current plan fresh, since either
          // may have just changed it under the other.
          {
            const plan = JSON.parse(llm.training_plan());

            // Decay start, adaptive: bring WSD's decay forward from the
            // fixed WSD_DECAY_FRACTION point to right now, the first
            // time the held-out curve stops clearing its own noise
            // floor — the same signal trainingPhase calls a plateau (or
            // worse, overfitting; either way, early cool-down is the
            // right response). Only ever moves decay earlier, never
            // later: once the fixed point arrives on its own, `step <
            // wsdDecayStart` is already false and there is nothing left
            // to do here.
            if (live.scheduleMode === 'wsd' && live.decayStartAdaptive && llm.step() < plan.wsdDecayStart) {
              const { trend, noise } = heldOutTrend(heldOut, plan.tokensSeen);
              if (trend !== null && trend < noise) {
                const fixedPoint = plan.wsdDecayStart;
                llm.set_decay_start(llm.step());
                log(
                  `adaptive cool-down: held-out loss has stopped improving — starting the decay ` +
                    `now at step ${llm.step().toLocaleString()}, ` +
                    `${(fixedPoint - llm.step()).toLocaleString()} steps ahead of the fixed point`,
                );
                post('train-advice', {
                  step: llm.step(),
                  advice:
                    'Held-out loss has plateaued — cooling down now instead of waiting for the ' +
                    'fixed point.',
                });
                recordEvent(llm.step(), 'decay-started',
                  `adaptive cool-down: decay pinned to step ${llm.step()} (the fixed point would ` +
                  `have been ${fixedPoint})`);
              }
            }

            // Plan length, adaptive: a plan that reaches its planned
            // length while still genuinely improving parks its rate at
            // the floor and stops getting anywhere — trainingPhase's own
            // "past-plan" phase exists because that happens. Extending
            // here is that phase's fix: grow the plan by
            // PLAN_EXTEND_FRACTION, capped at PLAN_EXTEND_MAX_TIMES per
            // run, so a run still earning its keep gets more room
            // instead of idling at the floor.
            if (
              live.planLengthAdaptive && plan.plannedSteps > 0 && plan.step >= plan.plannedSteps &&
              planExtensions < PLAN_EXTEND_MAX_TIMES
            ) {
              const { trend, noise } = heldOutTrend(heldOut, plan.tokensSeen);
              if (trend !== null && trend > noise) {
                const additional = Math.max(1, Math.round(plan.plannedSteps * PLAN_EXTEND_FRACTION));
                llm.extend_plan(additional);
                planExtensions += 1;
                const grownTo = JSON.parse(llm.training_plan()).plannedSteps;
                log(
                  `adaptive plan length: held-out loss is still improving past the planned ` +
                    `${plan.plannedSteps.toLocaleString()} steps — extending by ` +
                    `${additional.toLocaleString()} to ${grownTo.toLocaleString()} (extension ` +
                    `${planExtensions} of ${PLAN_EXTEND_MAX_TIMES})`,
                );
                post('train-advice', {
                  step: llm.step(),
                  advice: `Still improving at the planned length — extended to ${grownTo.toLocaleString()} steps.`,
                });
                recordEvent(llm.step(), 'plan-extended',
                  `adaptive plan length: extended from ${plan.plannedSteps} to ${grownTo} steps ` +
                  `(extension ${planExtensions} of ${PLAN_EXTEND_MAX_TIMES})`);
              }
            }
          }

          lastPhase = reportPlan({
            heldOut,
            trainingLoss: trainingProbe,
            stepsDone: steps,
            tokensPerStep,
            msPerStep: recentStepMs,
            lastPhase,
            quality: lastQuality,
            bitsPerByte: lastBitsPerByte,
          });
          // One row per measurement, with everything that was true at
          // that moment. This is the record a run can be read back
          // from — and the reason it carries the settings as well as
          // the results is that a curve without them cannot be
          // explained after the fact.
          {
            const plan = JSON.parse(llm.training_plan());
            post('train-record', {
              runId,
              kind: 'measurement',
              at: Date.now(),
              step: plan.step,
              runStep: Math.max(0, plan.step - plan.startStep),
              tokensSeen: plan.tokensSeen,
              epochs: plan.trainingTokens > 0 ? plan.tokensSeen / plan.trainingTokens : 0,
              loss: smoothedLoss,
              probe: trainingProbe,
              heldOut: measured,
              gap,
              bitsPerByte: bpb,
              lr: plan.lrNow,
              plateauScale: plan.plateauScale,
              gradNorm: lastGradNorm,
              tokensPerSecond: recentStepMs > 0 ? tokensPerStep / (recentStepMs / 1000) : 0,
              msPerStep: recentStepMs,
              elapsedSeconds: (performance.now() - startedAt) / 1000,
              batchSize: live.batchSize,
              tokensPerStep,
              phase: lastPhase,
              quality: lastQuality,
              // 'wsd' | 'cosine-cuts' | 'cosine' — see the Settings tab's
              // Scheduler panel. Recorded per row, not just in the
              // run-started event, so a row from after a schedule switch
              // mid-run (this session's own cosine-decay test, for one)
              // says which mode it was measured under instead of leaving
              // that only inferable from the event log.
              scheduleMode: live.scheduleMode,
            });
          }
        }
      } catch (error) {
        log(`held-out loss failed: ${(error && error.message) || error}`);
      }
    }

    // Write the model down, between slices like everything else that
    // needs the GPU to itself.
    //
    // This used to be driven from the page's progress handler, which
    // fires every 250 ms *during* a slice. Exporting takes the same
    // `busy` guard a training step takes, so the two collided: the
    // export won, the next `train_step` was refused, and the run died
    // on the spot — leaving the sample card frozen on whatever it had
    // last produced, which looks exactly like a model that stopped
    // getting better.
    if (llm.step() >= nextAutosaveAt) {
      nextAutosaveAt = nextOnGrid(llm.step(), live.autosaveEvery);
      const savedAt = performance.now();
      try {
        const bytes = await llm.export_checkpoint();
        post('train-autosave', {
          step: llm.step(),
          bytes: bytes.buffer,
          byteLength: bytes.length,
        }, [bytes.buffer]);
        log(`auto-saved at step ${llm.step().toLocaleString()} in ` +
          `${(performance.now() - savedAt).toFixed(0)} ms`);
      } catch (error) {
        // A failed save must never take the run with it.
        log(`auto-save failed: ${(error && error.message) || error}`);
      }
    }

    // Between slices, never inside one: sampling takes as long as it
    // takes and shouldn't be counted against a slice's time budget.
    // Keyed on the model's own step count, so the interval means the
    // same thing across stop/resume — and on the same absolute grid
    // nextOnGrid keeps it on, so a restart doesn't shift it either.
    if (live.sampleEvery > 0 && llm.step() >= nextSampleAt) {
      nextSampleAt = nextOnGrid(llm.step(), live.sampleEvery);
      if (inferenceDevice === 'cpu') {
        // CPU generation never pulls weights off the GPU on its own (see
        // WasmLLM::generate) - that's what makes it race-free, but left
        // alone it means a training sample would read whatever was last
        // synced, which for most of a run is nothing: the CPU side sits
        // at its initial random weights the entire time, and a sample
        // "shows progress" that never moves no matter how far training
        // gets. A quick sync first, best effort - if training is mid-step
        // (rare, between slices) this is simply skipped for this one
        // sample rather than waiting for it - then the generation itself
        // still runs unawaited, so it never pauses training.
        llm.sync_from_gpu().catch(() => {}).finally(() => {
          runTrainingSample(live.samplePrompt, live.sampleMaxTokens, live.sampling).catch((error) => {
            log(`sample failed: ${(error && error.message) || error}`);
          });
        });
      } else {
        // GPU sampling runs on the same device training does, so it has
        // to serialize with it — awaited here, between slices, never
        // inside one.
        try {
          await runTrainingSample(live.samplePrompt, live.sampleMaxTokens, live.sampling);
        } catch (error) {
          log(`sample failed: ${(error && error.message) || error}`);
        }
      }
    }

    // ...then hand the machine back. Always yield, even at full effort
    // where the pause is zero: this is the only point in the loop where
    // the worker's message queue gets a turn, so skipping it means a
    // `stop` message sits unread until training ends on its own.
    await new Promise((resolve) => setTimeout(resolve, live.pauseMs));
  }

  training = false;
  const elapsedSeconds = (performance.now() - startedAt) / 1000;
  log(
    `training run finished: ${steps} steps in ${elapsedSeconds.toFixed(1)} s ` +
      `(${(tokens / Math.max(elapsedSeconds, 1e-6)).toFixed(0)} tok/s overall, ` +
      `${((elapsedSeconds * 1000) / Math.max(steps, 1)).toFixed(0)} ms/step), ` +
      `loss ${smoothedLoss === null ? '—' : smoothedLoss.toFixed(4)}, ` +
      `reason ${stopRequested ? 'stopped' : 'done'}`,
  );
  recordEvent(llm.step(), 'run-ended',
    `run ${stopRequested ? 'stopped' : 'finished'} after ${steps.toLocaleString()} steps in ` +
    `${elapsedSeconds.toFixed(0)}s, ${(tokens / Math.max(elapsedSeconds, 1e-6)).toFixed(0)} ` +
    `tok/s overall, loss ${smoothedLoss === null ? '—' : smoothedLoss.toFixed(4)}`);
  return {
    steps,
    loss: smoothedLoss,
    stopReason: stopRequested ? 'stopped' : 'done',
    elapsedSeconds,
  };
}

/// A lost or reset device is the one failure worth naming precisely: it
/// means a submission ran past the driver's watchdog, and the answer is a
/// smaller batch or a shorter context, not a retry.
function describeTrainingFailure(error) {
  const message = (error && error.message) || String(error);
  if (/device.*(lost|hung|removed|reset)|DXGI_ERROR|GPUDevice/i.test(message)) {
    return (
      `the GPU device was reset mid-step (${message}). Lower the batch size or the context ` +
      'length and reload the page.'
    );
  }
  return message;
}

const handlers = {
  async 'load-model'({ bytes }) {
    return loadModelBytes(bytes);
  },

  async 'create-model'({ layers, uniqueLayers, hidden, heads, kvHeads, contextLen, window: attentionWindow, seed }) {
    await ensureWasm();
    log('creating model', { layers, uniqueLayers, hidden, heads, kvHeads, contextLen, window: attentionWindow });
    llm = new wasm.WasmLLM(
      layers,
      uniqueLayers || layers,
      hidden,
      heads,
      kvHeads,
      contextLen,
      attentionWindow,
      seed,
    );
    log('model created; asking for a GPU device');
    await initGpu();
    return describeModel();
  },

  async 'import-checkpoint'({ bytes }) {
    if (training) return { error: 'a training run is in flight - press Stop, then import a checkpoint' };
    await ensureWasm();
    if (!llm) {
      llm = wasm.WasmLLM.from_checkpoint(new Uint8Array(bytes));
    } else {
      llm.import_checkpoint(new Uint8Array(bytes));
    }
    await initGpu();
    return describeModel();
  },

  async 'export-checkpoint'() {
    const bytes = await llm.export_checkpoint();
    return { bytes: bytes.buffer, byteLength: bytes.length };
  },

  /// The Adam moment buffers, for the page to store beside the
  /// checkpoint. Weights alone resume with the momentum reset, and the
  /// loss jumps every time.
  async 'export-optimizer'() {
    const bytes = await llm.export_optimizer();
    return { bytes: bytes.buffer, byteLength: bytes.length };
  },

  async 'import-optimizer'({ bytes }) {
    await llm.import_optimizer(new Uint8Array(bytes));
    return { restored: true };
  },

  async 'model-info'() {
    return describeModel();
  },

  async 'parse-prompt'({ prompt }) {
    await ensureWasm();
    return describePrompt(text(prompt, 'the prompt'));
  },

  /// What a shape would cost, before anything is built from it. Needs no
  /// model — that is the point, since this is what somebody is looking
  /// at while they choose the numbers.
  async 'describe-shape'({ layers, uniqueLayers, hidden, heads, kvHeads, contextLen, window: win, corpusChars }) {
    await ensureWasm();
    return JSON.parse(
      wasm.describe_shape(
        layers,
        uniqueLayers || layers,
        hidden,
        heads,
        kvHeads,
        contextLen,
        win,
        corpusChars || 0,
      ),
    );
  },

  /// Which loaded sources are copies of another. Reported, never
  /// removed: which copy to keep is the user's call.
  async 'duplicate-sources'() {
    return { ids: llm.duplicate_sources() };
  },

  /// Per-source token counts and how many training windows have actually
  /// been drawn from each — for the Overview tab's corpus breakdown.
  /// Read-only, no GPU touch.
  async 'corpus-source-stats'() {
    return { sources: JSON.parse(llm.corpus_source_stats()) };
  },

  /// Restore a source's sample count after a fresh page load re-upserts
  /// it into a new corpus — the count is persisted per-source in
  /// SOURCES_STORE and handed back in here, once per source, right after
  /// `upsert-source` on the same source.
  async 'set-source-sample-count'({ id, count }) {
    llm.set_source_sample_count(id, count || 0);
    return {};
  },

  /// One source's progress through its own pass over its training
  /// windows, so the page can persist it back to SOURCES_STORE — see
  /// `set-window-progress` for the other half of the round trip.
  async 'window-progress'({ id }) {
    return { progress: JSON.parse(llm.window_progress(id)) };
  },

  /// Every source's window-pass progress that exists yet, in one round
  /// trip — for periodically flushing it all back to SOURCES_STORE (see
  /// `flushSourceSampleCounts` in app.js) rather than one call per
  /// source.
  async 'corpus-window-progress'() {
    return { sources: JSON.parse(llm.corpus_window_progress()) };
  },

  /// Restore a source's window-pass progress after a fresh page load
  /// re-upserts it into a new corpus, the same way
  /// `set-source-sample-count` restores the sample count — without this,
  /// every reload restarts that source's pass from its first window
  /// instead of continuing where the last session left off.
  async 'set-window-progress'({ id, epoch, cursor }) {
    llm.set_window_progress(id, epoch || 0, cursor || 0);
    return {};
  },

  async 'upsert-source'(payload) {
    // More text is a different problem from the one the last plateau
    // was found on: whatever cut the rate then does not apply now.
    llm.reset_plateau_scale();
    const stats = llm.upsert_source(
      text(payload.id, 'the source id'),
      text(payload.text, `the text of source ${payload.id}`),
      !!payload.isHtml,
    );
    return {
      charCount: stats.char_count,
      byteCount: stats.byte_count,
      tokenCount: stats.token_count,
      model: describeModel(),
    };
  },

  async 'remove-source'({ id }) {
    llm.reset_plateau_scale();
    llm.remove_source(text(id, 'the source id'));
    return describeModel();
  },

  async generate(payload) {
    const extraContext = payload.extraContext || '';
    const result = await generate({
      ...payload,
      prompt: text(payload.prompt, 'the prompt'),
      extraContext,
    });
    const parsed = describePrompt(text(payload.prompt, 'the prompt'));
    result.notes = llm.qa_check(result.text || '', parsed.targetWords || 0);
    return result;
  },

  /// Learn a BPE vocabulary from the loaded sources, then rebuild the
  /// (untrained) model and its GPU state around it.
  async 'profile-kernels'({ reps = 20 }) {
    if (!llm.has_gpu()) return { error: 'no GPU device' };
    if (training) return { error: 'a training run is in flight - press Stop, then profile' };
    const rows = JSON.parse(await llm.profile_kernels(reps));
    rows.sort((a, b) => b.msPerStep - a.msPerStep);
    for (const row of rows) {
      log(
        `kernel ${row.kernel.padEnd(28)} ${row.msEach.toFixed(3)} ms each ` +
          `x${row.perStep} per sequence = ${row.msPerStep.toFixed(1)} ms`,
      );
    }
    return { rows };
  },

  async 'learn-vocabulary'({ maxVocabSize } = {}) {
    if (training) return { error: 'a training run is in flight - press Stop, then learn a vocabulary' };
    // No UI control owns this ceiling; the wasm side exports the one it
    // and its own shape-estimate agree on, rather than the page
    // re-declaring the same number as a fallback here.
    const cap = maxVocabSize > 0 ? maxVocabSize : wasm.max_vocab_size();
    const before = llm.vocab_size();
    const started = performance.now();
    const size = llm.learn_vocabulary(cap);
    if (size === before) {
      log(`vocabulary unchanged (${size} tokens) - a trained model keeps the one it learned with`);
      return { vocabSize: size, changed: false };
    }
    log(
      `learned a ${size}-token vocabulary from your sources in ` +
        `${(performance.now() - started).toFixed(0)} ms (was ${before}); ` +
        'the model was rebuilt around it',
    );
    await initGpu();
    return { vocabSize: size, changed: true, model: describeModel() };
  },

  /// Time one step per phase, at a few command-buffer sizes. This is the
  /// measurement that says whether a step is bound by arithmetic or by
  /// per-submission cost, instead of anybody guessing.
  async profile({ batchSize = 2 }) {
    if (!llm.has_gpu()) return { error: 'no GPU device' };
    if (training) {
      const message = 'a training run is in flight — press Stop, then profile';
      log(`profile refused: ${message}`);
      return { error: message };
    }
    const rows = [];
    for (const perSubmit of [4, 16, 64, 256]) {
      const report = JSON.parse(await llm.profile_step(batchSize, perSubmit));
      rows.push(report);
      log(
        `profile: ${report.dispatchesPerSubmit} dispatches/submit -> ` +
          `${report.totalMs.toFixed(0)} ms total ` +
          `(${report.submits} submits) | zero ${report.zeroMs.toFixed(0)} ` +
          `forward ${report.forwardMs.toFixed(0)} loss ${report.lossMs.toFixed(0)} ` +
          `backward ${report.backwardMs.toFixed(0)} reduce ${report.reduceMs.toFixed(0)} ` +
          `readback ${report.readbackMs.toFixed(0)} adam ${report.adamMs.toFixed(0)}`,
      );
    }
    return { rows };
  },

  /// Measure a piece of text against the corpus: how much of it is
  /// words this corpus uses, how much of it repeats itself, and — given
  /// a loss — how many bits per byte that is.
  async evaluate({ text = '', loss = -1 }) {
    return JSON.parse(llm.evaluate(text, loss));
  },

  /// The plan as it stands right now, without training anything: which
  /// phase the model is in, and what would help. The page asks for this
  /// whenever the corpus or the model changes, so the advice is there
  /// before the first step rather than after the first validation.
  async 'training-plan'({ batchSize = 1 } = {}) {
    const info = llm.info();
    return buildPlan({
      heldOut: [],
      trainingLoss: null,
      stepsDone: 0,
      tokensPerStep: Math.max(1, batchSize) * info.context_len,
      msPerStep: null,
      lastPhase: null,
    });
  },

  /// Time this machine and return the settings it wants. The page
  /// stores the result and hands it back on the next visit.
  async benchmark(payload = {}) {
    if (!llm.has_gpu()) return { error: 'no GPU device' };
    if (training) {
      const message = 'a training run is in flight — press Stop, then benchmark';
      log(`benchmark refused: ${message}`);
      return { error: message };
    }
    if (!llm.can_train()) {
      return { error: 'not enough source text to fill one context window' };
    }
    return { profile: await benchmark(payload) };
  },

  /// Apply a stored profile's command-buffer size without re-measuring.
  async 'apply-machine-profile'({ dispatchesPerSubmit }) {
    if (dispatchesPerSubmit > 0) llm.set_dispatches_per_submit(dispatchesPerSubmit);
    const applied = llm.dispatches_per_submit();
    log(`machine profile applied: ${applied} dispatches per command buffer`);
    return { dispatchesPerSubmit: applied };
  },

  /// Which device the next Generate call prefers. Takes effect
  /// immediately — the very next call to `generate`, nothing queued or
  /// deferred.
  async 'set-inference-device'({ device }) {
    inferenceDevice = device === 'cpu' ? 'cpu' : 'gpu';
    log(`inference device set to ${inferenceDevice}`);
    return { device: inferenceDevice };
  },

  async train(payload) {
    // One run at a time, checked here rather than only on the page.
    //
    // The page has its own flag, but it is a page: it reloads, it gets
    // clicked twice, its flag can be cleared a moment before the run
    // actually ends. Two loops then share one model — both calling
    // train_step, both advancing the same step counter, both writing
    // history under different run ids — which is how a history ends up
    // with two "run started" events at the same step and a progress
    // line that disagrees with the table above it.
    if (training) {
      log('refused to start a second training run: one is already in flight');
      return { steps: 0, stopReason: 'already-training', elapsedSeconds: 0 };
    }
    // Training is GPU work. Without a device there is nothing to fall
    // back to, so say which of the two reasons stopped it.
    if (!llm.has_gpu()) {
      log('cannot train: no WebGPU device. Training has no CPU path.');
      return { steps: 0, stopReason: 'no-gpu', elapsedSeconds: 0 };
    }
    if (!llm.can_train()) {
      log('cannot train: not enough source text to fill one context window.');
      return { steps: 0, stopReason: 'no-data', elapsedSeconds: 0 };
    }
    try {
      const result = await train(payload);
      result.model = describeModel();
      return result;
    } catch (error) {
      training = false;
      const explained = describeTrainingFailure(error);
      log(`training failed: ${explained}`);
      throw new Error(explained);
    }
  },

  /// Apply a Settings/Inference-tab change to the run in flight, whether
  /// one is going or not — every field `applyLiveSettings` understands,
  /// through the one function `train()` itself seeds from. Harmless
  /// before any run exists: it just updates `live` (and, for the fields
  /// the model owns — peak rate, boundary-sample rate, planned steps —
  /// the model itself), which the next run also starts from.
  ///
  /// The model is the source of truth for anything it owns, so the
  /// reply reads it back rather than echoing what was sent — a value
  /// the model clamped or refused shows up here as what actually took,
  /// not as an intent that was merely logged.
  async 'update-training-settings'(settings) {
    applyLiveSettings(settings);
    const plan = JSON.parse(llm.training_plan());
    return {
      peakLr: plan.peakLr,
      lrNow: plan.lrNow,
      plannedSteps: plan.plannedSteps,
      boundarySampleRate: llm.boundary_sample_rate(),
      batchSize: live.batchSize,
      maxSteps: live.maxSteps,
      effort: live.effort,
    };
  },

  /// Put the schedule back to full strength after a plateau cut.
  ///
  /// A cut is a guess made from a curve, and a guess can be wrong — the
  /// detector that made it has been wrong, on a noisy measurement, at
  /// four-fifths of a pass. Until now the only way to undo one was to
  /// add or remove a source, which resets it as a side effect. That is
  /// not a control, it is a trick.
  async 'reset-schedule'() {
    const before = llm.plateau_scale();
    llm.reset_plateau_scale();
    const plan = JSON.parse(llm.training_plan());
    log(
      `schedule restored to full strength (was ${before.toFixed(2)}x); the rate in force is ` +
        `now ${plan.lrNow.toExponential(2)}`,
    );
    if (before < 1) {
      recordEvent(plan.step, 'schedule-restored',
        `plateau cut undone by hand: ${before.toFixed(2)}x back to 1.00x, rate in force now ` +
        `${plan.lrNow.toExponential(2)}`);
    }
    return { plateauScale: plan.plateauScale, lrNow: plan.lrNow, was: before };
  },

  async stop() {
    stopRequested = true;
    return { stopping: true };
  },
};

self.onmessage = async (event) => {
  // `rid` is the request id and `payload` is the message's own data.
  // They are separate fields because they used to be spread into one
  // object, where a payload key called `id` silently replaced the
  // request id.
  const { rid, type, payload = {} } = event.data || {};
  const handler = handlers[type];
  if (!handler) {
    fail(rid, new Error(`unknown message ${type}`));
    return;
  }
  // Everything except `stop` needs a model; saying so beats a wasm panic.
  if (!llm && !['load-model', 'create-model', 'import-checkpoint', 'parse-prompt', 'describe-shape', 'set-inference-device', 'stop'].includes(type)) {
    fail(
      rid,
      new Error(
        type === 'profile' || type === 'benchmark'
          ? `${type === 'profile' ? 'profiling' : 'benchmarking'} needs a model: press Train ` +
            'first (a model lives in this tab only, so a reload leaves none), then run it again'
          : 'no model loaded yet',
      ),
    );
    return;
  }
  try {
    const result = await handler(payload);
    if (result && result.bytes instanceof ArrayBuffer) {
      // Transfer rather than copy: an exported checkpoint is tens of MB.
      self.postMessage({ type: 'result', rid, result }, [result.bytes]);
    } else {
      post('result', { rid, result });
    }
  } catch (error) {
    fail(rid, error);
  }
};
