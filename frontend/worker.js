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
/// What a cut multiplies the rate by, and how far the cuts may go in
/// total. Halving is the standard move; a run that has fallen to a
/// twentieth of its schedule has a problem another cut will not fix.
const PLATEAU_FACTOR = 0.5;
const PLATEAU_FLOOR = 0.05;
/// How much better a measurement has to be to count as better at all.
/// Held-out loss wobbles by a few thousandths between measurements on a
/// corpus this size; without a threshold the patience counter resets on
/// noise and the rate is never cut.
const PLATEAU_MIN_DELTA = 0.005;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
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

async function generate({ prompt, extraContext, temperature, topK, topP, minP, repetitionPenalty, seed }) {
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
async function trainingSample(prompt, words) {
  const result = await llm.generate(
    prompt,
    '',
    0.9,
    40,
    0.95,
    // A training sample is the one place min-p earns its keep without
    // being asked for: an early model's distribution is nearly flat, so
    // this keeps the field wide, and a later one's is peaked, so it
    // stops the sample wandering into the tail.
    0.05,
    1.1,
    Math.floor(Math.random() * 1e9),
    (_piece, produced) => produced < words,
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

/// Roughly how many tokens a model of this size wants to see. The
/// Chinchilla result is ~20 tokens per parameter for a compute-optimal
/// run; it is a rule of thumb rather than a law, and it is the only
/// honest yardstick for "is this corpus big enough for this model".
const TOKENS_PER_PARAM = 20;

/// Which phase of a run this is, and what that means for what you are
/// looking at.
///
/// The phases are not decoration. "Loss is barely moving" means one
/// thing in the first fifty steps, when the learning rate is still a
/// fraction of its peak, and the opposite thing two thousand steps
/// later. Saying which is which is the difference between waiting and
/// wasting an afternoon.
function trainingPhase(plan, { heldOut, trainingLoss, stepsDone }) {
  const { step, plannedSteps, warmupSteps, peakLr, lrNow, minLrRatio } = plan;
  if (step < warmupSteps) {
    return {
      key: 'warm-up',
      title: 'Warm-up',
      detail:
        `the learning rate is ramping from nearly zero to ${peakLr.toExponential(1)} over the ` +
        `first ${warmupSteps.toLocaleString()} steps — it is at ${lrNow.toExponential(1)} now. ` +
        'Loss moves slowly here on purpose: a full-size step this early puts the model ' +
        'somewhere it spends thousands of steps climbing out of.',
    };
  }

  // Trend over the held-out curve, on the same window the corpus advice
  // uses, so the two never disagree about which way the line is going.
  const WINDOW = 5;
  let trend = null;
  if (heldOut.length >= WINDOW * 2) {
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    trend = mean(heldOut.slice(-WINDOW * 2, -WINDOW)) - mean(heldOut.slice(-WINDOW));
  }
  const gap = trainingLoss === null || heldOut.length === 0
    ? null
    : heldOut[heldOut.length - 1] - trainingLoss;

  if (trend !== null && trend < -0.01) {
    return {
      key: 'overfitting',
      title: 'Overfitting',
      detail:
        'held-out loss is rising while training loss falls. From here the model is learning ' +
        'your text rather than learning from it, and every further step makes it worse at ' +
        'writing anything new. The best model of this run is already saved.',
    };
  }
  if (trend !== null && trend < 0.01) {
    return {
      key: 'plateau',
      title: 'Plateau',
      detail:
        'held-out loss has stopped improving' +
        (gap === null ? '' : `, and sits ${gap.toFixed(2)} above training loss`) +
        '. More steps at this setting will not move it; something has to change.',
    };
  }
  // The cosine tail: the last stretch of a planned run, where the rate
  // is close to its floor and the model is settling rather than moving.
  if (plannedSteps > 0 && lrNow <= peakLr * (minLrRatio + 0.05)) {
    return {
      key: 'cooling',
      title: 'Cooling down',
      detail:
        `the cosine schedule has taken the rate down to ${lrNow.toExponential(1)}, near its ` +
        `floor of ${(peakLr * minLrRatio).toExponential(1)}. This is where a run consolidates: ` +
        'small improvements, and the least noisy weights it will have.',
    };
  }
  return {
    key: 'learning',
    title: 'Learning',
    detail:
      `past warm-up, rate ${lrNow.toExponential(1)}` +
      (plan.plateauScale < 1
        ? ` (cut to ${plan.plateauScale.toFixed(2)}x the schedule after a plateau)`
        : '') +
      ', held-out loss still improving' +
      (stepsDone > 0 ? ` — ${stepsDone.toLocaleString()} steps into this run.` : '.'),
  };
}

/// What a person could actually do about it, most useful first.
///
/// Every entry names a number and an action. "Add more text" on its own
/// is not advice; "this corpus is 6.5M tokens for a model that wants
/// 283M — roughly 40 more scripts" is.
function planActions(plan, phase, { heldOut, trainingLoss, tokensPerStep, tokensSeen }) {
  // Measurements taken from generated text, folded onto the plan object
  // by buildPlan so this reads them from one place.
  const actions = [];
  const { params, trainingTokens, validationTokens, contextLen, mix } = plan;
  const wanted = Math.round(params * TOKENS_PER_PARAM);
  const round = (n) => Math.round(n).toLocaleString();

  if (validationTokens < contextLen + 1) {
    actions.push({
      key: 'no-validation',
      urgency: 'high',
      text:
        'There is not enough text to hold any of it out, so nothing here can tell learning ' +
        'from memorizing. Add sources until the corpus is comfortably past ' +
        `${round((contextLen * 20) / 0.05)} tokens.`,
    });
  }

  if (trainingTokens > 0 && trainingTokens < wanted) {
    const ratio = trainingTokens / params;
    // Sized from what is already loaded rather than from an assumed
    // script length: their scripts are the right unit of "more".
    const perSource = plan.sources > 0 ? plan.corpusTokens / plan.sources : 0;
    const moreSources = perSource > 0 ? Math.ceil((wanted - trainingTokens) / perSource) : 0;
    actions.push({
      key: 'corpus-size',
      urgency: phase.key === 'overfitting' || phase.key === 'plateau' ? 'high' : 'normal',
      text:
        `${round(trainingTokens)} training tokens for ${round(params)} parameters is ` +
        `${ratio.toFixed(1)} tokens per parameter; the rule of thumb is ${TOKENS_PER_PARAM}, so ` +
        `this model wants about ${round(wanted)}. Either add roughly ${round(moreSources)} more ` +
        `sources the size of the ones already loaded, or build a smaller model — ` +
        `${round(trainingTokens / TOKENS_PER_PARAM)} parameters is what this corpus supports.`,
    });
  }

  // What kind of text is missing. A corpus that is all one thing teaches
  // the shape of that thing: thirty scripts teach a model that every
  // paragraph is one line of dialogue long.
  if (Array.isArray(mix) && mix.length > 0 && plan.corpusTokens > 0) {
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
            `${Math.round(share * 100)}% of your corpus is ${largest.label}. A model trained on ` +
            'one shape of writing learns that shape and not the language underneath it. ' +
            `Adding ${missing.slice(0, 2).join(' or ')} would widen what it can write.`,
        });
      }
    }
  }

  if (phase.key === 'overfitting') {
    actions.push({
      key: 'stop-here',
      urgency: 'high',
      text:
        'Stop this run and go back to the best model, or add text and keep training. ' +
        'Continuing without either only makes it worse.',
    });
  }
  if (plan.plateauScale <= 0.06) {
    actions.push({
      key: 'plateau-floor',
      urgency: 'high',
      text:
        `The learning rate has been cut to ${plan.plateauScale.toFixed(2)}x the schedule and ` +
        'held-out loss still is not improving. Cutting it further is not the answer: at this ' +
        'point the limit is the corpus or the shape of the model, not the rate.',
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
        ? 'Held-out sits well above training loss, which is the signature of too little text ' +
          'for this model, not of too few steps.'
        : 'Both curves have flattened together, which is the signature of a model too small ' +
          'for the text. A larger hidden size or another layer would do more than more steps.',
    });
  }

  // What the samples themselves say. Loss can fall for a long time
  // while the output is still not words, and that is exactly the
  // situation where somebody stares at a falling curve and wonders why
  // nothing reads like English.
  const q = plan.quality;
  if (q && q.words >= 20) {
    if (q.knownWordRate < 0.75) {
      actions.push({
        key: 'not-words-yet',
        urgency: 'normal',
        text:
          `Only ${Math.round(q.knownWordRate * 100)}% of the words in the last sample appear ` +
          'anywhere in your corpus — the model is still assembling letters rather than ' +
          'recalling words' +
          (q.unknownExamples && q.unknownExamples.length
            ? ` (${q.unknownExamples.slice(0, 4).join(', ')})`
            : '') +
          '. That is normal early and it is the first thing that should improve; if the loss ' +
          'is falling and this is not, the tokenizer or the corpus is the problem, not the ' +
          'number of steps.',
      });
    }
    if (q.repeated4gramRate > 0.25) {
      actions.push({
        key: 'repeating',
        urgency: 'normal',
        text:
          `${Math.round(q.repeated4gramRate * 100)}% of the four-word runs in the last sample ` +
          'had already appeared in it. The model is cycling rather than continuing. Raise the ' +
          'repetition penalty or min-p in the sampling settings before concluding anything ' +
          'about the training.',
      });
    }
  }

  // How many times the run has been over the same text. Past a few
  // passes a small corpus is being memorized whatever the loss says.
  if (trainingTokens > 0 && tokensSeen > 0) {
    const epochs = tokensSeen / trainingTokens;
    if (epochs >= 2) {
      actions.push({
        key: 'epochs',
        urgency: epochs >= 4 ? 'high' : 'normal',
        text:
          `This model has been over your text ${epochs.toFixed(1)} times. Repeated passes are ` +
          'how a small corpus gets memorized; each further pass buys less than the last.',
      });
    }
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
  // Tokens this model has seen over its whole life, not just this run.
  // Approximate on purpose: earlier runs may have used a different batch
  // size, and the step count is all that survives them.
  const tokensSeen = plan.step * tokensPerStep;
  const remaining = plan.plannedSteps > plan.step ? plan.plannedSteps - plan.step : 0;
  return {
    phase,
    actions: planActions(plan, phase, { ...state, tokensPerStep, tokensSeen }),
    numbers: {
      step: plan.step,
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
  }
  post('train-plan', plan);
  return plan.phase.key;
}

async function train({ batchSize, learningRate, maxSteps, effort, sampleEvery, samplePrompt, sampleWords }) {
  stopRequested = false;
  training = true;
  if (learningRate > 0) llm.set_learning_rate(learningRate);
  // The schedule has to know how long the run is, or its warmup and its
  // cosine decay are shaped for a run nobody asked for.
  llm.set_planned_steps(maxSteps > 0 ? maxSteps : 2000);

  const info = llm.info();
  log('training run starting', {
    device: llm.device_summary(),
    softwareRenderer: llm.gpu_is_software(),
    batchSize,
    contextLen: info.context_len,
    tokensPerStep: batchSize * info.context_len,
    dispatchesPerSubmit: llm.dispatches_per_submit(),
    maxSteps: maxSteps || 'until stopped',
    effort,
    learningRate: learningRate > 0 ? learningRate : 'automatic',
    plateauScale: llm.plateau_scale(),
    params: info.params,
  });

  const sliceMs = 120;
  const pauseMs = Math.max(0, Math.round(sliceMs * (1 - effort) / Math.max(effort, 0.05)));
  const tokensPerStep = batchSize * info.context_len;
  const startedAt = performance.now();
  let steps = 0;
  let tokens = 0;
  let lastPost = 0;
  let smoothedLoss = null;
  // First sample after the first interval, not immediately: a sample at
  // step 0 is noise from an untouched model.
  let nextSampleAt = llm.step() + (sampleEvery || 0);
  // Held-out loss on the same cadence as the loss chart's own reporting:
  // often enough to see the two curves separate, rare enough that its
  // forward passes are a rounding error against training.
  const validateEvery = 25;
  let nextValidateAt = llm.step() + validateEvery;
  let validationLoss = null;
  // Held-out losses in order, so the run can say when more text would
  // help more than more steps.
  const heldOut = [];
  let lastAdvice = null;
  let bestValidation = null;
  // The phase the run was last seen in, so a change of phase is
  // announced once instead of on every recomputation.
  let lastPhase = null;
  /// The last quality measurement taken from a generated sample, so the
  /// plan can carry it between samples instead of only at the moment one
  /// is produced.
  let lastQuality = null;
  let lastBitsPerByte = 0;
  // Held-out measurements since the last one that was actually better.
  let sinceImprovement = 0;
  let bestSeen = null;
  // Median-ish step cost, for the estimate of how long the rest takes.
  let recentStepMs = null;

  // Say where the run is starting from before it starts: which phase,
  // how much text there is for a model this size, what would help.
  lastPhase = reportPlan({
    heldOut, trainingLoss: null, stepsDone: 0, tokensPerStep, msPerStep: null, lastPhase: null,
  });

  while (!stopRequested && (maxSteps <= 0 || steps < maxSteps)) {
    const sliceStart = performance.now();
    // Work for a slice...
    while (performance.now() - sliceStart < sliceMs) {
      const stepStart = performance.now();
      const report = await llm.train_step(batchSize);
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
          lr: report.lr,
          gradNorm: report.grad_norm,
          elapsedSeconds: elapsed,
          tokensPerSecond: elapsed > 0 ? tokens / elapsed : 0,
          fractionDone: maxSteps > 0 ? steps / maxSteps : 0,
        });
      }
      if (stopRequested || (maxSteps > 0 && steps >= maxSteps)) break;
    }

    // Held-out loss, between slices for the same reason sampling is.
    if (llm.step() >= nextValidateAt) {
      nextValidateAt = llm.step() + validateEvery;
      try {
        const measured = await llm.validation_loss(batchSize);
        if (measured >= 0) {
          validationLoss = measured;
          heldOut.push(measured);
          const gap = smoothedLoss === null ? null : measured - smoothedLoss;
          // Bits per byte rather than nats per token: comparable
          // between two vocabularies, and against a reference anybody
          // can check — gzip lands around 2.5 on English prose.
          const bpb = JSON.parse(llm.evaluate('', measured)).bitsPerByte;
          log(
            `step ${llm.step().toLocaleString()}: held-out loss ${measured.toFixed(4)}` +
              (gap === null ? '' : ` (training ${smoothedLoss.toFixed(4)}, gap ${gap.toFixed(4)})`) +
              (bpb > 0 ? `, ${bpb.toFixed(3)} bits per byte` : ''),
          );
          lastBitsPerByte = bpb;

          // Plateau detection. A cosine schedule decays on a plan; it
          // has no idea whether the run is following it. When held-out
          // loss stops improving, the usual cause is steps too large to
          // settle into the minimum the model is circling, and the
          // usual answer is to cut the rate and let it.
          if (bestSeen === null || measured < bestSeen - PLATEAU_MIN_DELTA) {
            bestSeen = measured;
            sinceImprovement = 0;
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
                    `held-out loss has not improved in ${PLATEAU_PATIENCE} measurements, so the ` +
                    `learning rate has been cut to ${after.toFixed(2)}x the schedule. If it ` +
                    'does not start improving again, the corpus is the limit, not the rate.',
                });
              } else {
                log(
                  `plateau: the learning rate is already at its floor (${after.toFixed(2)}x the ` +
                    'schedule) and held-out loss is still not improving. More text, or a ' +
                    'different model shape — not a smaller rate.',
                );
              }
            }
          }
          // A run's best model is rarely its last, and training past the
          // best is exactly what a small corpus makes it do. Tell the
          // page whenever this is the best held-out loss so far so it can
          // keep a copy.
          if (bestValidation === null || measured < bestValidation) {
            bestValidation = measured;
            post('train-best', { step: llm.step(), validationLoss: measured });
          }
          lastPhase = reportPlan({
            heldOut,
            trainingLoss: smoothedLoss,
            stepsDone: steps,
            tokensPerStep,
            msPerStep: recentStepMs,
            lastPhase,
            quality: lastQuality,
            bitsPerByte: lastBitsPerByte,
          });
          const advice = corpusAdvice(heldOut, smoothedLoss);
          if (advice && advice !== lastAdvice) {
            lastAdvice = advice;
            log(`advice: ${advice}`);
            post('train-advice', { advice, step: llm.step() });
          }
        }
      } catch (error) {
        log(`held-out loss failed: ${(error && error.message) || error}`);
      }
    }

    // Between slices, never inside one: sampling takes as long as it
    // takes and shouldn't be counted against a slice's time budget.
    // Keyed on the model's own step count, so the interval means the
    // same thing across stop/resume.
    if (sampleEvery > 0 && llm.step() >= nextSampleAt) {
      nextSampleAt = llm.step() + sampleEvery;
      const sampleStart = performance.now();
      try {
        const text = await trainingSample(samplePrompt, sampleWords || 40);
        // What the loss curve cannot say: is this English, and is it
        // still saying anything new. Measured against the user's own
        // corpus, so no word list has to ship with the page.
        const quality = JSON.parse(llm.evaluate(text, validationLoss === null ? -1 : validationLoss));
        lastQuality = quality;
        post('train-sample', { step: llm.step(), loss: smoothedLoss, text, quality });
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
      } catch (error) {
        log(`sample failed: ${(error && error.message) || error}`);
        post('train-sample', {
          step: llm.step(),
          loss: smoothedLoss,
          text: `(sample failed: ${(error && error.message) || error})`,
        });
      }
    }

    // ...then hand the machine back. Always yield, even at full effort
    // where the pause is zero: this is the only point in the loop where
    // the worker's message queue gets a turn, so skipping it means a
    // `stop` message sits unread until training ends on its own.
    await new Promise((resolve) => setTimeout(resolve, pauseMs));
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
  return {
    steps,
    loss: smoothedLoss,
    stopReason: stopRequested ? 'stopped' : 'done',
    elapsedSeconds,
  };
}

/// What the two loss curves are saying, in a sentence, or null while
/// they are still saying "keep going".
///
/// Training loss falls whether a model is learning the language or
/// memorizing the corpus. Held-out loss is what separates those, and the
/// two signals worth acting on are: it stopped improving (this corpus
/// has taught what it can), or it started rising while training loss
/// falls (the model is memorizing). Both have the same answer - more
/// text - and both are invisible unless somebody watches the numbers,
/// so the run watches them.
function corpusAdvice(heldOut, trainingLoss) {
  const WINDOW = 5; // about 125 steps at the validation cadence
  if (heldOut.length < WINDOW * 2) return null;
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const now = mean(heldOut.slice(-WINDOW));
  const before = mean(heldOut.slice(-WINDOW * 2, -WINDOW));
  const improvement = before - now;
  const gap = trainingLoss === null ? 0 : now - trainingLoss;

  if (improvement < -0.01) {
    return (
      'held-out loss is rising while training loss falls - the model has started memorizing your ' +
      'text rather than learning from it. Add more source material, or stop here and keep the ' +
      'model as it is.'
    );
  }
  if (improvement < 0.01) {
    return gap > 0.3
      ? 'held-out loss has flattened and sits well above training loss - this corpus has taught ' +
        'what it can. More text will help more than more steps.'
      : 'held-out loss has flattened. More text, a bigger model or a higher learning rate would ' +
        'each do more than more steps at this setting.';
  }
  return null;
}

/// A lost or reset device is the one failure worth naming precisely: it
/// means a submission ran past the driver's watchdog, and the answer is a
/// smaller batch or a shorter context, not a retry.
function describeTrainingFailure(error) {
  const message = (error && error.message) || String(error);
  if (/device.*(lost|hung|removed|reset)|DXGI_ERROR|GPUDevice/i.test(message)) {
    return (
      `the GPU device was reset mid-step (${message}). That is the driver's watchdog: ` +
      'the work it was given took too long. Lower the batch size or the context length ' +
      'and reload the page.'
    );
  }
  return message;
}

const handlers = {
  async 'load-model'({ bytes }) {
    return loadModelBytes(bytes);
  },

  async 'create-model'({ layers, hidden, heads, kvHeads, contextLen, window: attentionWindow, seed }) {
    await ensureWasm();
    log('creating model', { layers, hidden, heads, kvHeads, contextLen, window: attentionWindow });
    llm = new wasm.WasmLLM(layers, hidden, heads, kvHeads, contextLen, attentionWindow, seed);
    log('model created; asking for a GPU device');
    await initGpu();
    return describeModel();
  },

  async 'import-checkpoint'({ bytes }) {
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

  /// Which loaded sources are copies of another. Reported, never
  /// removed: which copy to keep is the user's call.
  async 'duplicate-sources'() {
    return { ids: llm.duplicate_sources() };
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

  async 'story-state'() {
    return {
      characters: llm.story_characters() || [],
      locations: llm.story_locations() || [],
      sceneCount: llm.story_scene_count() || 0,
      preamble: llm.story_state_preamble() || '',
    };
  },

  async 'retrieve'({ query, k }) {
    return { chunks: llm.retrieve_context(text(query, 'the query'), k || 3) };
  },

  async generate(payload) {
    let extraContext = payload.extraContext || '';
    if (payload.useStoryState) {
      extraContext = [extraContext, llm.story_state_preamble()].filter(Boolean).join(' ');
    }
    if (payload.useRetrieval) {
      const retrieved = llm.retrieve_context_text(payload.prompt, 2);
      if (retrieved) extraContext = [extraContext, retrieved].filter(Boolean).join(' ');
    }
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

  async 'learn-vocabulary'({ maxVocabSize = 8192 }) {
    const before = llm.vocab_size();
    const started = performance.now();
    const size = llm.learn_vocabulary(maxVocabSize);
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

  async train(payload) {
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
  if (!llm && !['load-model', 'create-model', 'import-checkpoint', 'parse-prompt', 'stop'].includes(type)) {
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
