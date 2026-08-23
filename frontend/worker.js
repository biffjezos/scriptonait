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
    log(`WebGPU device acquired in ${(performance.now() - startedAt).toFixed(0)} ms`, report);
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

async function generate({ prompt, extraContext, temperature, topK, topP, repetitionPenalty, seed }) {
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
    1.1,
    Math.floor(Math.random() * 1e9),
    (_piece, produced) => produced < words,
  );
  return result.text;
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
    maxSteps: maxSteps || 'until stopped',
    effort,
    learningRate: learningRate > 0 ? learningRate : 'automatic',
    params: info.params,
  });

  const sliceMs = 120;
  const pauseMs = Math.max(0, Math.round(sliceMs * (1 - effort) / Math.max(effort, 0.05)));
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
          log(
            `step ${llm.step().toLocaleString()}: held-out loss ${measured.toFixed(4)}` +
              (gap === null ? '' : ` (training ${smoothedLoss.toFixed(4)}, gap ${gap.toFixed(4)})`),
          );
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
        post('train-sample', {
          step: llm.step(),
          loss: smoothedLoss,
          text: await trainingSample(samplePrompt, sampleWords || 40),
        });
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

  async 'upsert-source'(payload) {
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
        type === 'profile'
          ? 'profiling needs a model: press Train first (a model lives in this tab only, so ' +
            'a reload leaves none), then run scriptonait.profile() again'
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
