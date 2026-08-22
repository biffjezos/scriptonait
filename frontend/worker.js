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

import init, { WasmLLM, parse_prompt_standalone } from './pkg/wasm_app.js';

let llm = null;
let wasmReady = null;

// Set by a `stop` message. Checked by the generation callback and between
// training steps — both cooperative, since neither can be interrupted
// mid-step.
let stopRequested = false;

const PROGRESS_INTERVAL_MS = 250;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function fail(id, error) {
  post('error', { id, message: error && error.message ? error.message : String(error) });
}

async function ensureWasm() {
  if (!wasmReady) wasmReady = init();
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
  llm = WasmLLM.from_checkpoint(new Uint8Array(bytes));
  await initGpu();
  return describeModel();
}

/// Ask for a WebGPU device and upload the weights to it.
///
/// Not a setting and not a fallback the user has to think about: if the
/// browser has WebGPU, generation runs there; if it doesn't, it runs on
/// the CPU. Either way the page says which. A failure here is normal —
/// browsers without WebGPU exist — so it's reported, not thrown.
async function initGpu() {
  if (!llm) return;
  try {
    const summary = await llm.init_gpu();
    post('gpu-status', { available: true, device: summary });
  } catch (error) {
    post('gpu-status', {
      available: false,
      device: 'CPU',
      reason: (error && error.message) || String(error),
    });
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
  const parsed = llm ? llm.parse_prompt(prompt) : parse_prompt_standalone(prompt);
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
async function train({ batchSize, learningRate, maxSteps, effort }) {
  stopRequested = false;
  if (learningRate > 0) llm.set_learning_rate(learningRate);

  const sliceMs = 120;
  const pauseMs = Math.max(0, Math.round(sliceMs * (1 - effort) / Math.max(effort, 0.05)));
  const startedAt = performance.now();
  let steps = 0;
  let tokens = 0;
  let lastPost = 0;
  let smoothedLoss = null;

  while (!stopRequested && (maxSteps <= 0 || steps < maxSteps)) {
    const sliceStart = performance.now();
    // Work for a slice...
    while (performance.now() - sliceStart < sliceMs) {
      const report = llm.train_step(batchSize);
      if (!report) {
        return { steps, stopReason: 'no-data', elapsedSeconds: (performance.now() - startedAt) / 1000 };
      }
      steps += 1;
      tokens += report.tokens;
      smoothedLoss = smoothedLoss === null ? report.loss : smoothedLoss * 0.9 + report.loss * 0.1;

      const now = performance.now();
      if (now - lastPost >= PROGRESS_INTERVAL_MS) {
        lastPost = now;
        const elapsed = (now - startedAt) / 1000;
        post('train-progress', {
          step: report.step,
          steps,
          loss: report.loss,
          smoothedLoss,
          lr: report.lr,
          gradNorm: report.grad_norm,
          elapsedSeconds: elapsed,
          tokensPerSecond: elapsed > 0 ? tokens / elapsed : 0,
          fractionDone: maxSteps > 0 ? steps / maxSteps : 0,
        });
      }
      if (stopRequested || (maxSteps > 0 && steps >= maxSteps)) break;
    }
    // ...then hand the machine back.
    if (pauseMs > 0) await new Promise((resolve) => setTimeout(resolve, pauseMs));
  }

  return {
    steps,
    loss: smoothedLoss,
    stopReason: stopRequested ? 'stopped' : 'done',
    elapsedSeconds: (performance.now() - startedAt) / 1000,
  };
}

const handlers = {
  async 'load-model'({ bytes }) {
    return loadModelBytes(bytes);
  },

  async 'create-model'({ layers, hidden, heads, kvHeads, contextLen, window: attentionWindow, seed }) {
    await ensureWasm();
    llm = new WasmLLM(layers, hidden, heads, kvHeads, contextLen, attentionWindow, seed);
    await initGpu();
    return describeModel();
  },

  async 'import-checkpoint'({ bytes }) {
    await ensureWasm();
    if (!llm) {
      llm = WasmLLM.from_checkpoint(new Uint8Array(bytes));
    } else {
      llm.import_checkpoint(new Uint8Array(bytes));
    }
    await initGpu();
    return describeModel();
  },

  async 'export-checkpoint'() {
    const bytes = llm.export_checkpoint();
    return { bytes: bytes.buffer, byteLength: bytes.length };
  },

  async 'model-info'() {
    return describeModel();
  },

  async 'parse-prompt'({ prompt }) {
    await ensureWasm();
    return describePrompt(prompt);
  },

  async 'upsert-source'({ id, text, isHtml }) {
    const stats = llm.upsert_source(id, text, !!isHtml);
    return {
      charCount: stats.char_count,
      byteCount: stats.byte_count,
      tokenCount: stats.token_count,
      model: describeModel(),
    };
  },

  async 'remove-source'({ id }) {
    llm.remove_source(id);
    return describeModel();
  },

  async 'story-state'() {
    return {
      characters: llm.story_characters(),
      locations: llm.story_locations(),
      sceneCount: llm.story_scene_count(),
      preamble: llm.story_state_preamble(),
    };
  },

  async 'retrieve'({ query, k }) {
    return { chunks: llm.retrieve_context(query, k) };
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
    const result = await generate({ ...payload, extraContext });
    const parsed = describePrompt(payload.prompt);
    result.notes = llm.qa_check(result.text, parsed.targetWords || 0);
    return result;
  },

  async train(payload) {
    if (!llm.can_train()) {
      return { steps: 0, stopReason: 'no-data', elapsedSeconds: 0 };
    }
    const result = await train(payload);
    result.model = describeModel();
    return result;
  },

  async stop() {
    stopRequested = true;
    return { stopping: true };
  },
};

self.onmessage = async (event) => {
  const { id, type, ...payload } = event.data || {};
  const handler = handlers[type];
  if (!handler) {
    fail(id, new Error(`unknown message ${type}`));
    return;
  }
  // Everything except `stop` needs a model; saying so beats a wasm panic.
  if (!llm && !['load-model', 'create-model', 'import-checkpoint', 'parse-prompt', 'stop'].includes(type)) {
    fail(id, new Error('no model loaded yet'));
    return;
  }
  try {
    const result = await handler(payload);
    if (result && result.bytes instanceof ArrayBuffer) {
      // Transfer rather than copy: an exported checkpoint is tens of MB.
      self.postMessage({ type: 'result', id, result }, [result.bytes]);
    } else {
      post('result', { id, result });
    }
  } catch (error) {
    fail(id, error);
  }
};
