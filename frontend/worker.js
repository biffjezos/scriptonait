// Runs as a module Worker (see app.js: `new Worker('worker.js', {type:
// 'module'})`) so the wasm module's compute never blocks the UI thread.
// Owns exactly one WasmLLM instance and processes one request at a time;
// the training loop yields via `setTimeout(loop, 0)` between steps so
// other queued messages (stop training, generate, ...) get a turn.
//
// `./pkg/wasm_app.js` is wasm-pack's generated JS glue — it doesn't exist
// until you run the build (see the repo root README):
//   wasm-pack build crates/wasm-app --target web --out-dir ../../frontend/pkg

import init, { WasmLLM } from './pkg/wasm_app.js';

// Fixed settings for periodic in-training samples - deliberately not
// exposed in the UI, since these are meant as a lightweight qualitative
// progress check, not a replacement for the full Generate panel (which
// already has its own temperature/length/GPU controls).
const SAMPLE_MAX_TOKENS = 80;
const SAMPLE_TEMPERATURE = 0.8;

// Learning-rate decay: halve the learning rate every LR_DECAY_EVERY steps,
// down to a floor of LR_DECAY_FLOOR_FRACTION of whatever the "Learning
// rate" field is set to. A constant learning rate with Adam typically
// plateaus rather than converging further - it reaches a noise floor and
// oscillates around it instead of settling into a finer minimum - so this
// applies automatically, using the field's value as the *starting* rate
// rather than a fixed rate for the whole run. Step decay (not cosine) is
// used because training here is open-ended (start/stop by hand, no fixed
// step budget to plan a cosine schedule around).
const LR_DECAY_FACTOR = 0.5;
const LR_DECAY_EVERY = 1000;
const LR_DECAY_FLOOR_FRACTION = 0.05;

function effectiveLr(baseLr, stepsSoFar) {
  const decayed = baseLr * LR_DECAY_FACTOR ** Math.floor(stepsSoFar / LR_DECAY_EVERY);
  return Math.max(decayed, baseLr * LR_DECAY_FLOOR_FRACTION);
}

// --- GPU config benchmarking ------------------------------------------
//
// Times real train_step_gpu calls for a handful of candidate model
// shapes on a throwaway WasmLLM instance, so "which shape trains fastest
// on this GPU" is measured instead of guessed. Uses a synthetic
// placeholder source rather than the real corpus: step timing only
// depends on tensor shapes, not the numbers flowing through them, and a
// throwaway benchmark instance doesn't need real training data - just
// enough of it for sample_batch to succeed.
const BENCHMARK_WARMUP_RUNS = 1; // discards GpuModel::upload's one-time cost
const BENCHMARK_TIMED_RUNS = 5;
const BENCHMARK_PLACEHOLDER_TEXT = 'the quick brown fox jumps over the lazy dog. '.repeat(400);
// Bounded so a stuck call fails visibly instead of hanging forever with
// no feedback - but the warmup step needs a much longer allowance than
// the timed ones: many drivers compile each shader lazily on its first
// *dispatch*, not when the pipeline object is created, so the warmup
// step (the first time all ~20 forward/backward/Adam kernels ever run on
// a freshly-created device) can absorb a real one-time cold-compile cost
// that has nothing to do with this shape's steady-state speed - the
// thing actually being measured by the timed runs after it. Note either
// timeout can only give up on *waiting* for the call - wasm/WebGPU has no
// cancellation, so an abandoned operation may still be consuming GPU
// resources in the background afterwards.
const BENCHMARK_WARMUP_TIMEOUT_MS = 90_000;
const BENCHMARK_STEP_TIMEOUT_MS = 20_000;

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);
}

async function benchmarkOneConfig(cfg, batchSize, lr) {
  let tempLlm;
  try {
    tempLlm = new WasmLLM(cfg.numLayers, cfg.hiddenDim, cfg.numHeads, cfg.contextLen, cfg.localWindow, Date.now());
  } catch (err) {
    return { config: cfg, error: String(err) };
  }
  if (!tempLlm.gpu_supported()) {
    return { config: cfg, error: "This config's context/window exceeds the GPU backend's limit" };
  }
  tempLlm.upsert_source('__benchmark__', BENCHMARK_PLACEHOLDER_TEXT, false);
  try {
    await withTimeout(tempLlm.init_gpu(), BENCHMARK_WARMUP_TIMEOUT_MS, 'GPU init');
    for (let i = 0; i < BENCHMARK_WARMUP_RUNS; i++) {
      await withTimeout(tempLlm.train_step_gpu(batchSize, lr, i), BENCHMARK_WARMUP_TIMEOUT_MS, 'warmup step');
    }
    const durations = [];
    for (let i = 0; i < BENCHMARK_TIMED_RUNS; i++) {
      const t0 = performance.now();
      await withTimeout(tempLlm.train_step_gpu(batchSize, lr, BENCHMARK_WARMUP_RUNS + i), BENCHMARK_STEP_TIMEOUT_MS, `timed step ${i + 1}`);
      durations.push(performance.now() - t0);
    }
    durations.sort((a, b) => a - b);
    const medianStepMs = durations[Math.floor(durations.length / 2)];
    return { config: cfg, medianStepMs };
  } catch (err) {
    return { config: cfg, error: String(err) };
  }
}

let llm = null;
let training = false;
let trainParams = { batchSize: 4, lr: 0.01, useGpu: false, sampleEveryN: 0, samplePrompt: '' };
let gpuInitialized = false;
let gpuStepCounter = 0;
let wasmReady = init().then(() => {
  postMessage({ type: 'ready' });
});

function post(msg) {
  postMessage(msg);
}

// Builds the [GENRE: x] [TONE: y] preamble used both when adding a
// tagged source and when tagging a generation prompt — see db.js's note
// on `tags` for why this lives entirely in JS (llm-core just sees text).
function tagPreamble(tags) {
  if (!tags) return '';
  const parts = [];
  if (tags.genre) parts.push(`[GENRE: ${tags.genre}]`);
  if (tags.tone) parts.push(`[TONE: ${tags.tone}]`);
  return parts.length ? `${parts.join(' ')}\n` : '';
}

// Pauses training for one generation at the current (in-progress) weights,
// so the loss chart isn't the only signal of how training is going. GPU
// training only updates the GPU-resident weights (see train_step_gpu's
// docs), so this syncs first if needed - same as generate/export already
// do - meaning a sample always reflects the very latest training step.
async function maybeGenerateSample(step) {
  if (!trainParams.sampleEveryN || step <= 0 || step % trainParams.sampleEveryN !== 0) return;
  try {
    if (llm.gpu_training_dirty()) {
      await llm.sync_weights_from_gpu();
    }
    const text = llm.generate(trainParams.samplePrompt || '', SAMPLE_MAX_TOKENS, SAMPLE_TEMPERATURE, Date.now());
    post({ type: 'trainSample', step, text });
  } catch (err) {
    post({ type: 'error', context: 'trainSample', message: String(err) });
  }
}

async function trainingLoop() {
  if (!training || !llm) return;
  try {
    let loss, step, lr;
    // Timed around just the actual training call, not the periodic
    // sample generation below - that's a deliberate pause, not part of
    // real per-step throughput.
    const t0 = performance.now();
    if (trainParams.useGpu) {
      lr = effectiveLr(trainParams.lr, gpuStepCounter);
      loss = await llm.train_step_gpu(trainParams.batchSize, lr, gpuStepCounter);
      gpuStepCounter += 1;
      step = gpuStepCounter;
    } else {
      const stepsSoFar = Math.round(llm.step());
      lr = effectiveLr(trainParams.lr, stepsSoFar);
      loss = llm.train_step(trainParams.batchSize, lr);
      step = llm.step();
    }
    const stepMs = performance.now() - t0;
    if (loss !== undefined) {
      post({ type: 'trainProgress', step, loss, lr, stepMs, batchSize: trainParams.batchSize });
      await maybeGenerateSample(Math.round(step));
    } else {
      post({ type: 'trainStalled', message: 'Not enough training data yet — add a source with more text.' });
      training = false;
      return;
    }
  } catch (err) {
    post({ type: 'error', context: 'train_step', message: String(err) });
    training = false;
    return;
  }
  setTimeout(trainingLoop, 0);
}

async function handleMessage(msg) {
  await wasmReady;

  switch (msg.type) {
    case 'createModel': {
      const { numLayers, hiddenDim, numHeads, contextLen, localWindow, seed } = msg.config;
      llm = new WasmLLM(numLayers, hiddenDim, numHeads, contextLen, localWindow, seed ?? Date.now());
      post({
        type: 'modelCreated',
        paramCount: llm.param_count(),
        memoryInference: llm.memory_bytes(false),
        memoryTraining: llm.memory_bytes(true),
        gpuSupported: llm.gpu_supported(),
      });
      break;
    }

    case 'upsertSource': {
      // tags (genre/tone) get prepended as a short preamble so the model
      // can learn to associate them with what follows - see corpus.rs's
      // boundary-aligned sampling, which is what makes a preamble like
      // this actually learnable rather than buried mid-window most of
      // the time.
      const text = tagPreamble(msg.tags) + msg.rawText;
      const stats = llm.upsert_source(msg.id, text, msg.isHtml);
      post({
        type: 'sourceStats',
        id: msg.id,
        charCount: stats.char_count,
        byteCount: stats.byte_count,
        tokenCount: stats.token_count,
        numSources: llm.num_sources(),
        totalTokens: llm.total_tokens(),
      });
      break;
    }

    case 'removeSource': {
      llm.remove_source(msg.id);
      post({
        type: 'sourceRemoved',
        id: msg.id,
        numSources: llm.num_sources(),
        totalTokens: llm.total_tokens(),
      });
      break;
    }

    case 'previewRetrieval': {
      try {
        const chunks = llm.retrieve_context(msg.query, msg.k ?? 3);
        post({ type: 'retrievalPreview', chunks });
      } catch (err) {
        post({ type: 'error', context: 'previewRetrieval', message: String(err) });
      }
      break;
    }

    case 'benchmarkConfigs': {
      // Uses its own throwaway WasmLLM instance(s) per candidate - doesn't
      // touch the main `llm`/training state at all, so this is safe to run
      // whether or not a real model exists yet.
      const results = [];
      for (let i = 0; i < msg.configs.length; i++) {
        post({ type: 'benchmarkProgress', index: i, total: msg.configs.length, config: msg.configs[i] });
        results.push(await benchmarkOneConfig(msg.configs[i], msg.batchSize, msg.lr));
      }
      post({ type: 'benchmarkResult', results });
      break;
    }

    case 'startTraining': {
      // Re-verify against the actual model config rather than trusting
      // the requested flag as-is: a stale/mismatched checkbox state (or
      // any other caller) asking for GPU on a config the GPU backend
      // can't handle (context/window > MAX_GPU_WINDOW) should fall back
      // to CPU automatically instead of failing silently mid-training.
      const useGpu = !!msg.useGpu && llm.gpu_supported();
      if (msg.useGpu && !useGpu) {
        post({
          type: 'trainFallback',
          message: "This model's context/attention window is too large for the GPU backend — training on CPU instead.",
        });
      }
      trainParams = {
        batchSize: msg.batchSize,
        lr: msg.lr,
        useGpu,
        sampleEveryN: msg.sampleEveryN || 0,
        samplePrompt: msg.samplePrompt || '',
      };
      if (trainParams.useGpu && !gpuInitialized) {
        try {
          await llm.init_gpu();
          gpuInitialized = true;
          post({ type: 'gpuReady' });
        } catch (err) {
          post({ type: 'trainStalled', message: `WebGPU unavailable: ${err} — check "Train on WebGPU" only works after a WebGPU device is available.` });
          break;
        }
      }
      // Start from the persisted step count, not 0 - it stays meaningful
      // across stop/resume (llm.step() reflects real progress even after
      // a GPU sync, see sync_weights_from_gpu's docs) and keeps the LR
      // decay schedule continuous instead of restarting at full strength
      // every time GPU training is (re)started.
      gpuStepCounter = Math.round(llm.step());
      training = true;
      trainingLoop();
      break;
    }

    case 'stopTraining': {
      training = false;
      try {
        if (trainParams.useGpu && llm.gpu_training_dirty()) {
          await llm.sync_weights_from_gpu();
        }
      } catch (err) {
        post({ type: 'error', context: 'stopTraining', message: String(err) });
      }
      post({ type: 'trainStopped', step: llm.step() });
      break;
    }

    case 'generate': {
      try {
        // Generation on the GPU path (or syncing GPU-trained weights back
        // to the CPU) touches the same GpuModel scratch buffers the
        // training loop is actively writing to - nothing serializes GPU
        // work between separate worker messages, so running this
        // concurrently with training corrupts both operations' in-flight
        // state instead of erroring cleanly. Refuse outright rather than
        // silently producing garbage; see the matching guard on the debug
        // compare handlers below for the same reasoning.
        if (training && (msg.useGpu || llm.gpu_training_dirty())) {
          post({ type: 'error', context: 'generate', message: 'Stop training first — GPU generation shares GPU state with the running training loop and can\'t safely run at the same time.' });
          break;
        }
        // GPU training only updates the GPU-resident weights (see
        // train_step_gpu's docs) — bring the CPU copy up to date first so
        // generation (CPU or GPU) reflects the latest training.
        if (llm.gpu_training_dirty()) {
          await llm.sync_weights_from_gpu();
        }
        // Assemble the effective prompt: optional genre/tone tags, an
        // optional story-state reminder (characters/locations seen so
        // far), optional retrieved similar scenes, then the user's own
        // prompt text - in that order, so the user's words are always
        // what's freshest/closest to the generation point.
        let effectivePrompt = tagPreamble(msg.tags);
        if (msg.useStoryState) {
          effectivePrompt += llm.story_state_preamble();
        }
        if (msg.useRetrieval) {
          effectivePrompt += llm.retrieve_context_text(msg.prompt, msg.retrievalK ?? 3);
        }
        effectivePrompt += msg.prompt;

        const useGpu = !!msg.useGpu && llm.gpu_supported();
        const text = useGpu
          ? await llm.generate_gpu(effectivePrompt, msg.maxNewTokens, msg.temperature, msg.seed ?? Date.now())
          : llm.generate(effectivePrompt, msg.maxNewTokens, msg.temperature, msg.seed ?? Date.now());

        const qaNotes = llm.qa_check(text, msg.targetWordCount ?? 0);

        post({ type: 'generateResult', text, usedGpu: useGpu, effectivePrompt, qaNotes });
      } catch (err) {
        post({ type: 'error', context: 'generate', message: String(err) });
      }
      break;
    }

    case 'initGpu': {
      try {
        await llm.init_gpu();
        gpuInitialized = true;
        post({ type: 'gpuReady' });
      } catch (err) {
        post({ type: 'gpuUnavailable', message: String(err) });
      }
      break;
    }

    case 'debugCompareGpuCpu': {
      // Shares the GpuModel's scratch buffers with the training loop -
      // running this while training is active corrupts both, since
      // nothing serializes GPU work between separate worker messages (see
      // the matching guard on 'generate' above). Refuse instead of
      // silently producing a misleading "large diff" that looks like a
      // kernel bug but is really two operations racing each other.
      if (training) {
        post({ type: 'error', context: 'debugCompareGpuCpu', message: 'Stop training first — this shares GPU state with the running training loop and can\'t safely run at the same time.' });
        break;
      }
      try {
        const maxDiff = await llm.debug_compare_gpu_cpu(msg.prompt);
        post({ type: 'debugCompareResult', maxDiff });
      } catch (err) {
        post({ type: 'error', context: 'debugCompareGpuCpu', message: String(err) });
      }
      break;
    }

    case 'debugCompareGpuCpuGradient': {
      if (training) {
        post({ type: 'error', context: 'debugCompareGpuCpuGradient', message: 'Stop training first — this shares GPU state with the running training loop and can\'t safely run at the same time.' });
        break;
      }
      try {
        const maxDiff = await llm.debug_compare_gpu_cpu_gradient(msg.prompt);
        post({ type: 'debugCompareGradientResult', maxDiff });
      } catch (err) {
        post({ type: 'error', context: 'debugCompareGpuCpuGradient', message: String(err) });
      }
      break;
    }

    case 'exportWeights': {
      try {
        if (llm.gpu_training_dirty()) {
          await llm.sync_weights_from_gpu();
        }
        const bytes = llm.export_weights();
        post({ type: 'weightsExported', bytes, step: llm.step() }, [bytes.buffer]);
      } catch (err) {
        post({ type: 'error', context: 'exportWeights', message: String(err) });
      }
      break;
    }

    case 'importWeights': {
      try {
        llm.import_weights(new Uint8Array(msg.bytes));
        post({ type: 'weightsImported', step: llm.step() });
      } catch (err) {
        post({ type: 'error', context: 'importWeights', message: String(err) });
      }
      break;
    }

    default:
      post({ type: 'error', context: 'unknownMessage', message: `unknown message type: ${msg.type}` });
  }
}

onmessage = (event) => {
  handleMessage(event.data).catch((err) => {
    post({ type: 'error', context: event.data && event.data.type, message: String(err) });
  });
};
