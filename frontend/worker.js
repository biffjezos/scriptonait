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

let llm = null;
let training = false;
let trainParams = { batchSize: 4, lr: 0.01, useGpu: false };
let gpuInitialized = false;
let gpuStepCounter = 0;
let wasmReady = init().then(() => {
  postMessage({ type: 'ready' });
});

function post(msg) {
  postMessage(msg);
}

function storyState() {
  return {
    characters: llm.story_characters(),
    locations: llm.story_locations(),
    sceneCount: llm.story_scene_count(),
  };
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

async function trainingLoop() {
  if (!training || !llm) return;
  try {
    let loss, step;
    if (trainParams.useGpu) {
      loss = await llm.train_step_gpu(trainParams.batchSize, trainParams.lr, gpuStepCounter++);
      step = gpuStepCounter;
    } else {
      loss = llm.train_step(trainParams.batchSize, trainParams.lr);
      step = llm.step();
    }
    if (loss !== undefined) {
      post({ type: 'trainProgress', step, loss });
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
        storyState: storyState(),
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
        storyState: storyState(),
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
      trainParams = { batchSize: msg.batchSize, lr: msg.lr, useGpu };
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
      gpuStepCounter = 0;
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
      try {
        const maxDiff = await llm.debug_compare_gpu_cpu(msg.prompt);
        post({ type: 'debugCompareResult', maxDiff });
      } catch (err) {
        post({ type: 'error', context: 'debugCompareGpuCpu', message: String(err) });
      }
      break;
    }

    case 'debugCompareGpuCpuGradient': {
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
