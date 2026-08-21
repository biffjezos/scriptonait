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
let trainParams = { batchSize: 4, lr: 0.01 };
let wasmReady = init().then(() => {
  postMessage({ type: 'ready' });
});

function post(msg) {
  postMessage(msg);
}

async function trainingLoop() {
  if (!training || !llm) return;
  try {
    const loss = llm.train_step(trainParams.batchSize, trainParams.lr);
    if (loss !== undefined) {
      post({ type: 'trainProgress', step: llm.step(), loss });
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
      const stats = llm.upsert_source(msg.id, msg.rawText, msg.isHtml);
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
      post({ type: 'sourceRemoved', id: msg.id, numSources: llm.num_sources(), totalTokens: llm.total_tokens() });
      break;
    }

    case 'startTraining': {
      trainParams = { batchSize: msg.batchSize, lr: msg.lr };
      training = true;
      trainingLoop();
      break;
    }

    case 'stopTraining': {
      training = false;
      post({ type: 'trainStopped', step: llm.step() });
      break;
    }

    case 'generate': {
      try {
        const text = msg.useGpu
          ? await llm.generate_gpu(msg.prompt, msg.maxNewTokens, msg.temperature, msg.seed ?? Date.now())
          : llm.generate(msg.prompt, msg.maxNewTokens, msg.temperature, msg.seed ?? Date.now());
        post({ type: 'generateResult', text, usedGpu: msg.useGpu });
      } catch (err) {
        post({ type: 'error', context: 'generate', message: String(err) });
      }
      break;
    }

    case 'initGpu': {
      try {
        await llm.init_gpu();
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

    case 'exportWeights': {
      const bytes = llm.export_weights();
      post({ type: 'weightsExported', bytes, step: llm.step() }, [bytes.buffer]);
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
