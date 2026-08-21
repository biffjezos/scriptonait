import * as db from './db.js';

const el = (id) => document.getElementById(id);

const worker = new Worker('worker.js', { type: 'module' });

// --- App state -----------------------------------------------------------

let modelCreated = false;
let modelConfig = null;
let editingSourceId = null;
let lossHistory = [];
let stepMsHistory = [];
const STEP_MS_WINDOW = 20;
let training = false;

const webgpuBrowserSupported = 'gpu' in navigator;

// This app's own limit, not a hardware one: llm-gpu's attention kernel
// uses a fixed-size local array (MAX_WINDOW = 256u in attention.wgsl),
// and the backward pass needs a dense [heads, context, context] cache
// per layer sized off it - must match llm_gpu::MAX_GPU_WINDOW exactly.
// Real GPUs (see the details panel below) support far larger buffers and
// workgroups than this; it's a simplification in how this app's shaders
// are written, not something the hardware is short on.
const GPU_MAX_CONTEXT = 256;

function setWebGpuStatus(text, cls) {
  const banner = el('webgpu-status');
  banner.textContent = text;
  banner.className = 'webgpu-status' + (cls ? ` ${cls}` : '');
}

// Independent of the worker's own GpuContext (used for actual
// training/generation) - requesting an adapter just to read its info/
// limits doesn't need a device at all, so this can populate immediately
// on page load regardless of whether a model has been created yet.
// GpuContext::new() requests `required_limits: adapter.limits()` (the
// full set, not a reduced one), so what's shown here matches what
// training actually gets to use.
async function loadGpuDetails() {
  if (!webgpuBrowserSupported) return;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return;
    const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
    const limits = adapter.limits;

    const rows = [];
    if (info) {
      rows.push(['Vendor', info.vendor || 'unknown']);
      rows.push(['Architecture', info.architecture || 'unknown']);
      rows.push(['Device', info.device || 'unknown']);
      rows.push(['Description', info.description || '(none)']);
    }
    rows.push(['Type', adapter.isFallbackAdapter ? 'Software (fallback)' : 'Hardware']);
    // The limits actually relevant to this app's compute shaders - not
    // the full ~30-entry spec table, just what could plausibly constrain
    // (or explain slow) training here.
    rows.push(['Max compute invocations / workgroup', limits.maxComputeInvocationsPerWorkgroup.toLocaleString()]);
    rows.push([
      'Max workgroup size (X / Y / Z)',
      `${limits.maxComputeWorkgroupSizeX} / ${limits.maxComputeWorkgroupSizeY} / ${limits.maxComputeWorkgroupSizeZ}`,
    ]);
    rows.push(['Max storage buffer binding', formatBytes(limits.maxStorageBufferBindingSize)]);
    rows.push(['Max buffer size', formatBytes(limits.maxBufferSize)]);
    rows.push(['Max compute workgroup storage', formatBytes(limits.maxComputeWorkgroupStorageSize)]);

    const container = el('gpu-details-content');
    container.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'gpu-details-table';
    for (const [label, value] of rows) {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = label;
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(th, td);
      table.appendChild(tr);
    }
    container.appendChild(table);

    // Deliberately separate from the table above: everything in it comes
    // from the hardware/driver; this comes from this app's own shaders,
    // not your GPU, so labeling it as just another "limit" row would
    // wrongly suggest the GPU itself is what's capping this.
    const note = document.createElement('p');
    note.className = 'gpu-details-note';
    note.textContent =
      `This app's own context/attention-window limit: ${GPU_MAX_CONTEXT} tokens — not a hardware limit, ` +
      `a fixed-size array in this app's attention shader. Your GPU's own limits above are all far higher.`;
    container.appendChild(note);

    el('gpu-details').hidden = false;
  } catch (err) {
    // Non-fatal - this is a "nice to have" diagnostic, not required for
    // the app to function, so a failure here shouldn't surface as an error.
    console.warn('Could not read GPU adapter details:', err);
  }
}

loadGpuDetails();

if (!webgpuBrowserSupported) {
  setWebGpuStatus(
    'WebGPU: not available in this browser (navigator.gpu is undefined) — training and ' +
      'generation will use CPU (slower, but works everywhere). Try a recent Chrome or Edge for GPU acceleration.',
    'fallback'
  );
} else {
  setWebGpuStatus('WebGPU: available — will be used automatically once a model is created.', '');
}

// --- Worker <-> UI wiring --------------------------------------------------

worker.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'ready':
      break;

    case 'modelCreated': {
      modelCreated = true;
      el('model-status').textContent =
        `Model ready — ${Math.round(msg.paramCount).toLocaleString()} parameters ` +
        `(${formatBytes(msg.memoryInference)} inference / ${formatBytes(msg.memoryTraining)} training).`;
      el('train-panel').hidden = false;
      el('generate-panel').hidden = false;
      el('save-panel').hidden = false;
      const gpuUsable = msg.gpuSupported && webgpuBrowserSupported;
      el('gen-use-gpu').disabled = !gpuUsable;
      el('train-use-gpu').disabled = !gpuUsable;
      if (!gpuUsable) {
        // Both checkboxes start checked in the HTML (GPU is the default
        // preference) - disabling alone leaves a disabled checkbox still
        // reporting .checked === true, which would silently send
        // useGpu: true to the worker for a config the GPU backend can't
        // actually handle. Force them off too.
        el('gen-use-gpu').checked = false;
        el('train-use-gpu').checked = false;
      }
      if (!msg.gpuSupported) {
        el('gpu-status').textContent =
          'This attention window is larger than the simple GPU kernel supports — generation will use the CPU path only.';
        setWebGpuStatus(
          "WebGPU: this model's context/attention window is too large for the GPU backend — using CPU for this model.",
          'fallback'
        );
      } else if (webgpuBrowserSupported) {
        // Default to GPU: request a device right away rather than waiting
        // for the user to toggle a checkbox, so "Use WebGPU acceleration"/
        // "Train on WebGPU" being checked by default actually works the
        // first time someone clicks Generate/Start training.
        setWebGpuStatus('WebGPU: initializing device…', '');
        worker.postMessage({ type: 'initGpu' });
      }
      pushAllSourcesToWorker();
      break;
    }

    case 'sourceStats':
    case 'sourceRemoved': {
      refreshCorpusStatsFromWorker(msg);
      updateStoryStatePanel(msg.storyState);
      break;
    }

    case 'trainProgress': {
      lossHistory.push(msg.loss);
      if (lossHistory.length > 400) lossHistory.shift();
      const lrText = msg.lr !== undefined ? ` — lr ${msg.lr.toExponential(2)}` : '';
      el('train-status').textContent = `Step ${Math.round(msg.step).toLocaleString()} — loss ${msg.loss.toFixed(4)}${lrText}`;
      if (msg.stepMs !== undefined) {
        stepMsHistory.push(msg.stepMs);
        if (stepMsHistory.length > STEP_MS_WINDOW) stepMsHistory.shift();
        const avgMs = stepMsHistory.reduce((a, b) => a + b, 0) / stepMsHistory.length;
        let perfText = `≈${avgMs.toFixed(0)} ms/step (avg over last ${stepMsHistory.length})`;
        if (msg.batchSize && modelConfig?.contextLen) {
          const tokensPerSec = (msg.batchSize * modelConfig.contextLen * 1000) / avgMs;
          perfText += ` — ≈${Math.round(tokensPerSec).toLocaleString()} tokens/sec`;
        }
        el('train-perf').textContent = perfText;
      }
      drawLossChart();
      break;
    }

    case 'trainStalled': {
      training = false;
      setTrainingButtons(false);
      el('train-status').textContent = msg.message;
      break;
    }

    case 'trainSample': {
      // Overwritten each time, not appended - this is a live "how's it
      // doing right now" window, not a history.
      const container = el('train-samples');
      container.innerHTML = '';
      const entry = document.createElement('div');
      entry.className = 'sample';
      const label = document.createElement('div');
      label.className = 'step-label';
      label.textContent = `Step ${Math.round(msg.step).toLocaleString()}`;
      const text = document.createElement('div');
      text.className = 'text';
      text.textContent = msg.text;
      entry.append(label, text);
      container.appendChild(entry);
      break;
    }

    case 'trainFallback': {
      // Non-fatal: the worker downgraded this run to CPU (e.g. GPU was
      // requested for a config the GPU backend can't handle) - training
      // still proceeds normally right after this, just not on GPU.
      el('train-status').textContent = msg.message;
      el('train-use-gpu').checked = false;
      break;
    }

    case 'trainStopped': {
      training = false;
      setTrainingButtons(false);
      el('train-status').textContent = `Stopped at step ${Math.round(msg.step).toLocaleString()}.`;
      break;
    }

    case 'generateResult': {
      el('generate-output').textContent = msg.text;
      el('generate-btn').disabled = false;
      renderQaNotes(msg.qaNotes || []);
      el('effective-prompt').textContent = msg.effectivePrompt || '';
      el('effective-prompt-details').hidden = !msg.effectivePrompt;
      break;
    }

    case 'retrievalPreview': {
      renderRetrievalPreview(msg.chunks || []);
      el('preview-retrieval-btn').disabled = false;
      break;
    }

    case 'gpuReady': {
      setWebGpuStatus('WebGPU: enabled — training and generation run on the GPU.', 'enabled');
      el('gen-use-gpu').disabled = false;
      el('gen-use-gpu').checked = true;
      el('train-use-gpu').disabled = false;
      el('train-use-gpu').checked = true;
      el('gpu-status').textContent = 'WebGPU device ready.';
      el('debug-compare-btn').hidden = false;
      el('debug-compare-gradient-btn').hidden = false;
      break;
    }

    case 'gpuUnavailable': {
      setWebGpuStatus(`WebGPU: unavailable (${msg.message}) — using CPU.`, 'fallback');
      el('gen-use-gpu').checked = false;
      el('gen-use-gpu').disabled = true;
      el('train-use-gpu').checked = false;
      el('train-use-gpu').disabled = true;
      el('gpu-status').textContent = `WebGPU unavailable: ${msg.message} — using CPU generation instead.`;
      break;
    }

    case 'debugCompareResult': {
      el('gpu-status').textContent =
        `GPU vs CPU max logit difference: ${msg.maxDiff.toExponential(3)} ` +
        (msg.maxDiff < 0.01 ? '(looks correct)' : '(unexpectedly large — GPU output may be wrong)');
      break;
    }

    case 'debugCompareGradientResult': {
      el('train-status').textContent =
        `GPU vs CPU max embedding-gradient difference: ${msg.maxDiff.toExponential(3)} ` +
        (msg.maxDiff < 0.01 ? '(looks correct)' : '(unexpectedly large — GPU training may be wrong)');
      break;
    }

    case 'weightsImported': {
      el('model-status').textContent = `Weights imported (step reset to ${msg.step}).`;
      break;
    }

    case 'weightsExported':
      // Handled by a one-off listener registered at the call site (the
      // download and save-checkpoint buttons each need to do something
      // different with the exported bytes) — nothing to do here.
      break;

    case 'benchmarkProgress':
    case 'benchmarkResult':
      // Handled by a one-off listener registered at the suggest-settings
      // call site — nothing to do here.
      break;

    case 'error': {
      console.error(`[worker:${msg.context}]`, msg.message);
      alert(`${msg.context}: ${msg.message}`);
      // Reset whichever button might have triggered this (harmless if it
      // wasn't the one that failed).
      el('generate-btn').disabled = false;
      el('preview-retrieval-btn').disabled = false;
      break;
    }

    default:
      console.warn('unhandled worker message', msg);
  }
};

function refreshCorpusStatsFromWorker(msg) {
  el('corpus-stats').textContent =
    `${msg.numSources} source${msg.numSources === 1 ? '' : 's'} loaded into the model, ` +
    `${Math.round(msg.totalTokens).toLocaleString()} training tokens total.`;
}

function setTrainingButtons(isTraining) {
  el('start-train-btn').disabled = isTraining;
  el('stop-train-btn').disabled = !isTraining;
}

const STORY_STATE_DISMISSED_KEY = 'scriptonait.storyStateDismissed';

function isStoryStateDismissed() {
  try {
    return localStorage.getItem(STORY_STATE_DISMISSED_KEY) === '1';
  } catch {
    return false;
  }
}

function updateStoryStatePanel(storyState) {
  if (!storyState) return;
  const panel = el('story-state-panel');
  const hasAnything = storyState.characters.length > 0 || storyState.locations.length > 0;
  panel.hidden = !hasAnything || isStoryStateDismissed();
  if (!hasAnything) return;
  el('story-characters').textContent = storyState.characters.length ? storyState.characters.join(', ') : '—';
  el('story-locations').textContent = storyState.locations.length ? storyState.locations.join(', ') : '—';
  el('story-scene-count').textContent = String(storyState.sceneCount);
}

el('dismiss-story-state-btn').addEventListener('click', () => {
  el('story-state-panel').hidden = true;
  try {
    localStorage.setItem(STORY_STATE_DISMISSED_KEY, '1');
  } catch {
    // Storage unavailable (private mode, quota) - the panel still stays
    // hidden for this page load, it just won't persist across reloads.
  }
});

function renderQaNotes(notes) {
  const container = el('qa-notes');
  container.innerHTML = '';
  for (const note of notes) {
    const div = document.createElement('div');
    const isWarning = note.startsWith('[WARNING]');
    div.className = `note${isWarning ? ' warning' : ''}`;
    div.textContent = note;
    container.appendChild(div);
  }
}

function renderRetrievalPreview(chunks) {
  const container = el('retrieval-preview');
  container.innerHTML = '';
  container.hidden = chunks.length === 0;
  for (const chunk of chunks) {
    const div = document.createElement('div');
    div.className = 'chunk';
    div.textContent = chunk;
    container.appendChild(div);
  }
}

// --- Sources ---------------------------------------------------------------

async function pushAllSourcesToWorker() {
  const sources = await db.listSources();
  for (const src of sources) {
    worker.postMessage({
      type: 'upsertSource',
      id: src.id,
      rawText: src.rawText,
      isHtml: src.kind === 'url',
      tags: src.tags,
    });
  }
}

function currentSourceTags() {
  return {
    genre: el('source-genre-tag').value.trim(),
    tone: el('source-tone-tag').value.trim(),
  };
}

async function addSourceRecord({ title, kind, rawText, sourceUrl = null }) {
  if (!rawText || !rawText.trim()) return;
  const tags = currentSourceTags();
  const record = await db.addSource({ title, kind, rawText, sourceUrl, tags });
  if (modelCreated) {
    worker.postMessage({
      type: 'upsertSource',
      id: record.id,
      rawText: record.rawText,
      isHtml: kind === 'url',
      tags: record.tags,
    });
  }
  await refreshSourcesList();
}

async function refreshSourcesList() {
  const sources = await db.listSources();
  const container = el('sources-list');
  container.innerHTML = '';

  if (sources.length === 0) {
    const p = document.createElement('p');
    p.className = 'empty-hint';
    p.textContent = 'No sources yet — add one above to start building your training corpus.';
    container.appendChild(p);
    el('corpus-stats').textContent = '';
    return;
  }

  for (const src of sources) {
    container.appendChild(src.id === editingSourceId ? renderSourceEditor(src) : renderSourceRow(src));
  }

  if (!modelCreated) {
    const totalChars = sources.reduce((sum, s) => sum + s.rawText.length, 0);
    el('corpus-stats').textContent =
      `${sources.length} source${sources.length === 1 ? '' : 's'}, ${totalChars.toLocaleString()} ` +
      'characters (raw). Create a model above to start training on these.';
  }
}

function renderSourceRow(src) {
  const item = document.createElement('div');
  item.className = 'source-item';

  const meta = document.createElement('div');
  meta.className = 'meta';
  const title = document.createElement('div');
  title.className = 'title';
  const badge = document.createElement('span');
  badge.className = 'kind-badge';
  badge.textContent = src.kind;
  title.append(badge, src.title);
  const stats = document.createElement('div');
  stats.className = 'stats';
  const tagBits = [];
  if (src.tags?.genre) tagBits.push(`genre: ${src.tags.genre}`);
  if (src.tags?.tone) tagBits.push(`tone: ${src.tags.tone}`);
  const tagText = tagBits.length ? ` — ${tagBits.join(', ')}` : '';
  stats.textContent = `${src.rawText.length.toLocaleString()} characters${tagText}`;
  meta.append(title, stats);

  const actions = document.createElement('div');
  actions.className = 'actions row';
  const editBtn = document.createElement('button');
  editBtn.type = 'button';
  editBtn.className = 'secondary';
  editBtn.textContent = 'Edit';
  editBtn.addEventListener('click', () => {
    editingSourceId = src.id;
    refreshSourcesList();
  });
  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'secondary';
  delBtn.textContent = 'Delete';
  delBtn.addEventListener('click', async () => {
    await db.deleteSource(src.id);
    if (modelCreated) worker.postMessage({ type: 'removeSource', id: src.id });
    await refreshSourcesList();
  });
  actions.append(editBtn, delBtn);

  item.append(meta, actions);
  return item;
}

function renderSourceEditor(src) {
  const item = document.createElement('div');
  item.className = 'source-item';
  const wrap = document.createElement('div');
  wrap.style.flex = '1';

  const titleInput = document.createElement('input');
  titleInput.type = 'text';
  titleInput.value = src.title;
  titleInput.style.width = '100%';
  titleInput.style.marginBottom = '0.4rem';

  const ta = document.createElement('textarea');
  ta.rows = 6;
  ta.value = src.rawText;

  const tagRow = document.createElement('div');
  tagRow.className = 'row';
  tagRow.style.margin = '0.4rem 0';
  const genreInput = document.createElement('input');
  genreInput.type = 'text';
  genreInput.placeholder = 'Genre';
  genreInput.value = src.tags?.genre || '';
  const toneInput = document.createElement('input');
  toneInput.type = 'text';
  toneInput.placeholder = 'Tone';
  toneInput.value = src.tags?.tone || '';
  tagRow.append(genreInput, toneInput);

  const btnRow = document.createElement('div');
  btnRow.className = 'row';
  btnRow.style.marginTop = '0.4rem';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.textContent = 'Save';
  saveBtn.addEventListener('click', async () => {
    const tags = { genre: genreInput.value.trim(), tone: toneInput.value.trim() };
    const updated = await db.updateSource(src.id, { title: titleInput.value.trim() || src.title, rawText: ta.value, tags });
    editingSourceId = null;
    if (modelCreated) {
      worker.postMessage({
        type: 'upsertSource',
        id: updated.id,
        rawText: updated.rawText,
        isHtml: updated.kind === 'url',
        tags: updated.tags,
      });
    }
    await refreshSourcesList();
  });
  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'secondary';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', () => {
    editingSourceId = null;
    refreshSourcesList();
  });
  btnRow.append(saveBtn, cancelBtn);

  wrap.append(titleInput, ta, tagRow, btnRow);
  item.append(wrap);
  return item;
}

el('add-url-btn').addEventListener('click', async () => {
  const url = el('url-input').value.trim();
  if (!url) return;
  el('add-url-btn').disabled = true;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    await addSourceRecord({ title: url, kind: 'url', rawText: text, sourceUrl: url });
    el('url-input').value = '';
  } catch (err) {
    alert(
      `Could not fetch that URL — this is almost always the site blocking cross-origin ` +
      `requests (CORS), not a bug here. Copy the text and use "Paste text" instead.\n\n${err}`
    );
  } finally {
    el('add-url-btn').disabled = false;
  }
});

el('file-input').addEventListener('change', async (event) => {
  const files = Array.from(event.target.files || []);
  for (const file of files) {
    const text = await file.text();
    await addSourceRecord({ title: file.name, kind: 'file', rawText: text, sourceUrl: null });
  }
  event.target.value = '';
});

el('add-paste-btn').addEventListener('click', async () => {
  const text = el('paste-input').value;
  const title = el('paste-title').value.trim() || `Pasted text — ${new Date().toLocaleString()}`;
  await addSourceRecord({ title, kind: 'paste', rawText: text, sourceUrl: null });
  el('paste-input').value = '';
  el('paste-title').value = '';
});

// --- Model shape / size estimate -------------------------------------------

const VOCAB_SIZE = 259; // 256 bytes + PAD/BOS/EOS — must match llm-core's tokenizer::VOCAB_SIZE

function currentConfig() {
  return {
    numLayers: parseInt(el('cfg-layers').value, 10),
    hiddenDim: parseInt(el('cfg-hidden').value, 10),
    numHeads: parseInt(el('cfg-heads').value, 10),
    contextLen: parseInt(el('cfg-context').value, 10),
    localWindow: parseInt(el('cfg-window').value, 10),
  };
}

// Mirrors llm-core's config.rs `param_count`/`memory_bytes`/`default_ffn_dim`
// exactly (verified equivalent for integer hidden_dim), so the estimate
// updates live while adjusting settings, before a model exists.
function estimateParamsAndMemory(cfg) {
  const ffnDim = Math.ceil((cfg.hiddenDim * 8) / 3 / 32) * 32;
  const embedding = VOCAB_SIZE * cfg.hiddenDim;
  const perLayerPle = VOCAB_SIZE * cfg.hiddenDim;
  const perLayerAttn = cfg.hiddenDim + 4 * cfg.hiddenDim * cfg.hiddenDim;
  const perLayerMlp = cfg.hiddenDim + 3 * cfg.hiddenDim * ffnDim;
  const perLayer = perLayerPle + perLayerAttn + perLayerMlp;
  const params = embedding + cfg.numLayers * perLayer + cfg.hiddenDim;
  const inferenceBytes = params * 4;
  const trainingBytes = inferenceBytes * 4;
  return { params, inferenceBytes, trainingBytes };
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function updateSizeEstimate() {
  const cfg = currentConfig();
  if (!cfg.hiddenDim || !cfg.numLayers || !cfg.numHeads) return;
  if (cfg.hiddenDim % cfg.numHeads !== 0) {
    el('size-estimate').textContent = `Heads (${cfg.numHeads}) must evenly divide nodes (${cfg.hiddenDim}).`;
    return;
  }
  const { params, inferenceBytes, trainingBytes } = estimateParamsAndMemory(cfg);
  el('size-estimate').textContent =
    `≈ ${Math.round(params).toLocaleString()} parameters — ${formatBytes(inferenceBytes)} for inference, ` +
    `${formatBytes(trainingBytes)} while training.`;
}

['cfg-layers', 'cfg-hidden', 'cfg-heads', 'cfg-context', 'cfg-window'].forEach((id) => {
  el(id).addEventListener('input', updateSizeEstimate);
});

// --- Settings suggestion ---------------------------------------------

// Target ~2.3 parameters per training token: this is small-scale
// interactive training, where the model sees the corpus many times over
// many epochs rather than one Chinchilla-style pass, so it's tuned for
// "enough capacity to actually learn spelling and local structure" rather
// than one-epoch data efficiency. Clamped regardless of corpus size so a
// tiny corpus doesn't get an unusably small model and a huge one doesn't
// get a model too large to train in a browser tab.
const PARAMS_PER_TOKEN = 2.3;
const MIN_SUGGESTED_PARAMS = 1_000_000;
const MAX_SUGGESTED_PARAMS = 40_000_000;

function suggestNumLayers(totalTokens) {
  if (totalTokens < 500_000) return 4;
  if (totalTokens < 3_000_000) return 6;
  if (totalTokens < 15_000_000) return 8;
  return 10;
}

function suggestBatchSize(totalTokens) {
  if (totalTokens < 100_000) return 4;
  if (totalTokens < 1_000_000) return 8;
  return 16;
}

// Picks the smallest-error hidden size (in steps of 32) for the given
// layer count by search rather than a closed-form solve, since the real
// parameter formula isn't cleanly invertible (ffnDim rounds up to a
// multiple of 32). numHeads doesn't affect the parameter count at all
// (Q/K/V/O are hiddenDim×hiddenDim regardless of how many heads split
// it), so it's irrelevant here and picked separately below.
function suggestHiddenDim(numLayers, targetParams) {
  let best = 64;
  let bestDiff = Infinity;
  for (let h = 64; h <= 1536; h += 32) {
    const { params } = estimateParamsAndMemory({ numLayers, hiddenDim: h, numHeads: 1 });
    const diff = Math.abs(params - targetParams);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = h;
    }
  }
  return best;
}

// Closest number of heads to a 64-wide head_dim (a common convention)
// that still evenly divides hiddenDim with an even head_dim (required for
// RoPE) — see the create-model-btn handler's own validation of this.
function suggestNumHeads(hiddenDim) {
  const target = Math.max(1, Math.round(hiddenDim / 64));
  for (let delta = 0; delta <= hiddenDim; delta++) {
    for (const h of [target - delta, target + delta]) {
      if (h >= 1 && hiddenDim % h === 0 && (hiddenDim / h) % 2 === 0) {
        return h;
      }
    }
  }
  return 1;
}

function suggestSettings(totalTokens) {
  const targetParams = Math.min(MAX_SUGGESTED_PARAMS, Math.max(MIN_SUGGESTED_PARAMS, totalTokens * PARAMS_PER_TOKEN));
  const numLayers = suggestNumLayers(totalTokens);
  const hiddenDim = suggestHiddenDim(numLayers, targetParams);
  return {
    numLayers,
    hiddenDim,
    numHeads: suggestNumHeads(hiddenDim),
    // Always the GPU backend's max: longer context helps this kind of
    // text, and nothing about corpus size argues for giving it up when
    // the GPU backend can handle the full 256 anyway.
    contextLen: GPU_MAX_CONTEXT,
    localWindow: GPU_MAX_CONTEXT,
    batchSize: suggestBatchSize(totalTokens),
    lr: 0.003,
    sampleEveryN: 250,
  };
}

function applySuggestion(suggestion) {
  el('cfg-layers').value = suggestion.numLayers;
  el('cfg-hidden').value = suggestion.hiddenDim;
  el('cfg-heads').value = suggestion.numHeads;
  el('cfg-context').value = suggestion.contextLen;
  el('cfg-window').value = suggestion.localWindow;
  el('cfg-batch').value = suggestion.batchSize;
  el('cfg-lr').value = suggestion.lr;
  el('train-sample-every').value = suggestion.sampleEveryN;
  updateSizeEstimate();
}

// A handful of depth/width tradeoffs that all target the same parameter
// budget (half the baseline layer count, the baseline itself, and double
// it — each re-solved for hidden size via suggestHiddenDim) - benchmarked
// for real below rather than guessed at, since raw compute scales with
// hidden size squared but GPU dispatch-call overhead scales with layer
// count alone, and which one actually dominates depends on the GPU.
function benchmarkCandidateShapes(baselineLayers, targetParams) {
  const layerOptions = [...new Set([Math.max(2, Math.round(baselineLayers / 2)), baselineLayers, Math.min(24, baselineLayers * 2)])];
  return layerOptions.map((numLayers) => {
    const hiddenDim = suggestHiddenDim(numLayers, targetParams);
    return {
      numLayers,
      hiddenDim,
      numHeads: suggestNumHeads(hiddenDim),
      contextLen: GPU_MAX_CONTEXT,
      localWindow: GPU_MAX_CONTEXT,
    };
  });
}

el('suggest-settings-btn').addEventListener('click', async () => {
  el('suggest-settings-btn').disabled = true;
  try {
    const sources = await db.listSources();
    if (sources.length === 0) {
      el('suggest-settings-status').textContent =
        'Add at least one source first — there is nothing to size a model against yet.';
      return;
    }
    // Byte count, not character count: the tokenizer is byte-level (one
    // token per UTF-8 byte), so this is what actually determines training
    // token count — source.rawText.length would undercount any non-ASCII
    // text.
    const encoder = new TextEncoder();
    const totalTokens = sources.reduce((sum, s) => sum + encoder.encode(s.rawText).length, 0);
    const suggestion = suggestSettings(totalTokens);
    const sourceWord = `${sources.length} source${sources.length === 1 ? '' : 's'}, ≈${Math.round(totalTokens).toLocaleString()} training tokens`;

    if (!webgpuBrowserSupported) {
      applySuggestion(suggestion);
      el('suggest-settings-status').textContent =
        `Suggested for ${sourceWord}: ${suggestion.numLayers} layers, ${suggestion.hiddenDim} nodes, ` +
        `${suggestion.numHeads} heads, batch size ${suggestion.batchSize}, learning rate ${suggestion.lr} ` +
        `(no WebGPU in this browser, so the shape wasn't benchmarked — sized by corpus alone).`;
      return;
    }

    const targetParams = Math.min(MAX_SUGGESTED_PARAMS, Math.max(MIN_SUGGESTED_PARAMS, totalTokens * PARAMS_PER_TOKEN));
    const candidates = benchmarkCandidateShapes(suggestion.numLayers, targetParams);
    el('suggest-settings-status').textContent = `Benchmarking ${candidates.length} model shapes on your GPU for this corpus…`;

    const results = await new Promise((resolve) => {
      const handler = (event) => {
        if (event.data.type === 'benchmarkProgress') {
          const c = event.data.config;
          el('suggest-settings-status').textContent =
            `Benchmarking shape ${event.data.index + 1} of ${event.data.total} (${c.numLayers} layers, ${c.hiddenDim} nodes)…`;
        } else if (event.data.type === 'benchmarkResult') {
          worker.removeEventListener('message', handler);
          resolve(event.data.results);
        }
      };
      worker.addEventListener('message', handler);
      worker.postMessage({ type: 'benchmarkConfigs', configs: candidates, batchSize: suggestion.batchSize, lr: suggestion.lr });
    });

    const timed = results.filter((r) => r.medianStepMs !== undefined);
    if (timed.length === 0) {
      applySuggestion(suggestion);
      el('suggest-settings-status').textContent =
        `Suggested for ${sourceWord}: ${suggestion.numLayers} layers, ${suggestion.hiddenDim} nodes ` +
        `(couldn't benchmark on GPU: ${results[0]?.error || 'unknown error'} — sized by corpus alone).`;
      return;
    }
    const best = timed.reduce((a, b) => (b.medianStepMs < a.medianStepMs ? b : a));
    applySuggestion({ ...suggestion, numLayers: best.config.numLayers, hiddenDim: best.config.hiddenDim, numHeads: best.config.numHeads });

    const summary = timed.map((r) => `${r.config.numLayers}L/${r.config.hiddenDim}H: ${r.medianStepMs.toFixed(0)}ms/step`).join(', ');
    el('suggest-settings-status').textContent =
      `Fastest of ${timed.length} shapes benchmarked on your GPU for ${sourceWord}: ` +
      `${best.config.numLayers} layers, ${best.config.hiddenDim} nodes, ${best.config.numHeads} heads ` +
      `(${best.medianStepMs.toFixed(0)}ms/step). Tried: ${summary}.`;
  } finally {
    el('suggest-settings-btn').disabled = false;
  }
});

el('create-model-btn').addEventListener('click', () => {
  const cfg = currentConfig();
  if (cfg.hiddenDim % cfg.numHeads !== 0) {
    alert(`Attention heads (${cfg.numHeads}) must evenly divide nodes (${cfg.hiddenDim}).`);
    return;
  }
  if ((cfg.hiddenDim / cfg.numHeads) % 2 !== 0) {
    alert(`Nodes / heads (${cfg.hiddenDim / cfg.numHeads}) must be even (needed for rotary position embeddings).`);
    return;
  }
  modelConfig = cfg;
  lossHistory = [];
  el('create-model-btn').disabled = true;
  el('create-model-btn').textContent = 'Creating…';
  worker.postMessage({ type: 'createModel', config: cfg });
  setTimeout(() => {
    el('create-model-btn').disabled = false;
    el('create-model-btn').textContent = 'Re-create model (resets training)';
  }, 300);
});

// --- Training ----------------------------------------------------------

el('start-train-btn').addEventListener('click', () => {
  training = true;
  setTrainingButtons(true);
  el('train-samples').innerHTML = '';
  stepMsHistory = [];
  el('train-perf').textContent = '';
  worker.postMessage({
    type: 'startTraining',
    batchSize: parseInt(el('cfg-batch').value, 10),
    lr: parseFloat(el('cfg-lr').value),
    useGpu: el('train-use-gpu').checked,
    sampleEveryN: el('train-sample-toggle').checked ? parseInt(el('train-sample-every').value, 10) : 0,
    samplePrompt: el('train-sample-prompt').value,
  });
});

el('stop-train-btn').addEventListener('click', () => {
  worker.postMessage({ type: 'stopTraining' });
});

function drawLossChart() {
  const canvas = el('loss-chart');
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (lossHistory.length < 2) return;

  const max = Math.max(...lossHistory);
  const min = Math.min(...lossHistory);
  const range = Math.max(max - min, 1e-6);
  const styles = getComputedStyle(document.documentElement);
  ctx.strokeStyle = styles.getPropertyValue('--accent').trim() || '#7a4b2a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((loss, i) => {
    const x = (i / (lossHistory.length - 1)) * width;
    const y = height - ((loss - min) / range) * (height - 10) - 5;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// --- Generation ----------------------------------------------------------

el('gen-use-gpu').addEventListener('change', () => {
  if (el('gen-use-gpu').checked) {
    setWebGpuStatus('WebGPU: initializing device…', '');
    el('gpu-status').textContent = 'Initializing WebGPU…';
    worker.postMessage({ type: 'initGpu' });
  }
});

function currentGenerationTags() {
  return {
    genre: el('gen-genre-tag').value.trim(),
    tone: el('gen-tone-tag').value.trim(),
  };
}

el('generate-btn').addEventListener('click', () => {
  const prompt = el('prompt-input').value;
  el('generate-btn').disabled = true;
  el('generate-output').textContent = '';
  renderQaNotes([]);
  el('effective-prompt-details').hidden = true;
  worker.postMessage({
    type: 'generate',
    prompt,
    maxNewTokens: parseInt(el('gen-max-tokens').value, 10),
    temperature: parseFloat(el('gen-temp').value),
    useGpu: el('gen-use-gpu').checked,
    tags: currentGenerationTags(),
    useStoryState: el('gen-use-story-state').checked,
    useRetrieval: el('gen-use-retrieval').checked,
    retrievalK: 3,
    targetWordCount: parseInt(el('gen-target-words').value, 10) || 0,
    seed: Date.now(),
  });
});

el('preview-retrieval-btn').addEventListener('click', () => {
  const query = el('prompt-input').value;
  if (!query.trim()) {
    alert('Type a prompt first — retrieval searches your sources for scenes similar to it.');
    return;
  }
  el('preview-retrieval-btn').disabled = true;
  worker.postMessage({ type: 'previewRetrieval', query, k: 3 });
});

el('debug-compare-btn').addEventListener('click', () => {
  worker.postMessage({ type: 'debugCompareGpuCpu', prompt: el('prompt-input').value || 'hello' });
});

el('debug-compare-gradient-btn').addEventListener('click', () => {
  worker.postMessage({ type: 'debugCompareGpuCpuGradient', prompt: el('prompt-input').value || 'hello' });
});

// --- Save / load -----------------------------------------------------------

function downloadWeights(bytes, step) {
  const blob = new Blob([bytes], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `scriptonait-weights-step${Math.round(step)}.bin`;
  a.click();
  URL.revokeObjectURL(url);
}

el('download-weights-btn').addEventListener('click', () => {
  const handler = (event) => {
    if (event.data.type !== 'weightsExported') return;
    worker.removeEventListener('message', handler);
    downloadWeights(event.data.bytes, event.data.step);
  };
  worker.addEventListener('message', handler);
  worker.postMessage({ type: 'exportWeights' });
});

el('upload-weights-input').addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const bytes = new Uint8Array(await file.arrayBuffer());
  worker.postMessage({ type: 'importWeights', bytes }, [bytes.buffer]);
  event.target.value = '';
});

async function refreshModelSelect() {
  const models = await db.listModels();
  const select = el('load-model-select');
  select.innerHTML = '';
  if (models.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'No saved checkpoints';
    opt.disabled = true;
    select.appendChild(opt);
    return;
  }
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = `${m.name} (step ${m.step}, ${new Date(m.createdAt).toLocaleString()})`;
    select.appendChild(opt);
  }
}

el('save-model-btn').addEventListener('click', () => {
  const name = el('save-name').value.trim();
  if (!name) {
    alert('Give this checkpoint a name first.');
    return;
  }
  if (!modelConfig) {
    alert('Create a model first.');
    return;
  }
  const handler = (event) => {
    if (event.data.type !== 'weightsExported') return;
    worker.removeEventListener('message', handler);
    db.saveModel({ name, config: modelConfig, weightBytes: event.data.bytes.buffer, step: event.data.step })
      .then(refreshModelSelect);
  };
  worker.addEventListener('message', handler);
  worker.postMessage({ type: 'exportWeights' });
});

el('load-model-btn').addEventListener('click', async () => {
  const id = el('load-model-select').value;
  if (!id) return;
  const record = await db.getModel(id);
  if (!record) return;
  modelConfig = record.config;
  el('cfg-layers').value = record.config.numLayers;
  el('cfg-hidden').value = record.config.hiddenDim;
  el('cfg-heads').value = record.config.numHeads;
  el('cfg-context').value = record.config.contextLen;
  el('cfg-window').value = record.config.localWindow;
  updateSizeEstimate();

  worker.postMessage({ type: 'createModel', config: record.config });
  const handler = (event) => {
    if (event.data.type !== 'modelCreated') return;
    worker.removeEventListener('message', handler);
    worker.postMessage({ type: 'importWeights', bytes: new Uint8Array(record.weightBytes) });
  };
  worker.addEventListener('message', handler);
});

el('delete-model-btn').addEventListener('click', async () => {
  const id = el('load-model-select').value;
  if (!id) return;
  await db.deleteModel(id);
  await refreshModelSelect();
});

// --- Init --------------------------------------------------------------

updateSizeEstimate();
refreshSourcesList();
refreshModelSelect();
