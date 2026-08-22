// Main-thread UI. All the expensive work is in worker.js; this file is
// about telling the user what is happening.
//
// That is the actual feature here. The previous version reported a loss
// number and nothing else, so a job that took twenty minutes was
// indistinguishable from a job that had crashed. Now every long operation
// has: a progress bar, a throughput figure, an elapsed time, a stop
// button, a title-bar indicator for when the tab is in the background,
// and — if you ask for it — a notification when it finishes.

import * as db from './db.js';

const worker = new Worker('./worker.js', { type: 'module' });

// --- Worker plumbing ---------------------------------------------------
// Requests are promise-shaped; streaming updates arrive out-of-band and
// are dispatched to whatever handler is currently interested.

let nextRequestId = 1;
const pending = new Map();
const streamHandlers = new Map();

// Long jobs (generation, fine-tuning) legitimately run for minutes and
// opt out with `timeoutMs: 0`. Everything else gets a deadline, because
// a request that never settles is the worst possible failure: the UI
// waits forever, shows nothing, and says nothing. That is exactly what
// happened when the worker's own fetch hung — the page sat on "Loading
// the model..." and every subsequent action queued up behind it in
// silence.
const DEFAULT_TIMEOUT_MS = 60000;

function call(type, payload = {}, transfer = [], timeoutMs = DEFAULT_TIMEOUT_MS) {
  const id = nextRequestId++;
  return new Promise((resolve, reject) => {
    let timer = null;
    const settle = (fn) => (value) => {
      if (timer) clearTimeout(timer);
      pending.delete(id);
      fn(value);
    };
    pending.set(id, { resolve: settle(resolve), reject: settle(reject) });
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`the worker didn't answer "${type}" within ${Math.round(timeoutMs / 1000)}s`));
      }, timeoutMs);
    }
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

function onStream(type, handler) {
  streamHandlers.set(type, handler);
}

worker.onmessage = (event) => {
  const { type, id } = event.data;
  if (type === 'result') {
    const entry = pending.get(id);
    if (entry) {
      pending.delete(id);
      entry.resolve(event.data.result);
    }
    return;
  }
  if (type === 'error') {
    const entry = pending.get(id);
    if (entry) {
      pending.delete(id);
      entry.reject(new Error(event.data.message));
    } else {
      showError(event.data.message);
    }
    return;
  }
  const handler = streamHandlers.get(type);
  if (handler) handler(event.data);
};

worker.onerror = (event) => showError(event.message || 'the worker failed');

// --- Small helpers -----------------------------------------------------

const $ = (id) => document.getElementById(id);

function showError(message) {
  const banner = $('error-banner');
  banner.textContent = message;
  banner.hidden = false;
}

function clearError() {
  $('error-banner').hidden = true;
}

function formatCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

function formatDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m${String(s).padStart(2, '0')}s`;
}

function setProgress(barId, fraction) {
  const clamped = Math.max(0, Math.min(1, fraction || 0));
  $(barId).style.width = `${(clamped * 100).toFixed(1)}%`;
}

// --- Title-bar and notifications --------------------------------------
//
// A background tab shows only its title, so that's where progress goes
// when the page isn't visible. Notifications need a user gesture to ask
// for permission, so the checkbox asks when it's ticked rather than on
// page load — an unprompted permission dialog is exactly the thing
// everyone has learned to dismiss.

const BASE_TITLE = document.title;

function setTitleProgress(label, fraction) {
  if (!label) {
    document.title = BASE_TITLE;
    return;
  }
  const percent = fraction > 0 ? ` ${(fraction * 100).toFixed(0)}%` : '';
  document.title = `${label}${percent} — ${BASE_TITLE}`;
}

$('notify-toggle').addEventListener('change', async (event) => {
  if (!event.target.checked) return;
  if (!('Notification' in window)) {
    event.target.checked = false;
    showError("this browser doesn't support notifications");
    return;
  }
  if (Notification.permission === 'default') {
    const result = await Notification.requestPermission();
    if (result !== 'granted') {
      event.target.checked = false;
      showError('notifications were blocked — the page will still show progress');
    }
  } else if (Notification.permission === 'denied') {
    event.target.checked = false;
    showError('notifications are blocked for this site in your browser settings');
  }
});

function notify(title, body) {
  if (!$('notify-toggle').checked) return;
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  // Only worth interrupting for if the user isn't already looking at it.
  if (document.visibilityState === 'visible') return;
  try {
    new Notification(title, { body, tag: 'scriptonait' });
  } catch (error) {
    // Some browsers refuse constructor notifications outside a service
    // worker. Not worth surfacing — the in-page status still updated.
  }
}

// --- Model state -------------------------------------------------------

let model = null;
let generating = false;
let training = false;

const MODEL_URL = './model/scriptonait.ckpt';

function setModelStatus(state, text) {
  const el = $('model-status');
  el.className = `model-status ${state}`;
  $('model-status-text').textContent = text;
}

function renderModel(info) {
  model = info;
  $('generate-btn').disabled = !info;
  $('train-btn').disabled = !info;
  if (!info) return;

  const params = formatCount(info.params);
  setModelStatus(
    'ready',
    info.pretrained
      ? `Ready — ${params} parameters, trained for ${formatCount(info.step)} steps.`
      : `Untrained model created (${params} parameters). It will write nonsense until you fine-tune it.`,
  );
  $('model-details').innerHTML = `
    <dl>
      <div><dt>Parameters</dt><dd>${params}</dd></div>
      <div><dt>Layers</dt><dd>${info.layers}</dd></div>
      <div><dt>Hidden size</dt><dd>${info.hidden}</dd></div>
      <div><dt>Heads</dt><dd>${info.heads} (${info.kvHeads} key/value)</dd></div>
      <div><dt>Context</dt><dd>${info.contextLen} tokens, ${info.window}-token attention window</dd></div>
      <div><dt>Vocabulary</dt><dd>${info.vocabSize} tokens</dd></div>
      <div><dt>Training steps</dt><dd>${formatCount(info.step)}</dd></div>
    </dl>`;
  $('corpus-stats').textContent = info.sources
    ? `${info.sources} source${info.sources === 1 ? '' : 's'}, ${formatCount(info.corpusTokens)} tokens`
    : 'No sources added yet.';
}

/// Download the shipped checkpoint, reporting progress as it streams.
async function downloadModel(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(`no model published at ${url} (HTTP ${response.status})`);
  }
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body && response.body.getReader ? response.body.getReader() : null;
  if (!reader) {
    return new Uint8Array(await response.arrayBuffer());
  }
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const fraction = total > 0 ? received / total : 0;
    $('model-progress').hidden = false;
    setProgress('model-progress-bar', fraction);
    setModelStatus(
      'loading',
      `Downloading the model — ${(received / 1e6).toFixed(1)}` +
        `${total > 0 ? ` of ${(total / 1e6).toFixed(1)}` : ''} MB`,
    );
    setTitleProgress('Loading', fraction);
  }
  const bytes = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.length;
  }
  return bytes;
}

async function loadModel() {
  try {
    const bytes = await downloadModel(MODEL_URL);
    // Transferred, not copied: the checkpoint is tens of MB.
    const info = await call('load-model', { bytes: bytes.buffer }, [bytes.buffer]);
    renderModel(info);
  } catch (error) {
    setModelStatus(
      'absent',
      `${error.message}. You can still add your own material below, load a ` +
        'model file, or create an untrained one under "The model".',
    );
    $('model-panel').open = true;
  } finally {
    $('model-progress').hidden = true;
    setTitleProgress(null);
  }
}

// --- Prompt understanding ---------------------------------------------

let parseTimer = null;

$('prompt-input').addEventListener('input', () => {
  clearTimeout(parseTimer);
  // Debounced: this is a round trip to the worker on every keystroke
  // otherwise, and it's only there to reassure.
  parseTimer = setTimeout(updateUnderstanding, 250);
});

async function updateUnderstanding() {
  const prompt = $('prompt-input').value.trim();
  const box = $('understood');
  if (!prompt) {
    box.hidden = true;
    return;
  }
  try {
    const parsed = await call('parse-prompt', { prompt });
    const chips = [];
    chips.push(`<span class="chip">${parsed.form === 'any' ? 'form: your choice' : parsed.form}</span>`);
    if (parsed.targetWords) chips.push(`<span class="chip">${parsed.targetWords} words</span>`);
    if (parsed.subject) chips.push(`<span class="chip">about: ${escapeHtml(parsed.subject)}</span>`);
    if (parsed.reference) chips.push(`<span class="chip">echoing: ${escapeHtml(parsed.reference)}</span>`);
    box.innerHTML = `<span class="understood-label">Understood as</span>${chips.join('')}`;
    box.hidden = false;
  } catch (error) {
    box.hidden = true;
  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

// --- Generating --------------------------------------------------------

let targetWords = 0;

onStream('generate-piece', ({ piece }) => {
  const output = $('output');
  output.hidden = false;
  output.textContent += piece;
  // Only follow the text if the user hasn't scrolled up to read.
  const nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 40;
  if (nearBottom) output.scrollTop = output.scrollHeight;
});

onStream('generate-progress', ({ words, tokens, elapsedSeconds, tokensPerSecond }) => {
  const fraction = targetWords > 0 ? words / targetWords : 0;
  setProgress('generate-progress-bar', fraction);
  const of = targetWords > 0 ? ` of ${targetWords}` : '';
  $('generate-stats').textContent =
    `${words} words${of} · ${tokens} tokens · ${tokensPerSecond.toFixed(0)} tokens/s · ${formatDuration(elapsedSeconds)}`;
  setTitleProgress('Writing', fraction);
});

$('generate-btn').addEventListener('click', async () => {
  if (generating) return;
  const prompt = $('prompt-input').value.trim();
  if (!prompt) {
    showError('type what you want written first');
    return;
  }
  clearError();
  generating = true;
  $('generate-btn').disabled = true;
  $('stop-btn').hidden = false;
  $('generate-status').hidden = false;
  $('qa-notes').hidden = true;
  $('output').textContent = '';
  setProgress('generate-progress-bar', 0);
  $('generate-stats').textContent = 'Starting…';

  const parsed = await call('parse-prompt', { prompt });
  targetWords = parsed.targetWords;

  try {
    const result = await call('generate', {
      prompt,
      temperature: Number($('opt-temperature').value),
      topK: Number($('opt-top-k').value),
      topP: Number($('opt-top-p').value),
      repetitionPenalty: Number($('opt-repetition').value),
      seed: Number($('opt-seed').value) || Math.floor(Math.random() * 1e9),
      useStoryState: $('use-story-state').checked,
      useRetrieval: $('use-retrieval').checked,
    }, [], 0);
    setProgress('generate-progress-bar', 1);
    const why = {
      'end-of-text': 'the model ended the piece',
      length: 'reached the length you asked for',
      stopped: 'stopped',
    }[result.stopReason] || result.stopReason;
    $('generate-stats').textContent =
      `${result.wordCount} words in ${formatDuration(result.elapsedSeconds)} · ` +
      `${result.tokensPerSecond.toFixed(0)} tokens/s · ${why}`;
    renderNotes(result.notes);
    notify('Your piece is ready', `${result.wordCount} words — ${why}.`);
  } catch (error) {
    showError(error.message);
  } finally {
    generating = false;
    $('generate-btn').disabled = false;
    $('stop-btn').hidden = true;
    setTitleProgress(null);
  }
});

$('stop-btn').addEventListener('click', () => call('stop'));

function renderNotes(notes) {
  const box = $('qa-notes');
  if (!notes || notes.length === 0) {
    box.hidden = true;
    return;
  }
  box.innerHTML = notes
    .map((note) => {
      const warning = note.startsWith('[WARNING]');
      const text = note.replace(/^\[(INFO|WARNING)\]\s*/, '');
      return `<p class="note${warning ? ' warning' : ''}">${escapeHtml(text)}</p>`;
    })
    .join('');
  box.hidden = false;
}

// --- Sources -----------------------------------------------------------

/// Render the source list from IndexedDB.
///
/// From the database, not from the model — that distinction is the whole
/// bug fix here. Adding a source used to mean: write it to IndexedDB,
/// then await the worker, then re-render. With no model loaded (or a
/// worker that never answered) the await never returned, so the loop
/// stopped on the first file, the list never re-rendered, and the only
/// way to see what had actually been saved was to reload the page.
/// Uploading thirty files added thirty database rows and displayed one.
///
/// Your material belongs to you and to the database. Handing it to the
/// model is a separate, best-effort step.
async function refreshSources() {
  const sources = await db.listSources();
  const list = $('sources-list');
  if (sources.length === 0) {
    list.innerHTML = '<p class="empty-hint">Nothing added yet.</p>';
  } else {
    list.innerHTML = sources
      .map(
        (source) => `
        <div class="source-item" data-id="${source.id}">
          <div class="meta">
            <span class="title">${escapeHtml(source.title)}</span>
            <span class="kind-badge">${source.kind}</span>
            <span class="stats">${formatCount((source.rawText || '').length)} chars</span>
          </div>
          <button type="button" class="secondary remove-source" data-id="${source.id}">Remove</button>
        </div>`,
      )
      .join('');
    for (const button of list.querySelectorAll('.remove-source')) {
      button.addEventListener('click', async () => {
        await db.deleteSource(button.dataset.id);
        await refreshSources();
        try {
          renderModel(await call('remove-source', { id: button.dataset.id }));
          await refreshStoryState();
        } catch (error) {
          /* no model: it was only ever in the database anyway */
        }
      });
    }
  }
  $('train-btn').disabled = !model || sources.length === 0;
  updateSourceSummary(sources);
}

/// Say how much material is stored, and whether the model has seen it.
function updateSourceSummary(sources, note = '') {
  const stats = $('corpus-stats');
  if (sources.length === 0) {
    stats.textContent = note || 'No sources added yet.';
    return;
  }
  const chars = sources.reduce((sum, s) => sum + (s.rawText || '').length, 0);
  const seen = model ? '' : ' — no model loaded yet, so nothing is using them';
  stats.textContent =
    `${sources.length} source${sources.length === 1 ? '' : 's'}, ` +
    `${formatCount(chars)} characters${seen}${note ? ` · ${note}` : ''}`;
}

/// Hand one stored source to the model. Best effort: with no model
/// loaded this is expected to fail, and failing here must not stop
/// anything.
async function syncSource(source) {
  try {
    const result = await call('upsert-source', {
      id: source.id,
      text: source.rawText,
      isHtml: source.kind === 'url',
    });
    if (result && result.model) renderModel(result.model);
    return true;
  } catch (error) {
    return false;
  }
}

/// Add sources one at a time, reporting progress and surviving
/// individual failures.
///
/// Each entry is a `{ title, kind, read() }`, where `read()` returns the
/// text. Reading is deferred into this loop on purpose: with thirty
/// files, reading all thirty before storing any means the list stays
/// empty for the whole batch, and one unreadable file loses the lot.
/// Here each file is read, stored, and *rendered* before the next one
/// starts, so thirty files look like thirty files arriving.
async function addSources(entries) {
  clearError();
  const failures = [];
  let added = 0;
  for (const [index, entry] of entries.entries()) {
    const progress = entries.length > 1 ? `adding ${index + 1} of ${entries.length}…` : 'adding…';
    updateSourceSummary(await db.listSources(), progress);
    try {
      const rawText = await entry.read();
      const record = await db.addSource({
        title: entry.title,
        kind: entry.kind,
        rawText,
        sourceUrl: entry.sourceUrl || null,
      });
      added += 1;
      await refreshSources();
      await syncSource(record);
    } catch (error) {
      failures.push(`${entry.title}: ${error.message}`);
    }
  }
  await refreshSources();
  await refreshStoryState();
  if (failures.length) {
    showError(
      failures.length === 1
        ? `couldn't add ${failures[0]}`
        : `${failures.length} of ${entries.length} couldn't be added — first was ${failures[0]}`,
    );
  }
  updateSourceSummary(await db.listSources(), added ? `added ${added} just now` : '');
}

async function refreshStoryState() {
  const box = $('story-state');
  try {
    const state = await call('story-state');
    if (!state.characters.length && !state.locations.length) {
      box.hidden = true;
      return;
    }
    const parts = [];
    if (state.characters.length) parts.push(`<strong>Characters:</strong> ${escapeHtml(state.characters.join(', '))}`);
    if (state.locations.length) parts.push(`<strong>Locations:</strong> ${escapeHtml(state.locations.join(', '))}`);
    if (state.sceneCount) parts.push(`<strong>Scenes:</strong> ${state.sceneCount}`);
    box.innerHTML = `${parts.join('<br />')}<p class="hint">Found by looking at line shapes, not by understanding the text — unusual formatting can fool it.</p>`;
    box.hidden = false;
  } catch (error) {
    box.hidden = true;
  }
}

$('add-paste-btn').addEventListener('click', async () => {
  const text = $('paste-input').value.trim();
  if (!text) return;
  const title = $('paste-title').value.trim() || `Pasted ${new Date().toLocaleString()}`;
  await addSources([{ title, kind: 'paste', read: () => text }]);
  $('paste-input').value = '';
  $('paste-title').value = '';
});

$('file-input').addEventListener('change', async (event) => {
  const files = Array.from(event.target.files);
  // Cleared immediately so picking the same files again still fires a
  // change event, and so a slow batch can't be submitted twice.
  event.target.value = '';
  await addSources(
    files.map((file) => ({ title: file.name, kind: 'file', read: () => file.text() })),
  );
});

$('add-url-btn').addEventListener('click', async () => {
  const url = $('url-input').value.trim();
  if (!url) return;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`the server answered ${response.status}`);
    const text = await response.text();
    await addSources([{ title: url, kind: 'url', read: () => text, sourceUrl: url }]);
    $('url-input').value = '';
  } catch (error) {
    showError(`couldn't fetch that URL (${error.message}). Most sites block cross-origin fetches — paste the text instead.`);
  }
});

// --- Fine-tuning -------------------------------------------------------

const lossHistory = [];

onStream('train-progress', (progress) => {
  setProgress('train-progress-bar', progress.fractionDone);
  $('train-stats').textContent =
    `step ${formatCount(progress.step)} · loss ${progress.smoothedLoss.toFixed(3)} · ` +
    `${progress.tokensPerSecond.toFixed(0)} tokens/s · ${formatDuration(progress.elapsedSeconds)} elapsed`;
  setTitleProgress('Fine-tuning', progress.fractionDone);
  lossHistory.push(progress.smoothedLoss);
  drawLossChart();
});

$('train-btn').addEventListener('click', async () => {
  if (training) return;
  clearError();
  training = true;
  lossHistory.length = 0;
  $('train-btn').disabled = true;
  $('train-stop-btn').hidden = false;
  $('train-status').hidden = false;
  $('loss-chart').hidden = false;
  $('train-stats').textContent = 'Starting…';

  try {
    const result = await call('train', {
      batchSize: Number($('train-batch').value),
      learningRate: Number($('train-lr').value),
      maxSteps: Number($('train-steps').value),
      effort: Number($('train-effort').value),
    }, [], 0);
    if (result.stopReason === 'no-data') {
      showError('there isn\'t enough source text yet to fill even one context window — add more.');
    } else {
      setProgress('train-progress-bar', 1);
      $('train-stats').textContent =
        `${result.steps} steps in ${formatDuration(result.elapsedSeconds)} · final loss ${(result.loss ?? NaN).toFixed(3)}`;
      notify('Fine-tuning finished', `${result.steps} steps, loss ${(result.loss ?? NaN).toFixed(3)}.`);
    }
    if (result.model) renderModel(result.model);
  } catch (error) {
    showError(error.message);
  } finally {
    training = false;
    $('train-btn').disabled = false;
    $('train-stop-btn').hidden = true;
    setTitleProgress(null);
  }
});

$('train-stop-btn').addEventListener('click', () => call('stop'));

function drawLossChart() {
  const canvas = $('loss-chart');
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (lossHistory.length < 2) return;

  const min = Math.min(...lossHistory);
  const max = Math.max(...lossHistory);
  const span = max - min || 1;
  ctx.strokeStyle = '#7aa2f7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((loss, i) => {
    const x = (i / (lossHistory.length - 1)) * width;
    const y = height - ((loss - min) / span) * (height - 12) - 6;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#8891a8';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(max.toFixed(3), 4, 12);
  ctx.fillText(min.toFixed(3), 4, height - 4);
}

// --- Saving and loading models ----------------------------------------

$('export-btn').addEventListener('click', async () => {
  const { bytes } = await call('export-checkpoint');
  const blob = new Blob([bytes], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'scriptonait.ckpt';
  link.click();
  URL.revokeObjectURL(url);
});

$('import-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  clearError();
  setModelStatus('loading', `Loading ${file.name}…`);
  try {
    const buffer = await file.arrayBuffer();
    renderModel(await call('import-checkpoint', { bytes: buffer }, [buffer]));
  } catch (error) {
    showError(`that file didn't load: ${error.message}`);
    setModelStatus('absent', 'No model loaded.');
  }
  event.target.value = '';
});

$('create-model-btn').addEventListener('click', async () => {
  clearError();
  setModelStatus('loading', 'Creating an untrained model…');
  try {
    renderModel(
      await call('create-model', {
        layers: Number($('cfg-layers').value),
        hidden: Number($('cfg-hidden').value),
        heads: Number($('cfg-heads').value),
        kvHeads: Number($('cfg-kv-heads').value),
        contextLen: Number($('cfg-context').value),
        window: Number($('cfg-window').value),
        seed: Math.floor(Math.random() * 1e9),
      }),
    );
    // A fresh model has an empty corpus; re-feed whatever is stored.
    for (const source of await db.listSources()) {
      await call('upsert-source', { id: source.id, text: source.rawText, isHtml: source.kind === 'url' });
    }
    await refreshSources();
    await refreshStoryState();
  } catch (error) {
    showError(error.message);
    setModelStatus('absent', 'No model loaded.');
  }
});

// --- Start -------------------------------------------------------------

(async function start() {
  // Sources render before anything touches the model, so the page is
  // useful (and visibly alive) even if the model never arrives.
  await refreshSources();
  await loadModel();
  if (model) {
    for (const source of await db.listSources()) {
      await syncSource(source);
    }
    try {
      renderModel(await call('model-info'));
    } catch (error) {
      /* the banner already says what went wrong */
    }
    await refreshStoryState();
  }
  updateSourceSummary(await db.listSources());
})();
