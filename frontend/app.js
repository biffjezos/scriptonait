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
    const error = new Error(event.data.message);
    if (event.data.stack) error.stack = event.data.stack;
    const entry = pending.get(id);
    if (entry) {
      pending.delete(id);
      entry.reject(error);
    } else {
      showError(error);
    }
    return;
  }
  const handler = streamHandlers.get(type);
  if (handler) handler(event.data);
};

worker.onerror = (event) => showError(event.message || 'the worker failed');

// --- Small helpers -----------------------------------------------------

const $ = (id) => document.getElementById(id);

/// Show a failure, with enough of the stack to act on.
///
/// An error message on its own is often useless — "Cannot read
/// properties of undefined (reading 'length')" says nothing about which
/// call produced it. The first frame does, so it goes on screen, and the
/// whole thing goes to the console.
function showError(error) {
  const banner = $('error-banner');
  const message = typeof error === 'string' ? error : error.message;
  const frame = typeof error === 'object' && error && error.stack
    ? String(error.stack).split('\n').slice(1).find((line) => line.trim()) || ''
    : '';
  banner.textContent = frame ? `${message}  (${frame.trim()})` : message;
  banner.hidden = false;
  if (typeof error === 'object') console.error('scriptonait:', error);
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
  // The device is stated, never chosen. It's a fact about the machine,
  // and the first question anyone asks about speed.
  const where = info.usingGpu ? `running on ${info.device}` : 'running on the CPU (no WebGPU here)';
  setModelStatus(
    'ready',
    info.pretrained
      ? `Ready — ${params} parameters, trained for ${formatCount(info.step)} steps, ${where}.`
      : `Untrained model created (${params} parameters), ${where}. It will write nonsense until you fine-tune it.`,
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
      <div><dt>Generating on</dt><dd>${escapeHtml(info.device || 'CPU')}</dd></div>
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

/// Download the shipped checkpoint. Only ever called from a click.
async function loadModel() {
  try {
    const bytes = await downloadModel(MODEL_URL);
    // Transferred, not copied: the checkpoint is tens of MB.
    const info = await call('load-model', { bytes: bytes.buffer }, [bytes.buffer]);
    renderModel(info);
  } catch (error) {
    // Only claim there's no model if there still isn't one. A slow 404
    // here used to land *after* the user had loaded a model file by
    // hand and overwrite "Ready" with "no model published" — the page
    // then looked broken while holding a perfectly good model.
    if (!model) {
      setModelStatus(
        'absent',
        `${error.message}. You can still add your own material below, load a ` +
          'model file, or create an untrained one under "The model".',
      );
      $('model-panel').open = true;
    }
  } finally {
    $('model-progress').hidden = true;
    if (!generating && !training) setTitleProgress(null);
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

onStream('gpu-status', ({ available, device, reason }) => {
  // Logged rather than shown: someone debugging why generation is slow
  // wants the reason, and everyone else has it in the status line.
  if (available) {
    console.info(`scriptonait: generating on ${device}`);
  } else {
    console.info(`scriptonait: no WebGPU (${reason || 'unavailable'}), generating on the CPU`);
  }
});

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

$('load-model-btn').addEventListener('click', async () => {
  $('load-model-btn').disabled = true;
  await loadModel();
  $('load-model-btn').disabled = false;
});

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
//
// The list you see is this array. Not the database.
//
// It used to be the database, and that was the bug: every add did a
// write, then a read, then a re-render, so anything that made IndexedDB
// slow, blocked or unavailable — another tab holding an old connection,
// a private window, storage pressure — froze the whole batch on the
// first file. Thirty files went in, one appeared, and a reload was the
// only way to find out what had actually been stored.
//
// Now: read the file, put it in this array, draw it. Persisting it and
// handing it to the model are both things that happen afterwards and are
// both allowed to fail without you losing sight of your own files.

let sources = [];

/// Persistence, entirely best-effort.
///
/// Every call is time-boxed, because the failure that matters here isn't
/// an error — it's a promise that never settles. The first time one does
/// that, persistence switches off for the session and the page says so
/// once, rather than hanging on every subsequent file.
let persistenceWorks = true;
const DB_TIMEOUT_MS = 4000;

/// Writes are chained rather than awaited by the caller, so a slow or
/// dead IndexedDB can never hold up the next file. This is the whole
/// reason thirty files used to show up as one: each add waited on its
/// own write, and a write that hangs for four seconds thirty times over
/// is two minutes of an apparently frozen page.
let persistChain = Promise.resolve();

function persistLater(what, action) {
  persistChain = persistChain.then(() => persist(what, action));
  return persistChain;
}

async function persist(what, action) {
  if (!persistenceWorks) return null;
  try {
    return await Promise.race([
      action(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`${what} timed out`)), DB_TIMEOUT_MS),
      ),
    ]);
  } catch (error) {
    persistenceWorks = false;
    showError(
      `Couldn't save to this browser's storage (${error.message}). Your files are ` +
        'loaded and usable, but they won\'t still be here after a reload.',
    );
    return null;
  }
}

function newSourceId() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/// Draw the list from memory. Synchronous on purpose: nothing it needs
/// can be slow, so nothing can stop it running.
function renderSources() {
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
      button.addEventListener('click', () => removeSource(button.dataset.id));
    }
  }
  $('train-btn').disabled = !model || sources.length === 0;
  updateSourceSummary(sources);
}

async function removeSource(id) {
  sources = sources.filter((source) => source.id !== id);
  renderSources();
  await persist('deleting a source', () => db.deleteSource(id));
  try {
    renderModel(await call('remove-source', { id }));
    await refreshStoryState();
  } catch (error) {
    /* no model loaded: it was only ever in the list */
  }
}

/// Load whatever was stored in a previous session and merge it in.
///
/// Merge, not replace. This read is slow — it waits on IndexedDB opening
/// — and files can be added while it's still in flight. Assigning its
/// result would then delete them, which is a race that looks exactly
/// like "I added thirty files and they vanished, and a reload brought
/// some back". Anything already in the list wins; stored records only
/// fill in what isn't there.
async function refreshSources() {
  const stored = await persist('reading saved sources', () => db.listSources());
  if (Array.isArray(stored)) {
    const have = new Set(sources.map((source) => source.id));
    const restored = stored.filter(
      (source) => typeof source.rawText === 'string' && !have.has(source.id),
    );
    sources = [...restored, ...sources];
  }
  renderSources();
}

/// Say how much material is loaded, and whether the model has seen it.
function updateSourceSummary(list, note = '') {
  const stats = $('corpus-stats');
  if (!list || list.length === 0) {
    stats.textContent = note || 'Nothing added yet.';
    return;
  }
  const chars = list.reduce((sum, s) => sum + (s.rawText || '').length, 0);
  const seen = model ? '' : ' — no model loaded, so nothing is using them yet';
  const saved = persistenceWorks ? '' : ' · not saved (storage unavailable)';
  stats.textContent =
    `${list.length} source${list.length === 1 ? '' : 's'}, ` +
    `${formatCount(chars)} characters${seen}${saved}${note ? ` · ${note}` : ''}`;
}

/// Hand one source to the model. Best effort: with no model loaded this
/// is expected to fail, and failing must not stop anything.
async function syncSource(source) {
  if (typeof source.rawText !== 'string' || source.rawText.length === 0) {
    return false;
  }
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

/// Add sources one at a time: read it, show it, then try to save it and
/// try to give it to the model. Each entry is `{ title, kind, read() }`.
///
/// The order matters. Reading and displaying come first and depend on
/// nothing; storage and the model come after and are both allowed to
/// fail. One unreadable file is reported by name and the rest continue.
async function addSources(entries) {
  clearError();
  const failures = [];
  const syncing = [];
  let added = 0;

  for (const [index, entry] of entries.entries()) {
    if (entries.length > 1) {
      updateSourceSummary(sources, `reading ${index + 1} of ${entries.length}…`);
    }
    let rawText;
    try {
      rawText = await entry.read();
    } catch (error) {
      failures.push(`${entry.title} (${error.message})`);
      continue;
    }
    if (typeof rawText !== 'string' || rawText.length === 0) {
      failures.push(`${entry.title} (empty)`);
      continue;
    }

    const source = {
      id: newSourceId(),
      title: entry.title,
      kind: entry.kind,
      rawText,
      sourceUrl: entry.sourceUrl || null,
      createdAt: Date.now(),
    };
    sources.push(source);
    added += 1;
    // Drawn immediately, and nothing below is awaited: the next file
    // starts reading straight away. Saving and handing it to the model
    // both catch up on their own.
    renderSources();
    persistLater('saving a source', () => db.putSource(source));
    syncing.push(syncSource(source));
  }

  // Only now, once every file is on screen, wait for the model to have
  // caught up — and only for that, never for storage.
  await Promise.allSettled(syncing);
  renderSources();
  await refreshStoryState();
  if (failures.length) {
    showError(
      `${failures.length} of ${entries.length} couldn't be added: ${failures.slice(0, 3).join(', ')}` +
        (failures.length > 3 ? ', …' : ''),
    );
  }
  updateSourceSummary(sources, added ? `added ${added}` : '');
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
    // A fresh model has an empty corpus; re-feed what's loaded.
    for (const source of sources) {
      await syncSource(source);
    }
    renderSources();
    await refreshStoryState();
  } catch (error) {
    showError(error.message);
    setModelStatus('absent', 'No model loaded.');
  }
});

// --- Start -------------------------------------------------------------

(async function start() {
  // Nothing is fetched here. The page loads, shows what you already
  // have, and waits. Downloading an 18 MB file because a page opened is
  // not a thing this should do to you.
  await refreshSources();
  setModelStatus(
    'absent',
    'No model loaded. Use "Load the writing model" below to download the ' +
      'one built from public-domain books, or load a model file you already have.',
  );
  updateSourceSummary(sources);
})();
