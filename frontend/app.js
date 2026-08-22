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
    // The payload goes in its own field rather than being spread
    // alongside the request id. It used to be `{ id, type, ...payload }`,
    // and a payload carrying its own `id` — every upsert-source does —
    // overwrote the request id with it. The worker then took that as the
    // request id, so the source id vanished ("the source id was
    // missing"), and the reply came back under an id no caller
    // recognised, so that call never settled. One key collision, three
    // symptoms.
    worker.postMessage({ rid: id, type, payload }, transfer);
  });
}

function onStream(type, handler) {
  streamHandlers.set(type, handler);
}

worker.onmessage = (event) => {
  const { type, rid: id } = event.data;
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

worker.onerror = (event) =>
  showError(
    `${event.message || 'the worker failed'} (${event.filename || 'worker'}:${event.lineno || '?'})`,
  );

// Anything that escapes a handler lands here rather than only in the
// console. An error nobody sees is a page that "does nothing"; an error
// with a stack frame on it is a bug report.
window.addEventListener('error', (event) => {
  showError(event.error || new Error(`${event.message} (${event.filename}:${event.lineno})`));
});
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  showError(reason instanceof Error ? reason : new Error(String(reason)));
});

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

function setModelStatus(state, text) {
  const el = $('model-status');
  el.className = `model-status ${state}`;
  $('model-status-text').textContent = text;
}

/// Say, in one sentence, what to do next.
///
/// The page has three things you can do and they have an order, but
/// which one applies depends on what you've already got. Rather than
/// making that a puzzle, this states it, and disables the buttons that
/// can't work yet.
function updateGuidance() {
  const step = $('next-step');
  const words = sources.reduce((sum, s) => sum + (s.rawText || '').length, 0);
  const enoughText = words > 4000;

  $('train-btn').disabled = training || sources.length === 0;
  $('generate-btn').disabled = generating || !model;

  $('train-btn').textContent = model
    ? (model.pretrained ? 'Keep training on my writing' : 'Keep training this model')
    : 'Train a model on my writing';

  const explains = $('train-explains');
  explains.textContent = !model
    ? 'New model, from scratch. Slow — hours, not minutes.'
    : model.pretrained
      ? 'Nudges the loaded model toward your writing.'
      : 'Continues where it stopped.';

  if (training) {
    step.textContent = 'Training. Stop any time — progress is kept.';
  } else if (generating) {
    step.textContent = 'Writing…';
  } else if (!model && sources.length === 0) {
    step.textContent = 'Step 1: add your writing.';
  } else if (!model && !enoughText) {
    step.textContent = `Only ${formatCount(words)} characters. Add more, then train.`;
  } else if (!model) {
    step.textContent = 'Step 2: train.';
  } else if (model.step < 500) {
    step.textContent = 'Barely trained. Keep training, or try step 3.';
  } else {
    step.textContent = 'Ready. Step 3: type a prompt.';
  }
}

/// Hand the model everything already in the list.
///
/// Called whenever a model appears, because sources can be added before
/// one exists — that's the normal order now — and the model starts empty.
async function syncAllSources() {
  for (const source of sources) {
    await syncSource(source);
  }
}

function renderModel(info) {
  model = info;
  $('generate-btn').disabled = !info;
  $('train-btn').disabled = !info;
  if (!info) return;

  const params = formatCount(info.params);
  // The device is stated, never chosen. It's a fact about the machine,
  // and the first question anyone asks about speed.
  // wgpu reports things like " (BrowserWebGpu, Other)" — already
  // parenthesised, sometimes with an empty name in front. Unwrap one
  // enclosing pair rather than stripping brackets blindly, which left
  // the parentheses unbalanced.
  const device = (info.device || '').trim().replace(/^\((.*)\)$/, '$1').trim();
  const where = info.usingGpu
    ? `writing on your GPU${device ? ` (${device})` : ''}`
    : 'writing on the CPU — this browser has no WebGPU';
  setModelStatus(
    'ready',
    info.step > 0
      ? `Your model: ${params} parameters, trained ${formatCount(info.step)} steps, ${where}.`
      : `Your model: ${params} parameters, not trained yet, ${where}.`,
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
  updateGuidance();
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
  $('stop-btn').disabled = false;
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

// Stop is fire-and-forget. The worker can only read it when its message
// queue gets a turn, and a slow step or a long wasm call can hold that up
// for a while — with a deadline on the reply, waiting for one turns a
// working stop into "the worker didn't answer". The job itself reports
// how it ended.
$('stop-btn').addEventListener('click', () => {
  $('stop-btn').disabled = true;
  call('stop', {}, [], 0).catch(() => {});
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
    showError(`Storage unavailable (${error.message}). Files load but won't survive a reload.`);
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
  updateSourceSummary(sources);
  updateGuidance();
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
      (source) =>
        source && source.id && typeof source.rawText === 'string' && !have.has(source.id),
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
  // No model means nothing to hand it to. This used to be attempted
  // anyway, so adding thirty files with no model logged thirty
  // "no model loaded yet" errors — noise that buried the real ones.
  if (!model) return false;
  // A record with no id or no text can't be handed over: wasm-bindgen
  // reads `.length` off whatever it gets for a string parameter, and a
  // missing one throws from inside generated glue. Records like this
  // exist in databases written by earlier versions of this page.
  if (!source.id || typeof source.rawText !== 'string' || source.rawText.length === 0) {
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

// Samples from the model as it trains, newest first, so the top of the
// list is always the current state of the writing.
onStream('train-sample', ({ step, loss, text }) => {
  const box = $('train-samples');
  const block = document.createElement('div');
  block.className = 'train-sample';
  const head = document.createElement('div');
  head.className = 'train-sample-head';
  head.textContent = `step ${formatCount(step)}` +
    (typeof loss === 'number' ? ` · loss ${loss.toFixed(3)}` : '');
  const body = document.createElement('pre');
  body.textContent = text;
  block.append(head, body);
  box.prepend(block);
  while (box.children.length > 20) box.lastElementChild.remove();
});

$('train-btn').addEventListener('click', async () => {
  if (training) return;
  clearError();
  training = true;
  lossHistory.length = 0;
  $('train-btn').disabled = true;
  $('train-stop-btn').hidden = false;
  $('train-stop-btn').disabled = false;
  $('train-status').hidden = false;
  $('loss-chart').hidden = false;
  $('train-stats').textContent = 'Starting…';
  $('train-samples').replaceChildren();

  try {
    // One button, two jobs. With no model, make one first — nobody
    // should have to know that "create an untrained model" is a separate
    // step from "train it", because it never isn't.
    if (!model) {
      $('train-stats').textContent = 'Making a new model…';
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
      await syncAllSources();
    }

    const fromScratch = model && !model.pretrained;
    const chosenRate = Number($('train-lr').value);
    const result = await call('train', {
      batchSize: Number($('train-batch').value),
      // 0 means "pick one": a new model needs a rate large enough to
      // learn a language from nothing; a working one needs a small
      // enough rate not to forget it.
      learningRate: chosenRate > 0 ? chosenRate : (fromScratch ? 3e-4 : 5e-5),
      maxSteps: Number($('train-steps').value),
      effort: Number($('train-effort').value),
      // 0 turns sampling off; anything else is a step interval.
      sampleEvery: $('sample-toggle').checked ? Number($('sample-every').value) : 0,
      samplePrompt: $('sample-prompt').value.trim() || 'Write a 40 word scene.',
      sampleWords: 40,
    }, [], 0);

    if (result.stopReason === 'no-data') {
      showError('Not enough text to train on. Add more in step 1.');
    } else {
      setProgress('train-progress-bar', 1);
      const loss = typeof result.loss === 'number' ? result.loss.toFixed(3) : '—';
      $('train-stats').textContent =
        `${result.steps} steps in ${formatDuration(result.elapsedSeconds)} · loss ${loss}` +
        ' · press Train again to continue';
      notify('Training finished', `${result.steps} steps, loss ${loss}.`);
    }
    if (result.model) renderModel(result.model);
  } catch (error) {
    showError(error);
  } finally {
    training = false;
    $('train-btn').disabled = false;
    $('train-stop-btn').hidden = true;
    setTitleProgress(null);
    updateGuidance();
  }
});

$('train-stop-btn').addEventListener('click', () => {
  $('train-stop-btn').disabled = true;
  $('train-stats').textContent = 'Stopping after this step…';
  call('stop', {}, [], 0).catch(() => {});
});

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
    await syncAllSources();
    await refreshStoryState();
  } catch (error) {
    showError(`that file didn't load: ${error.message}`);
    setModelStatus('absent', 'No model loaded.');
  }
  event.target.value = '';
});


// --- Start -------------------------------------------------------------

(async function start() {
  // Guidance first, before anything that can be slow. Reading saved
  // sources waits on IndexedDB, and until it answers the page would
  // otherwise sit there saying nothing at all — which is the state this
  // whole line of text exists to prevent.
  renderSources();
  updateGuidance();

  // Nothing is fetched here. The page loads, shows what you already
  // have, and waits. No model is downloaded, ever.
  await refreshSources();
  setModelStatus('absent', 'No model yet.');
  updateGuidance();
})();
