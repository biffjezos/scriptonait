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

/// Matches worker.js. A stored profile from an older benchmark measured
/// something the current one does not, so it is discarded rather than
/// trusted.
const BENCH_VERSION = 1;

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
  // Past an hour, minutes-and-seconds stops being readable: a planned
  // run's estimate is often hours, and "214m30s" is not an answer.
  if (seconds >= 3600) {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return `${h}h${String(m).padStart(2, '0')}m`;
  }
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

  // Training is GPU work and has no CPU path: with no device the button
  // is not something to press and find out.
  const canTrain = !model || model.usingGpu;
  $('train-btn').disabled = training || sources.length === 0 || (model && !model.usingGpu);
  $('generate-btn').disabled = generating || !model;

  // Say what a step will actually cover, since batch size and context
  // multiply and neither number means much alone.
  const typedBatch = Number($('train-batch').value);
  const batch = chosenBatchSize();
  const context = model ? model.contextLen : Number($('cfg-context').value) || 0;
  const where = typedBatch > 0
    ? ''
    : machineProfile && profileShapeMatches(machineProfile)
      ? ' (measured on this machine)'
      : ' — a fallback, not a measurement: press Train and this machine gets benchmarked first, ' +
        'or type a number here';
  $('batch-hint').textContent =
    `Batch size costs time, not memory: the sequences of a batch run one at a time and their ` +
    `gradients add up. ${batch}${where} x ${context} = ` +
    `${(batch * context).toLocaleString()} tokens per step.`;

  $('train-btn').textContent = model
    ? (model.pretrained ? 'Keep training on my writing' : 'Keep training this model')
    : 'Train a model on my writing';

  const explains = $('train-explains');
  explains.textContent = !canTrain
    ? 'Training needs WebGPU. This browser did not give the page a GPU.'
    : !model
      ? 'New model, from scratch, trained on your GPU.'
      : model.pretrained
        ? 'Nudges the loaded model toward your writing.'
        : 'Continues where it stopped.';

  if (training) {
    step.textContent = 'Training on your GPU. Stop any time — progress is kept.';
  } else if (generating) {
    step.textContent = 'Writing…';
  } else if (!model && sources.length === 0) {
    step.textContent = 'Step 1: add your writing.';
  } else if (!model && !enoughText) {
    step.textContent = `Only ${formatCount(words)} characters. Add more, then train.`;
  } else if (!model) {
    step.textContent = 'Step 2: train.';
  } else if (!canTrain) {
    step.textContent = 'No WebGPU here, so this model can write but not train.';
  } else if (model.step < 500) {
    step.textContent = 'Barely trained. Keep training, or try step 3.';
  } else {
    step.textContent = 'Ready. Step 3: type a prompt.';
  }

  renderMachineProfile();
}

/// Hand the model everything already in the list.
///
/// Called whenever a model appears, because sources can be added before
/// one exists — that's the normal order now — and the model starts empty.
async function syncAllSources() {
  for (const source of sources) {
    await syncSource(source);
  }
  await reportDuplicates();
  await refreshPlan();
}

/// Say when the same text is loaded twice.
///
/// A duplicate is trained on twice, which weights that script double and
/// flatters the held-out number for it. The page names them; removing
/// one is the user's decision, not the page's.
async function reportDuplicates() {
  if (!model) return;
  try {
    const { ids } = await call('duplicate-sources');
    if (!ids || ids.length === 0) return;
    const titles = ids
      .map((id) => (sources.find((s) => s.id === id) || {}).title || id)
      .slice(0, 3);
    showError(
      `${ids.length} source${ids.length === 1 ? ' is a copy' : 's are copies'} of another ` +
        `(${titles.join(', ')}${ids.length > 3 ? ', …' : ''}). Training on a script twice ` +
        'weights it double — remove the copies in step 1.',
    );
  } catch (error) {
    console.warn('[scriptonait] duplicate check failed:', error);
  }
}

/// The shape fields describe the model that exists, once one does.
///
/// They start at the defaults a new model would be built with, and until
/// this ran they kept showing those defaults next to a loaded model with
/// an entirely different shape — four layers on screen, eight in the
/// model. With a model loaded the shape is fixed, so the fields state it
/// and stop being editable.
function renderModelShape(info) {
  const fields = [
    ['cfg-layers', info && info.layers],
    ['cfg-hidden', info && info.hidden],
    ['cfg-heads', info && info.heads],
    ['cfg-kv-heads', info && info.kvHeads],
    ['cfg-context', info && info.contextLen],
    ['cfg-window', info && info.window],
  ];
  for (const [id, value] of fields) {
    const field = $(id);
    if (info) field.value = value;
    field.disabled = Boolean(info);
  }
  $('shape-hint').textContent = info
    ? "This model's shape. Fixed — training continues the model you have."
    : 'New model shape:';
}

function renderModel(info) {
  model = info;
  $('generate-btn').disabled = !info;
  $('train-btn').disabled = !info;
  renderModelShape(info);
  if (!info) return;

  const params = formatCount(info.params);
  // The device is stated, never chosen. It's a fact about the machine,
  // and the first question anyone asks about speed.
  // wgpu reports things like " (BrowserWebGpu, Other)" — already
  // parenthesised, sometimes with an empty name in front. Unwrap one
  // enclosing pair rather than stripping brackets blindly, which left
  // the parentheses unbalanced.
  const device = (info.device || '').trim().replace(/^\((.*)\)$/, '$1').trim();
  // Training only ever happens on the GPU, so a machine without one is
  // told that here rather than when it presses Train.
  const where = info.usingGpu
    ? `training and writing on your GPU${device ? ` (${device})` : ''}`
    : 'no WebGPU in this browser — it can write on the CPU, but not train';
  setModelStatus(
    'ready',
    info.step > 0
      ? `Your model: ${params} parameters, trained ${info.step.toLocaleString()} steps, ${where}.`
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
      <div><dt>Training steps</dt><dd>${info.step.toLocaleString()}</dd></div>
      <div><dt>Training and generating on</dt><dd>${escapeHtml(info.device || 'no GPU — cannot train')}</dd></div>
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

// The worker's log, mirrored into the page's console so both are in one
// place: which device was acquired, what each training step cost, and
// why a run refused to start.
// A worker that fails to parse or throws outside a handler used to be
// invisible: no reply, no error, every call timing out after a minute.
// It reports itself now, and the page says so.
worker.addEventListener('error', (event) => {
  const detail = (event && (event.message || String(event.error))) || 'unknown error';
  console.error('[scriptonait] worker failed to load or crashed:', detail, event);
  showError(`the background worker failed: ${detail}`);
});

onStream('worker-error', ({ message, stack }) => {
  console.error(`[scriptonait] worker error: ${message}`, stack || '');
  showError(`worker error: ${message}`);
});

onStream('log', ({ message, data }) => {
  if (data === null || data === undefined) {
    console.info(`[scriptonait] ${message}`);
  } else {
    console.info(`[scriptonait] ${message}`, data);
  }
});

onStream('gpu-status', ({ available, device, reason, report }) => {
  if (available) {
    gpuReport = report || null;
    loadMachineProfile().catch((error) => console.warn('[scriptonait]', error));
    console.info(`[scriptonait] device: ${device}`, report || {});
    if (report && report.isSoftware) {
      console.warn(
        '[scriptonait] this is a software renderer, not your GPU — training will be slow',
      );
    }
  } else {
    console.warn(
      `[scriptonait] no WebGPU (${reason || 'unavailable'}): generation runs on the CPU, ` +
        'and training cannot run at all',
    );
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
      minP: Number($('opt-min-p').value),
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
    // Measure what came out, to the console. Not on the page: the piece
    // is the point, and a person reading it does not need a score
    // stapled to it — but "37% of those words are not in your corpus"
    // is the answer to "why does this read like that".
    try {
      const quality = await call('evaluate', { text: result.text || '', loss: -1 });
      if (quality && quality.words > 0) {
        console.info(
          `[scriptonait] this piece: ${Math.round(quality.knownWordRate * 100)}% of its words ` +
            `are in your corpus, ${Math.round(quality.repeated4gramRate * 100)}% of its ` +
            `four-word runs are repeats, ${Math.round(quality.distinctWordRate * 100)}% ` +
            'of its words are distinct',
          quality,
        );
      }
    } catch (error) {
      console.warn('[scriptonait] could not measure the generated text', error);
    }
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

// A few hundred sources is a normal amount to load at once, and a few
// hundred rows is not a list anybody can use: it buries the rest of the
// page, and every render would rebuild all of it. The list scrolls
// (capped in CSS), rows above this many are not drawn at all, and the
// filter is how you reach the ones that aren't.
const MAX_SOURCE_ROWS = 50;
let sourceFilter = '';

/// Draw the list from memory. Synchronous on purpose: nothing it needs
/// can be slow, so nothing can stop it running.
function renderSources() {
  const list = $('sources-list');
  const toolbar = $('sources-toolbar');
  // The filter and Remove all only exist to make a long list usable, so
  // they stay out of the way of a short one.
  toolbar.hidden = sources.length <= MAX_SOURCE_ROWS && !sourceFilter;
  const needle = sourceFilter.toLowerCase();
  const matches = needle
    ? sources.filter((source) => (source.title || '').toLowerCase().includes(needle))
    : sources;

  if (sources.length === 0) {
    list.innerHTML = '<p class="empty-hint">Nothing added yet.</p>';
  } else if (matches.length === 0) {
    list.innerHTML = `<p class="empty-hint">No source matches "${escapeHtml(sourceFilter)}".</p>`;
  } else {
    const shown = matches.slice(0, MAX_SOURCE_ROWS);
    const hidden = matches.length - shown.length;
    list.innerHTML =
      shown
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
        .join('') +
      (hidden > 0
        ? `<p class="empty-hint">${hidden.toLocaleString()} more not shown — filter by name to reach them.</p>`
        : '');
  }
  updateSourceSummary(sources);
  updateGuidance();
}

// One listener on the container instead of one per row: with hundreds of
// sources, re-binding a button per row on every render is the expensive
// part of drawing the list.
$('sources-list').addEventListener('click', (event) => {
  const button = event.target.closest('.remove-source');
  if (button) removeSource(button.dataset.id);
});

$('sources-filter').addEventListener('input', (event) => {
  sourceFilter = event.target.value.trim();
  renderSources();
});

$('remove-all-btn').addEventListener('click', async () => {
  if (sources.length === 0) return;
  if (!confirm(`Remove all ${sources.length.toLocaleString()} sources? This can't be undone.`)) {
    return;
  }
  const removed = sources;
  sources = [];
  sourceFilter = '';
  $('sources-filter').value = '';
  renderSources();
  for (const source of removed) {
    await persist('deleting a source', () => db.deleteSource(source.id));
    try {
      renderModel(await call('remove-source', { id: source.id }));
    } catch (error) {
      /* no model loaded: it was only ever in the list */
    }
  }
  await refreshStoryState();
  await refreshPlan();
});

async function removeSource(id) {
  sources = sources.filter((source) => source.id !== id);
  renderSources();
  await persist('deleting a source', () => db.deleteSource(id));
  try {
    renderModel(await call('remove-source', { id }));
    await refreshStoryState();
    await refreshPlan();
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
  // Sources live in two places: this list, which is the browser's, and
  // the model's corpus, which is the wasm side's. Without a model the
  // second one does not exist, so a file added now is in the list and
  // nowhere else until a model is made — and any number computed from
  // the corpus is about a corpus that no longer matches this list.
  const seen = model
    ? ''
    : ' — no model yet, so none of this is in a corpus: make or open one and it is all handed over';
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
  // The corpus just changed, so every number the plan is built from
  // did too. Without this the plan keeps reporting the corpus it was
  // last computed against.
  await refreshPlan();
  if (failures.length) {
    showError(
      `${failures.length} of ${entries.length} couldn't be added: ${failures.slice(0, 3).join(', ')}` +
        (failures.length > 3 ? ', …' : ''),
    );
  }
  updateSourceSummary(sources, added ? `added ${added}` : '');
}

// Across a few hundred sources these run to thousands of entries, and a
// paragraph of comma-separated names tells nobody anything. Show the
// first handful and count the rest.
const MAX_NAMES_SHOWN = 25;

function nameList(names) {
  const shown = names.slice(0, MAX_NAMES_SHOWN);
  const hidden = names.length - shown.length;
  return escapeHtml(shown.join(', ')) + (hidden > 0 ? ` <span class="hint">+${hidden.toLocaleString()} more</span>` : '');
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
    if (state.characters.length) parts.push(`<strong>Characters:</strong> ${nameList(state.characters)}`);
    if (state.locations.length) parts.push(`<strong>Locations:</strong> ${nameList(state.locations)}`);
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
/// Held-out loss, with the index into `lossHistory` it was measured at,
/// so the two curves share a time axis despite different cadences.
const validationHistory = [];
/// Loss on a fixed set of *training* windows, drawn exactly as the
/// held-out set is drawn.
///
/// The blue per-step curve cannot be compared with the held-out one:
/// 40% of training windows start at a source's opening and no held-out
/// window ever does, so the two separate as soon as the model learns
/// what an opening looks like — a few hundred steps in, long before
/// anything could be memorized. This third curve is the fair
/// comparison, and the distance between it and the amber one is the
/// only gap worth reading.
const probeHistory = [];

onStream('train-progress', (progress) => {
  setProgress('train-progress-bar', progress.fractionDone);
  const on = model && model.device ? ` · on ${model.device}` : '';
  // Held-out loss is the one that says whether it is learning the
  // language or the sample, so it sits next to the training loss.
  const held =
    typeof progress.validationLoss === 'number'
      ? ` · held-out ${progress.validationLoss.toFixed(3)}`
      : '';
  $('train-stats').textContent =
    `step ${progress.step.toLocaleString()} · loss ${progress.smoothedLoss.toFixed(3)}${held} · ` +
    `${progress.tokensPerSecond.toFixed(0)} tokens/s · ${formatDuration(progress.elapsedSeconds)} elapsed${on}`;
  setTitleProgress('Fine-tuning', progress.fractionDone);
  lossHistory.push(progress.smoothedLoss);
  if (
    typeof progress.validationLoss === 'number' &&
    (validationHistory.length === 0 ||
      validationHistory[validationHistory.length - 1].loss !== progress.validationLoss)
  ) {
    validationHistory.push({ at: lossHistory.length - 1, loss: progress.validationLoss });
    if (typeof progress.trainingProbe === 'number' && progress.trainingProbe >= 0) {
      probeHistory.push({ at: lossHistory.length - 1, loss: progress.trainingProbe });
    }
  }
  drawLossChart();
});

// Samples from the model as it trains. Exactly one card, rewritten in
// place every time a sample arrives: `replaceChildren` runs on every
// event, so the box holds this card and nothing else no matter what was
// in it before. Never append - a stack of stale samples buries the only
// one worth reading, which is the current one.
// What the held-out curve is saying to do about the corpus. It appears
// when the numbers earn it and stays until the next run.
// A new best held-out loss: keep a copy of this model, because training
// continues past it and the last model of a run is usually not its best.
let bestSaveInFlight = false;
onStream('train-best', async ({ step, validationLoss }) => {
  if (bestSaveInFlight || !model) return;
  bestSaveInFlight = true;
  try {
    const { bytes } = await call('export-checkpoint');
    await db.putBestModel({ bytes, step, params: model.params, validationLoss });
    console.info(
      `[scriptonait] kept the best model so far (held-out ${validationLoss.toFixed(3)} ` +
        `at step ${step.toLocaleString()})`,
    );
    renderBestModel({ step, validationLoss });
  } catch (error) {
    console.warn('[scriptonait] could not keep the best model:', error);
  } finally {
    bestSaveInFlight = false;
  }
});

/// Show what the best kept model is, and offer to go back to it.
function renderBestModel(best) {
  const row = $('best-model');
  if (!best) {
    row.hidden = true;
    return;
  }
  $('best-model-text').textContent =
    `Best so far: held-out ${best.validationLoss.toFixed(3)} at step ${best.step.toLocaleString()}.`;
  row.hidden = false;
}

$('restore-best-btn').addEventListener('click', async () => {
  const best = await db.getBestModel();
  if (!best) return;
  if (!confirm(`Go back to the model from step ${best.step.toLocaleString()}? The current one is replaced.`)) {
    return;
  }
  setModelStatus('loading', 'Restoring the best model…');
  renderModel(await call('load-model', { bytes: best.bytes }, [], 0));
  await syncAllSources();
  await refreshStoryState();
  await saveModel();
  console.info('[scriptonait] restored the best model');
});

// --- The training plan -------------------------------------------------
//
// Loss and step count say what is happening; the plan says what it means
// and what to do. It is computed in the worker from the schedule's own
// numbers and the corpus's own numbers — see `buildPlan` there — and this
// only draws it.

/// The phase last drawn, so a change of phase can be announced once
/// rather than every time the plan is recomputed.
let lastPhaseKey = null;

function renderPlan(plan) {
  const box = $('train-plan');
  // No model means no corpus on the wasm side, so there is nothing to
  // compute a plan from — and the numbers on screen are from whatever
  // was there last. Stale numbers next to a list that has changed under
  // them is exactly how the page stops being believed: 66 sources and
  // 17.67M characters above, a token count from a 30-source corpus
  // below, and no way to tell which is current.
  if (!model || !plan || !plan.phase) {
    box.hidden = true;
    return;
  }
  $('plan-phase-title').textContent = plan.phase.title;
  $('plan-phase-detail').textContent = plan.phase.detail;

  const n = plan.numbers;
  // Two lines, because they answer two different questions and running
  // them together is how "0.3 tokens per parameter" ended up beside
  // "1.17M tokens seen" as though they were the same kind of fact.
  //
  // First line: how much training has happened. Second: what there is
  // to train on. Corpus size is given in characters as well as tokens,
  // because the token count changes when the vocabulary is relearned
  // and the character count does not - and a number that moves for
  // reasons the user did not cause is a number they stop believing.
  const progress = [`step ${n.step.toLocaleString()}`];
  // Step counts in two frames: the model's lifetime, and this run. The
  // schedule works in the second, so a run's progress has to be shown
  // in it — "step 4,977 of 500 planned" is what happens otherwise.
  if (n.plannedSteps > n.runStep) {
    progress.push(`${n.runStep.toLocaleString()} of ${n.plannedSteps.toLocaleString()} this run`);
  }
  progress.push(
    n.tokensSeen > 0
      ? `${formatCount(n.tokensSeen)} tokens trained on`
      : 'tokens trained on: not recorded for this model',
  );
  if (n.epochs >= 0.01) progress.push(`${n.epochs.toFixed(2)} passes over your text`);
  if (n.bitsPerByte > 0) progress.push(`${n.bitsPerByte.toFixed(2)} bits/byte`);
  if (n.quality && n.quality.words > 0) {
    progress.push(`${Math.round(n.quality.knownWordRate * 100)}% real words`);
  }
  if (n.etaSeconds !== null && n.etaSeconds > 0) {
    progress.push(`${formatDuration(n.etaSeconds)} left in this run`);
  }

  const corpus = [];
  if (n.corpusChars > 0) corpus.push(`${formatCount(n.corpusChars)} characters`);
  corpus.push(`${formatCount(n.trainingTokens)} tokens at this vocabulary`);
  corpus.push(`${formatCount(n.params)} parameters`);

  // The list and the corpus are two different things, and they can
  // disagree: a source added while no model existed is in the list and
  // not in the corpus. Saying so beats letting somebody compare the two
  // lines and conclude the page is making numbers up.
  const listedChars = sources.reduce((sum, s) => sum + (s.rawText || '').length, 0);
  if (n.corpusChars > 0 && Math.abs(listedChars - n.corpusChars) > n.corpusChars * 0.05) {
    corpus.push(
      `the list above holds ${formatCount(listedChars)} characters — press Train to hand the ` +
        'difference over',
    );
  }

  $('plan-numbers').textContent = `Trained: ${progress.join(' · ')}`;
  $('plan-corpus').textContent = `Corpus: ${corpus.join(' · ')}`;

  const list = $('plan-actions');
  list.replaceChildren();
  for (const action of plan.actions || []) {
    const li = document.createElement('li');
    li.textContent = action.text;
    if (action.urgency === 'high') li.className = 'high';
    list.append(li);
  }
  box.hidden = false;

  // A change of phase is worth a notification: the whole point of naming
  // the phases is that "the loss is barely moving" means opposite things
  // in two of them, and nobody watches a tab for twenty minutes to find
  // out which.
  if (plan.phase.key !== lastPhaseKey) {
    if (lastPhaseKey !== null) notify(`Training: ${plan.phase.title}`, plan.phase.detail);
    console.info(`[scriptonait] phase: ${plan.phase.title} — ${plan.phase.detail}`, n);
    lastPhaseKey = plan.phase.key;
  }
}

onStream('train-plan', (plan) => renderPlan(plan));

/// Ask for the plan outside a training run — after a model loads, or the
/// corpus changes. The answer is what to do next, and it should not take
/// a training run to see it.
async function refreshPlan() {
  if (!model || training) return;
  try {
    renderPlan(await call('training-plan', { batchSize: chosenBatchSize() }));
  } catch (error) {
    console.warn('[scriptonait] could not build the training plan', error);
  }
}

onStream('train-advice', ({ advice, step }) => {
  const box = $('train-advice');
  box.textContent = `At step ${step.toLocaleString()}: ${advice}`;
  box.hidden = false;
  console.info(`[scriptonait] advice: ${advice}`);
  notify('Your model has stopped improving', advice);
});

onStream('train-sample', ({ step, loss, text, quality }) => {
  const box = $('train-samples');
  let block = box.firstElementChild;
  if (!block || box.children.length !== 1) {
    block = document.createElement('div');
    block.className = 'train-sample';
    const head = document.createElement('div');
    head.className = 'train-sample-head';
    block.append(head, document.createElement('pre'));
  }
  // The header carries the measurement, because the sample is where a
  // person decides whether this is working, and "is it words" is not
  // something you can eyeball reliably at 40 words a time.
  const head = [`step ${step.toLocaleString()}`];
  if (typeof loss === 'number') head.push(`loss ${loss.toFixed(3)}`);
  if (quality && quality.words > 0) {
    head.push(`${Math.round(quality.knownWordRate * 100)}% real words`);
    if (quality.repeated4gramRate > 0.05) {
      head.push(`${Math.round(quality.repeated4gramRate * 100)}% repeated runs`);
    }
  }
  block.firstElementChild.textContent = head.join(' \u00b7 ');
  block.lastElementChild.textContent = text;
  box.replaceChildren(block);
});


// --- The machine profile -----------------------------------------------
//
// Nothing about how fast a step runs can be worked out from here. How
// much work fits in one command buffer before the driver's watchdog
// takes the device away, and how many sequences a batch can hold before
// a step stops being interruptible, are properties of the GPU, its
// driver and the model's shape together.
//
// So the page measures them, once, on the machine it is actually running
// on, and stores the answer. Later visits load it and set the settings
// from it. Nothing here is a constant chosen for anybody's hardware.

/// The device report from the worker, which names the adapter the
/// profile is keyed by.
let gpuReport = null;
/// The stored profile for this adapter, once it has been read back.
let machineProfile = null;
let benchmarking = false;

/// True when a stored profile was measured against a model of a
/// different shape. Its command-buffer size still applies — that is
/// about the driver — but its batch size does not, because the batch
/// ceiling is a step-duration ceiling and the step's duration is the
/// model's.
function profileShapeMatches(profile) {
  if (!profile || !profile.shape || !model) return false;
  const s = profile.shape;
  return (
    s.layers === model.layers &&
    s.hidden === model.hidden &&
    s.heads === model.heads &&
    s.kvHeads === model.kvHeads &&
    s.contextLen === model.contextLen &&
    s.window === model.window &&
    s.vocabSize === model.vocabSize
  );
}

function renderMachineProfile() {
  const text = $('machine-profile-text');
  const forget = $('bench-forget-btn');
  if (!text) return;
  if (benchmarking) {
    text.textContent = 'Measuring this machine…';
    forget.hidden = true;
    return;
  }
  if (!machineProfile) {
    text.textContent = gpuReport
      ? 'Machine profile: not measured yet. The first training run measures it once.'
      : 'Machine profile: needs a GPU.';
    forget.hidden = true;
    return;
  }
  const p = machineProfile;
  const stale = model && !profileShapeMatches(p);
  text.textContent =
    `Machine profile: ${p.adapter} — ${p.dispatchesPerSubmit} dispatches per command buffer, ` +
    `batch ${p.batchSize}, ${Math.round(p.msPerStep)} ms per step ` +
    `(${Math.round(p.tokensPerSecond)} tokens/s)` +
    (stale ? '. Measured on a differently shaped model — measure again for its batch size.' : '.');
  forget.hidden = false;
}

/// Read the stored profile for whatever adapter the browser handed over,
/// and put its command-buffer size back into the worker.
async function loadMachineProfile() {
  if (!gpuReport || !gpuReport.available) return null;
  let stored = null;
  try {
    stored = await db.getMachineProfile(gpuReport);
  } catch (error) {
    console.warn('[scriptonait] could not read the machine profile', error);
    return null;
  }
  // A profile from an older benchmark measured something else; re-measure
  // rather than act on it.
  if (stored && stored.version !== BENCH_VERSION) {
    console.info('[scriptonait] the stored machine profile is from an older benchmark — ignoring it');
    stored = null;
  }
  machineProfile = stored;
  if (stored) {
    await call('apply-machine-profile', { dispatchesPerSubmit: stored.dispatchesPerSubmit });
    console.info('[scriptonait] machine profile loaded', stored);
  }
  renderMachineProfile();
  updateGuidance();
  return stored;
}

/// Run the benchmark and store what it found. Needs a model and enough
/// text, since it times the real step on the real shapes.
async function runBenchmark() {
  // Not gated on `training`: the first thing a training run does is
  // measure the machine it is about to train on, and by then the page
  // already considers itself training. The worker refuses a benchmark
  // that would land in the middle of an actual run.
  if (benchmarking) return machineProfile;
  benchmarking = true;
  $('bench-btn').disabled = true;
  renderMachineProfile();
  try {
    const result = await call('benchmark', {}, [], 0);
    if (result.error) {
      console.warn(`[scriptonait] benchmark: ${result.error}`);
      return machineProfile;
    }
    machineProfile = await db.putMachineProfile(result.profile);
    console.info('[scriptonait] machine profile stored', machineProfile);
    return machineProfile;
  } finally {
    benchmarking = false;
    $('bench-btn').disabled = false;
    renderMachineProfile();
    updateGuidance();
  }
}

/// Batch size to train at: the box if it was filled in, otherwise what
/// the benchmark measured, otherwise one sequence — which is the only
/// size that is safe on an unmeasured machine.
function chosenBatchSize() {
  const typed = Number($('train-batch').value);
  if (typed > 0) return typed;
  if (machineProfile && profileShapeMatches(machineProfile)) return machineProfile.batchSize;
  // The fallback, and it is a bad one to take silently: batch 1 is a
  // quarter of the throughput this machine can do, and a run left on it
  // for four thousand steps has done a quarter of the training its step
  // count suggests. Relearning the vocabulary rebuilds the model, which
  // makes the stored profile's shape stop matching and lands here, so
  // this is a real path and not a theoretical one.
  return 1;
}

/// True when "auto" is about to fall back rather than use a measurement.
function batchSizeIsGuessed() {
  return (
    Number($('train-batch').value) <= 0 &&
    !(machineProfile && profileShapeMatches(machineProfile))
  );
}

/// Effort, when it is left on Auto.
///
/// Effort is the share of the time the worker spends training rather
/// than sitting idle, and it exists so a run does not make the rest of
/// the machine unusable. Auto is full speed, and that is a measurement
/// rather than an opinion: the benchmark refuses any batch whose step
/// runs past the interruptible ceiling, so a step already ends often
/// enough for the browser to get the device back, and the worker spends
/// almost all of a step awaiting the GPU rather than holding a core.
/// Insert an idle share only if you want the machine back — the two
/// slower settings are still there, and Auto never picks them for you.
function chosenEffort() {
  const value = $('train-effort').value;
  return value === 'auto' ? 1 : Number(value);
}

onStream('bench-progress', ({ stage, dispatchesPerSubmit }) => {
  if (!benchmarking || stage !== 'chunk') return;
  $('machine-profile-text').textContent =
    `Measuring this machine — ${dispatchesPerSubmit} dispatches per command buffer is ` +
    'fastest so far; now finding the batch size…';
});

$('bench-btn').addEventListener('click', () => {
  clearError();
  runBenchmark().catch(showError);
});

$('bench-forget-btn').addEventListener('click', async () => {
  if (!gpuReport) return;
  await db.deleteMachineProfile(gpuReport);
  machineProfile = null;
  renderMachineProfile();
  updateGuidance();
});

$('train-btn').addEventListener('click', async () => {
  if (training) return;
  clearError();
  training = true;
  lastPhaseKey = null;
  lossHistory.length = 0;
  validationHistory.length = 0;
  probeHistory.length = 0;
  $('train-btn').disabled = true;
  $('train-stop-btn').hidden = false;
  $('train-stop-btn').disabled = false;
  $('train-status').hidden = false;
  $('loss-chart').hidden = false;
  $('train-stats').textContent = 'Starting…';
  $('train-samples').replaceChildren();
  $('train-advice').hidden = true;

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
      // Learn the vocabulary from the text now, while the model is still
      // untrained: one token per byte costs about four times the tokens,
      // and therefore four times the training time, for the same text.
      // The embedding table is one row per token, so this has to happen
      // before training starts, and it rebuilds the model.
      $('train-stats').textContent = 'Learning a vocabulary from your text…';
      const learned = await call('learn-vocabulary', { maxVocabSize: 8192 }, [], 0);
      if (learned && learned.model) renderModel(learned.model);
    }

    // Measure the machine once, before the first run on it. The
    // settings that follow are read off that measurement, so it has to
    // happen first — and it only ever happens once per adapter and
    // model shape, because the answer is stored.
    if (!machineProfile || !profileShapeMatches(machineProfile)) {
      $('train-stats').textContent =
        'Measuring this machine — timing a few steps to pick the settings…';
      await runBenchmark().catch((error) => {
        // A benchmark that fails is not a reason not to train: the
        // fallbacks are one sequence per batch and the default command
        // buffer, which are the safe values.
        console.warn('[scriptonait] the machine benchmark failed', error);
      });
    }

    const fromScratch = model && !model.pretrained;
    const chosenRate = Number($('train-lr').value);
    const batchSize = chosenBatchSize();
    const result = await call('train', {
      batchSize,
      // 0 means "pick one": a new model needs a rate large enough to
      // learn a language from nothing; a working one needs a small
      // enough rate not to forget it.
      //
      // 6e-4 is what nanoGPT uses for a 768-wide GPT-2, and a narrower
      // model tolerates more rather than less, so it is a conservative
      // choice at the widths this page builds — and twice the 3e-4 that
      // was here, which was simply timid. With warm-up, gradient-norm
      // clipping at 1.0 and the plateau cut watching held-out loss,
      // there are three separate things that catch a rate that turns
      // out to be too high; there is nothing that catches one that is
      // too low except hours of your time.
      learningRate: chosenRate > 0 ? chosenRate : (fromScratch ? 6e-4 : 5e-5),
      maxSteps: Number($('train-steps').value),
      effort: chosenEffort(),
      // 0 turns sampling off; anything else is a step interval.
      sampleEvery: $('sample-toggle').checked ? Number($('sample-every').value) : 0,
      samplePrompt: $('sample-prompt').value.trim() || 'Write a 40 word scene.',
      sampleWords: 40,
    }, [], 0);

    if (result.stopReason === 'no-gpu') {
      showError(
        'Training runs on your GPU, and this browser did not give the page one. ' +
          'Try a browser with WebGPU (Chrome or Edge 113+, Safari 18+), or enable it in ' +
          "your browser's flags.",
      );
    } else if (result.stopReason === 'no-data') {
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
    await saveModel();
    training = false;
    await refreshPlan();
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

$('train-batch').addEventListener('input', updateGuidance);

$('train-stop-btn').addEventListener('click', () => {
  $('train-stop-btn').disabled = true;
  $('train-stats').textContent = 'Stopping after this step…';
  call('stop', {}, [], 0).catch(() => {});
});

/// Both curves on one pair of axes: training loss, and the held-out loss
/// measured every 25 steps.
///
/// They are drawn against the same scale on purpose. Training loss alone
/// always falls; what tells you whether the model is learning the
/// language rather than memorizing your text is the two curves drifting
/// apart, and that is only visible if they share an axis.
function drawLossChart() {
  const canvas = $('loss-chart');
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (lossHistory.length < 2) return;

  const points = lossHistory
    .concat(validationHistory.map((p) => p.loss))
    .concat(probeHistory.map((p) => p.loss));
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const yFor = (loss) => height - ((loss - min) / span) * (height - 12) - 6;

  ctx.strokeStyle = '#7aa2f7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((loss, i) => {
    const x = (i / (lossHistory.length - 1)) * width;
    const y = yFor(loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  /// Both fixed-set curves, positioned by where in the run they were
  /// measured so they line up in time rather than by index.
  const drawMeasured = (series, colour, dashed) => {
    if (series.length < 2) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.beginPath();
    series.forEach((point, i) => {
      const x = (point.at / Math.max(lossHistory.length - 1, 1)) * width;
      const y = yFor(point.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };
  // Dashed, and the same blue as the per-step curve: it is training
  // loss, measured the way held-out loss is measured. The gap that
  // matters is between this and the amber one.
  drawMeasured(probeHistory, '#7aa2f7', true);
  drawMeasured(validationHistory, '#e0af68', false);

  ctx.fillStyle = '#8891a8';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(max.toFixed(3), 4, 12);
  ctx.fillText(min.toFixed(3), 4, height - 4);
  ctx.fillStyle = '#7aa2f7';
  ctx.fillText('training', width - 168, 12);
  ctx.fillText('· same windows', width - 122, 12);
  ctx.fillStyle = '#e0af68';
  ctx.fillText('held-out', width - 52, 12);
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


// Profiling from the console: `scriptonait.profile()` runs one step per
// command-buffer size and logs where the milliseconds go.
window.scriptonait = {
  /// Times one step per phase at four command-buffer sizes and logs the
  /// result. Returns the rows too, but the log is the point - and it
  /// catches its own failure, so a missing model prints one line instead
  /// of an unhandled rejection.
  /// Times each kernel at this model's shapes and logs a table sorted by
  /// how much of a step it accounts for.
  kernels: (reps = 20) =>
    call('profile-kernels', { reps }, [], 0).catch((error) => {
      console.error(`[scriptonait] kernels: ${(error && error.message) || error}`);
      return null;
    }),

  profile: (batchSize = 2) =>
    call('profile', { batchSize }, [], 0).catch((error) => {
      console.error(`[scriptonait] profile: ${(error && error.message) || error}`);
      return null;
    }),

  /// Re-measure this machine and store the result, whatever is stored
  /// now. The first training run does this by itself; this is for after
  /// a driver update, or to see the sweep again.
  benchmark: () =>
    runBenchmark().catch((error) => {
      console.error(`[scriptonait] benchmark: ${(error && error.message) || error}`);
      return null;
    }),

  /// What the machine measured, as stored.
  machine: () => machineProfile,

  /// Measure any text against your corpus: what fraction of its words
  /// your sources contain, how much of it repeats itself, and — with a
  /// loss — how many bits per byte that is.
  evaluate: (text, loss = -1) =>
    call('evaluate', { text, loss }).catch((error) => {
      console.error(`[scriptonait] evaluate: ${(error && error.message) || error}`);
      return null;
    }),
};

/// Write the trained model to IndexedDB.
///
/// A run is hours of someone's GPU and it lived in the tab only: a
/// reload threw it away without asking. Saved after every training run,
/// so the most that can be lost is the run you are in.
async function saveModel() {
  if (!model) return;
  try {
    const started = performance.now();
    const { bytes } = await call('export-checkpoint');
    const optimizer = await call('export-optimizer').then((r) => r.bytes).catch(() => null);
    await db.putModel({ bytes, step: model.step, params: model.params, optimizer });
    console.info(
      `[scriptonait] saved the model (${formatCount(bytes.byteLength)} bytes` +
        `${optimizer ? ` + ${formatCount(optimizer.byteLength)} of optimizer state` : ''}, ` +
        `step ${model.step.toLocaleString()}) in ${(performance.now() - started).toFixed(0)} ms`,
    );
  } catch (error) {
    // Storage can be full or denied; that must not cost you the run
    // you just did, so it is a warning rather than an error.
    console.warn('[scriptonait] could not save the model:', error);
    showError(
      `the model could not be saved (${(error && error.message) || error}). ` +
        'It is still loaded — use "Save this model to a file" to keep it.',
    );
  }
}

/// Load the model saved by an earlier visit, if there is one.
async function restoreModel() {
  let stored = null;
  try {
    stored = await db.getModel();
  } catch (error) {
    console.warn('[scriptonait] could not read the saved model:', error);
    return;
  }
  if (!stored || !stored.bytes) return;
  setModelStatus('loading', 'Loading your saved model…');
  try {
    renderModel(await call('load-model', { bytes: stored.bytes }, [], 0));
    await syncAllSources();
    await refreshStoryState();
    if (stored.optimizer) {
      // Momentum, so training continues where it left off instead of
      // restarting Adam from nothing.
      try {
        await call('import-optimizer', { bytes: stored.optimizer }, [], 0);
        console.info('[scriptonait] restored the optimizer state with the model');
      } catch (error) {
        console.warn('[scriptonait] optimizer state did not fit this model:', error);
      }
    }
  } catch (error) {
    console.warn('[scriptonait] the saved model would not load:', error);
    setModelStatus('absent', 'No model yet.');
  }
}

// --- Start -------------------------------------------------------------

(async function start() {
  // Guidance first, before anything that can be slow. Reading saved
  // sources waits on IndexedDB, and until it answers the page would
  // otherwise sit there saying nothing at all — which is the state this
  // whole line of text exists to prevent.
  renderSources();
  updateGuidance();

  // Nothing is fetched from the network here. The page loads, shows what
  // you already have — including the model your last visit trained — and
  // waits. No model is downloaded, ever.
  await refreshSources();
  setModelStatus('absent', 'No model yet.');
  updateGuidance();
  await restoreModel();
  try {
    const best = await db.getBestModel();
    if (best) renderBestModel(best);
  } catch (error) {
    console.warn('[scriptonait] could not read the best model:', error);
  }
  updateGuidance();
  // What to do next, before anything has been trained: with a model and
  // a corpus already restored from the last visit, the plan is readable
  // immediately and is the most useful thing on the page.
  await refreshPlan();
})();
