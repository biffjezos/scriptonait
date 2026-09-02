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
import * as project from './project.js';
import { LocalWasmBackend } from './backend/local-wasm-backend.js';
import { RemoteBackend } from './backend/remote-backend.js';

// --- Compute backend -----------------------------------------------------
// `app.js` never touches the worker or a remote server directly — only a
// backend's call()/onStream(), the interface every backend implements
// (see backend/backend.js). `showError` isn't defined yet at this point
// in the file, but the arrow functions below only resolve the name when
// a backend actually reports a fatal error, well after the whole module
// (including `showError`'s own declaration) has run.
//
// Corpus, model state, checkpoint I/O and inference always run on this
// browser's own WASM+WebGPU, whichever way Training is set — the model
// stays real here even when Training is Remote, since the client is
// always the durable owner (see crates/llm-server's own design note).
const localBackend = new LocalWasmBackend({ onFatalError: (error) => showError(error) });

/// Whichever backend Training is currently pointed at — `localBackend`
/// (device 'gpu'/'cpu') or a `RemoteBackend` (device 'remote'), set by
/// the Settings-tab Training select. Inference has no remote option (see
/// llm-server's own non-goals), so nothing else ever needs its own
/// backend variable.
let trainingBackend = localBackend;

/// Every `onStream` registration so far, replayed onto a new
/// `trainingBackend` when Settings switches it — every call site in this
/// file registers once, at module load, so this list only ever grows.
const streamRegistrations = [];

function call(type, payload = {}, transfer = [], timeoutMs) {
  return localBackend.call(type, payload, transfer, timeoutMs);
}

/// Training-only message types ('train', 'stop', 'update-training-settings',
/// 'reset-schedule') go to whichever backend Training is set to, instead
/// of always the local one — the one place local and remote genuinely
/// diverge.
function trainCall(type, payload = {}, transfer = [], timeoutMs) {
  return trainingBackend.call(type, payload, transfer, timeoutMs);
}

function onStream(type, handler) {
  streamRegistrations.push({ type, handler });
  localBackend.onStream(type, handler);
  if (trainingBackend !== localBackend) trainingBackend.onStream(type, handler);
}

/// Called whenever Settings swaps in a new Training backend — replays
/// every handler this file has ever registered onto it, so a stream
/// event fired by whichever backend is now training still reaches the
/// same code the local worker's events already do.
function replayStreamRegistrations(target) {
  for (const { type, handler } of streamRegistrations) target.onStream(type, handler);
}

/// Converts a checkpoint's raw bytes to base64 for the one place this
/// page ever needs it — handing a checkpoint to a remote server as JSON.
/// Chunked because `btoa(String.fromCharCode(...bytes))` blows the call
/// stack on a checkpoint of any real size.
function bytesToBase64(bytes) {
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/// The corpus snapshot a remote training session starts from — see the
/// "snapshot at start, edits wait" design: this is read once, when Train
/// is pressed, not kept in sync afterward.
function remoteSourcesPayload() {
  return sources
    .filter((s) => s.id && typeof s.rawText === 'string' && s.rawText.length > 0)
    .map((s) => ({ id: s.id, text: s.rawText, isHtml: s.kind === 'url' }));
}

// Mirrors worker.js's own constants of the same name and values —
// duplicated, not imported, because the main thread and the training
// worker are separate execution contexts with no shared module scope.
// Needed here only for the one case worker.js's own live Auto-mode loop
// can't reach: a remote run is a single job started once, not a series
// of local slices this page keeps adjusting, so it needs one concrete
// rate decided up front rather than deferring to a loop that runs
// somewhere else entirely (`crates/llm-server`, which has no equivalent
// of it). Keep both copies in sync if either changes.
const REMOTE_TOKENS_PER_PARAM = 20;
const REMOTE_AUTO_LR_FROM_SCRATCH = 6e-4;
const REMOTE_AUTO_LR_TRAINED = 5e-5;

/// Exports the local model's current checkpoint and starts a remote
/// training run from it — always the checkpoint path, never a bare
/// config, since by the time this is called `readTrainingSettings()`'s
/// own caller has already ensured a local model exists (Train creates
/// one first if there wasn't one), carrying the already-learned
/// vocabulary the remote session has to match.
async function startRemoteTraining(settings) {
  const { bytes } = await call('export-checkpoint');
  const checkpointBase64 = bytesToBase64(new Uint8Array(bytes));
  // Auto mode's rate isn't in `settings` (see readTrainingSettings) — a
  // remote job needs the one number decided now, from this model's
  // current token count, since nothing on the remote end will revisit
  // it the way the local worker's own loop does.
  let peakLearningRate = settings.peakLearningRate;
  if (settings.autoLearningRate) {
    const { numbers } = await call('training-plan', { batchSize: settings.batchSize });
    peakLearningRate = numbers.tokensSeen < numbers.params * REMOTE_TOKENS_PER_PARAM
      ? REMOTE_AUTO_LR_FROM_SCRATCH
      : REMOTE_AUTO_LR_TRAINED;
  }
  return trainCall('train', {
    checkpointBase64,
    sources: remoteSourcesPayload(),
    batchSize: settings.batchSize,
    peakLearningRate,
    maxSteps: settings.maxSteps,
    autosaveFrequencySteps: settings.autosaveFrequencySteps,
  });
}

/// Matches worker.js. A stored profile from an older benchmark measured
/// something the current one does not, so it is discarded rather than
/// trusted.
const BENCH_VERSION = 1;

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

// --- Setting help tooltips ------------------------------------------------
//
// Every value field in Settings gets a small [?] next to its label: hover
// for a one-line rationale, click through to the docs section that
// explains it in full — and, for anything that came from a specific
// paper, the paper itself. The rationale lives here, once per field, so
// the docs prose and this tooltip can't drift into contradicting each
// other by drifting independently.
const DOCS = 'https://github.com/biffjezos/scriptonait/blob/dev/docs';
const HELP_MODEL_ARCH = `${DOCS}/model.md#architecture`;
const HELP_MODEL_SHARING = `${DOCS}/model.md#layer-sharing`;
const HELP_GEN_SAMPLING = `${DOCS}/generation.md#sampling-settings-settings-tab-inference`;
const HELP_GEN_LENGTH = `${DOCS}/generation.md#length-control`;
const HELP_TRAIN_OPTIMIZER = `${DOCS}/training.md#optimizer`;
const HELP_TRAIN_MODE = `${DOCS}/training.md#training-mode-settings-tab`;
const HELP_TRAIN_BENCH = `${DOCS}/training.md#machine-benchmark-settings-tab`;
const HELP_TRAIN_PLAN = `${DOCS}/training.md#training-plan-settings-tab-live-during-a-run`;
const HELP_TRAIN_REMOTE = `${DOCS}/training.md#remote-training-settings-tab-training-location`;
const HELP_SCHEDULER = `${DOCS}/scheduler.md`;
const HELP_PROJECT_AUTOSAVE = `${DOCS}/project.md#auto-save-settings-tab`;

const SETTING_HELP = {
  // Model Shape (Settings tab → Model Shape) — docs/model.md
  'cfg-layers': { url: HELP_MODEL_ARCH, text: 'Transformer depth. Vaswani et al., Attention Is All You Need, 2017 (arXiv:1706.03762).' },
  'cfg-layer-sharing': { url: HELP_MODEL_SHARING, text: 'Off, Uniform groups (ALBERT-style), or Recurrent core (variable loop count) — trading parameters for repetition.' },
  'cfg-unique-layers': { url: HELP_MODEL_SHARING, text: 'Distinct weight sets when sharing is Uniform groups. Lan et al., ALBERT, 2019 (arXiv:1909.11942).' },
  'cfg-prelude-layers': { url: HELP_MODEL_SHARING, text: 'Non-shared layers before the recurrent core. Geiping et al., 2025 (arXiv:2502.05171).' },
  'cfg-coda-layers': { url: HELP_MODEL_SHARING, text: 'Non-shared layers after the recurrent core. Geiping et al., 2025 (arXiv:2502.05171).' },
  'cfg-core-loop-min': { url: HELP_MODEL_SHARING, text: 'Fewest times the shared core repeats in training. Geiping et al., 2025 (arXiv:2502.05171).' },
  'cfg-core-loop-max': { url: HELP_MODEL_SHARING, text: "Most times the shared core repeats; also this model's depth. Geiping et al., 2025 (arXiv:2502.05171)." },
  'cfg-hidden': { url: HELP_MODEL_ARCH, text: 'Residual stream width. Vaswani et al., Attention Is All You Need, 2017 (arXiv:1706.03762).' },
  'cfg-heads': { url: HELP_MODEL_ARCH, text: 'Attention heads. Vaswani et al., Attention Is All You Need, 2017 (arXiv:1706.03762).' },
  'cfg-kv-heads': { url: HELP_MODEL_ARCH, text: 'Key/value heads (GQA); must divide Heads. Ainslie et al., GQA, 2023 (arXiv:2305.13245).' },
  'cfg-context': { url: HELP_MODEL_ARCH, text: "Max sequence length — also RoPE's trained position range. Su et al., RoFormer, 2021 (arXiv:2104.09864)." },
  'cfg-window': { url: HELP_MODEL_ARCH, text: 'Sliding-window attention size, ≤ Context. Jiang et al., Mistral 7B, 2023 (arXiv:2310.06825).' },

  // Manual Settings (Training) — docs/training.md
  'train-effort': { url: HELP_TRAIN_MODE, text: 'Fraction of wall-clock time a step may occupy the GPU before yielding it back.' },
  'train-batch': { url: HELP_TRAIN_MODE, text: 'Sequences per training step; 0 defers to the machine benchmark.' },
  'train-lr': { url: HELP_TRAIN_OPTIMIZER, text: "Overrides the scheduler's peak rate entirely. AdamW: Loshchilov & Hutter, 2019 (arXiv:1711.05101)." },

  // Scheduler — docs/scheduler.md
  'warmup-strategy': { url: HELP_SCHEDULER, text: "How many steps ramp from 0 to peak rate; Fixed-length is sized from AdamW's own β₂ time constant." },
  'stable-phase': { url: HELP_SCHEDULER, text: 'Whether the rate can be cut mid-run on a detected plateau.' },
  'cooldown-shape': { url: HELP_SCHEDULER, text: 'Deferred = WSD (Hu et al., MiniCPM, 2024, arXiv:2404.06395); Immediate = Cosine (Loshchilov & Hutter, SGDR, 2017, arXiv:1608.03983).' },
  'decay-start': { url: HELP_SCHEDULER, text: 'When a Deferred cool-down starts decaying; Adaptive pins it to a detected plateau instead of a fixed fraction.' },
  'plan-length': { url: HELP_SCHEDULER, text: 'Adaptive stretches the plan while the run is still genuinely improving.' },

  // Training Plan — docs/training.md
  'train-steps': { url: HELP_TRAIN_PLAN, text: "Total steps this run's schedule is shaped against." },
  'metrics-every': { url: HELP_TRAIN_PLAN, text: 'Steps between held-out loss measurements.' },
  'opening-rate': { url: HELP_TRAIN_PLAN, text: "% of training windows drawn from a source's opening rather than a random offset." },
  'show-training-window': { url: HELP_TRAIN_PLAN, text: "Whether the current training batch's text is displayed live." },
  'training-window-chars': { url: HELP_TRAIN_PLAN, text: 'Truncation length for the displayed training window.' },
  'sample-every': { url: HELP_TRAIN_PLAN, text: 'Steps between generating a training sample.' },

  // Auto-save — docs/project.md
  'autosave-enabled': { url: HELP_PROJECT_AUTOSAVE, text: 'Periodically writes the whole project — checkpoint, corpus, history, and settings.' },
  'autosave-frequency': { url: HELP_PROJECT_AUTOSAVE, text: 'Steps between auto-saves.' },
  'autosave-mode': { url: HELP_PROJECT_AUTOSAVE, text: 'Overwrite keeps one copy; Add writes a new file per save.' },
  'autosave-filename': { url: HELP_PROJECT_AUTOSAVE, text: 'Connects a real file (or, in Add mode, a folder) auto-save writes into.' },

  // Training backend — docs/training.md
  'training-device': { url: HELP_TRAIN_REMOTE, text: 'Local trains in this browser tab; Remote sends the job to llm-server on another GPU machine.' },
  'remote-server-url': { url: HELP_TRAIN_REMOTE, text: 'Address of the llm-server instance to train on.' },
  'remote-server-token': { url: HELP_TRAIN_REMOTE, text: 'Bearer token llm-server was started with.' },

  // Inference — docs/generation.md
  'inference-device': { url: HELP_GEN_SAMPLING, text: 'GPU (default) or CPU for this generation.' },
  'opt-temperature': { url: HELP_GEN_SAMPLING, text: 'Logit scaling before sampling; 0 forces greedy decoding (always the top token).' },
  'opt-top-k': { url: HELP_GEN_SAMPLING, text: 'Keep only the k most likely tokens. Fan et al., Hierarchical Neural Story Generation, 2018 (arXiv:1805.04833).' },
  'opt-top-p': { url: HELP_GEN_SAMPLING, text: 'Nucleus sampling — smallest token set whose probability reaches p. Holtzman et al., 2020 (arXiv:1904.09751).' },
  'opt-min-p': { url: HELP_GEN_SAMPLING, text: "Keeps tokens scaled to the leading token's own probability. Nguyen et al., 2024 (arXiv:2407.01082)." },
  'opt-repetition': { url: HELP_GEN_SAMPLING, text: 'Pushes down the logits of recently-seen tokens. Keskar et al., CTRL, 2019 (arXiv:1909.05858).' },
  'opt-seed': { url: HELP_GEN_SAMPLING, text: 'RNG seed for sampling.' },
  'opt-core-loops': { url: HELP_MODEL_SHARING, text: 'Decoding depth when Layer sharing is Recurrent core — no retraining needed. Geiping et al., 2025 (arXiv:2502.05171).' },
  'opt-length-mode': { url: HELP_GEN_LENGTH, text: 'Continuous runs to the parsed/estimated target; Limit sets a hard token ceiling.' },
  'opt-max-tokens': { url: HELP_GEN_LENGTH, text: 'Hard token ceiling used when Length is set to Limit to.' },

  // Machine Benchmark — docs/training.md
  'benchmark-enabled': { url: HELP_TRAIN_BENCH, text: "Re-measures this machine's GPU throughput automatically before training." },

  // Mode toggles
  'train-mode': { url: HELP_TRAIN_MODE, text: "Auto reads batch size and effort from this machine's own benchmark; Manual exposes them directly." },
  'scheduler-mode': { url: HELP_SCHEDULER, text: "Auto uses this app's chosen defaults for all five schedule axes; Manual exposes them." },
};

/// Insert one `[?]` after each mapped field's label text — before any
/// element nested inside the label (some labels wrap their own input),
/// after any of the label's own leading text otherwise.
function attachSettingHelp() {
  for (const [id, help] of Object.entries(SETTING_HELP)) {
    const label = document.querySelector(`label[for="${id}"]`);
    if (!label) continue;
    const sup = document.createElement('sup');
    sup.className = 'setting-help';
    const link = document.createElement('a');
    link.href = help.url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.title = help.text;
    link.textContent = '[?]';
    sup.appendChild(link);
    label.insertBefore(sup, label.firstElementChild);
  }
}
attachSettingHelp();

// --- Notifications -------------------------------------------------------
//
// One bar, at the top of the page, for every transient notice: errors,
// confirmations, training events. Persistent data that belongs to a
// specific tab (the plan, the machine profile)
// stays inline in that tab instead of passing through here — this bar is
// for things that happened once and should be acknowledged and
// forgotten.

let noticeTimeout = null;

/// `level` is 'error' | 'info' | 'success'. Errors stay until replaced or
/// dismissed; info/success clear themselves after a few seconds.
const NOTICE_ALERT_CLASS = { error: 'alert-danger', info: 'alert-info', success: 'alert-success' };
function notice(text, level = 'error') {
  const bar = $('notification-bar');
  bar.textContent = text;
  bar.className = `notification-bar alert ${NOTICE_ALERT_CLASS[level] || NOTICE_ALERT_CLASS.error}`;
  bar.hidden = false;
  if (noticeTimeout) clearTimeout(noticeTimeout);
  if (level !== 'error') {
    noticeTimeout = setTimeout(() => { bar.hidden = true; }, 5000);
  }
}

/// Show a failure, with enough of the stack to act on.
///
/// An error message on its own is often useless — "Cannot read
/// properties of undefined (reading 'length')" says nothing about which
/// call produced it. The first frame does, so it goes on screen, and the
/// whole thing goes to the console.
function showError(error) {
  const message = typeof error === 'string' ? error : error.message;
  const frame = typeof error === 'object' && error && error.stack
    ? String(error.stack).split('\n').slice(1).find((line) => line.trim()) || ''
    : '';
  notice(frame ? `${message}  (${frame.trim()})` : message, 'error');
  if (typeof error === 'object') console.error('scriptonait:', error);
}

function clearError() {
  $('notification-bar').hidden = true;
}

/// Run one action, posting a start notice immediately and exactly one
/// end notice however it finishes — every button click and background
/// job owes the bar at least that much.
///
/// `fn` may still post its own staged `notice()` calls for progress
/// (Import Project and startup do); those just get overwritten by the
/// final one here. `startLabel`/`endLabel` are short functional phrases
/// only ("Training"/"Trained") — never explanatory or decorative text.
/// If `fn`'s resolved value is a string, it replaces `endLabel` in the
/// success notice; anything else falls back to `${endLabel}.`.
async function withNotice(startLabel, endLabel, fn) {
  notice(`${startLabel}…`, 'info');
  try {
    const result = await fn();
    notice(typeof result === 'string' ? result : `${endLabel}.`, 'success');
    return result;
  } catch (error) {
    showError(`${endLabel} failed: ${(error && error.message) || error}`);
    throw error;
  }
}

// --- Tabs ----------------------------------------------------------------

function switchTab(name) {
  for (const button of document.querySelectorAll('#tabs .tab-button')) {
    button.classList.toggle('active', button.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll('.tab-panel')) {
    panel.hidden = panel.id !== `tab-${name}`;
  }
}

for (const button of document.querySelectorAll('#tabs .tab-button')) {
  button.addEventListener('click', () => switchTab(button.dataset.tab));
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

// --- Title bar -----------------------------------------------------------
//
// A background tab shows only its title, so that's where progress goes
// when the page isn't visible.

const BASE_TITLE = document.title;

function setTitleProgress(label, fraction) {
  if (!label) {
    document.title = BASE_TITLE;
    return;
  }
  const percent = fraction > 0 ? ` ${(fraction * 100).toFixed(0)}%` : '';
  document.title = `${label}${percent} — ${BASE_TITLE}`;
}

// --- Model state -------------------------------------------------------

let model = null;
let generating = false;
let training = false;
/// The promise a local run's own `#train-btn` handler is awaiting —
/// exposed so another action (Branch) can wait for a stop to genuinely
/// finish. `trainCall('stop', ...)` only flips a flag in the worker and
/// returns immediately; the training loop exits on its own schedule, and
/// this is the only thing that resolves once it actually has. `null`
/// whenever no local run is in flight. Not set for a remote run — there
/// is nothing here for Branch to wait on in that case, which is why it's
/// scoped out of this feature for now.
let activeTrainingCall = null;

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
  // Importing a checkpoint mid-run replaces the model out from under the
  // in-flight step — see the wasm side's own `busy` guard, which now
  // refuses this too; disabling it here is the friendlier first line of
  // defense.
  $('import-input').disabled = training;

  // Say what a step will actually cover, since batch size and context
  // multiply and neither number means much alone.
  const typedBatch = Number($('train-batch').value);
  const batch = chosenBatchSize();
  const context = model ? model.contextLen : Number($('cfg-context').value) || 0;
  const where = typedBatch > 0
    ? ''
    : machineProfile && profileShapeMatches(machineProfile)
      ? ' (measured)'
      : ' (default)';
  $('batch-hint').textContent =
    `${batch}${where} x ${context} = ${(batch * context).toLocaleString()} tokens per step.`;

  const explains = $('train-explains');
  explains.textContent = !canTrain
    ? 'Training needs WebGPU. This browser did not give the page a GPU.'
    : '';

  if (training) {
    step.textContent = 'Training on your GPU. Stop any time — progress is kept.';
  } else if (generating) {
    step.textContent = 'Writing…';
  } else if (!model && sources.length === 0) {
    step.textContent = 'Add your writing in Corpus.';
  } else if (!model && !enoughText) {
    step.textContent = `Only ${formatCount(words)} characters. Add more, then train.`;
  } else if (!model) {
    step.textContent = 'Ready to train.';
  } else if (!canTrain) {
    step.textContent = 'No WebGPU here, so this model can write but not train.';
  } else if (model.step < 500) {
    step.textContent = 'Barely trained. Keep training, or generate in Inference.';
  } else {
    step.textContent = '';
  }

  renderMachineProfile();
}

/// Hand the model everything already in the list.
///
/// Called whenever a model appears, because sources can be added before
/// one exists — that's the normal order now — and the model starts empty.
async function syncAllSources() {
  // The corpus this hands sources to is a fresh one, back at the wasm
  // side's own default rate — reapply whatever was last saved, the same
  // reason `set-source-sample-count`/`set-window-progress` below exist.
  const openingRate = Number($('opening-rate').value);
  if (openingRate >= 0) {
    trainCall('update-training-settings', { boundarySampleRate: openingRate / 100 }).catch(() => {});
  }
  // Counted, because syncSource swallows its failures — it has to, since
  // one unreadable record must not stop the other sixty-five. But a
  // silent partial hand-over means training on a fraction of the corpus
  // while every number on the page describes the whole of it, and
  // nothing anywhere says which.
  let handed = 0;
  const failed = [];
  for (const source of sources) {
    if (await syncSource(source)) handed += 1;
    else failed.push(source.title || source.id);
  }
  if (failed.length > 0) {
    console.warn(`[scriptonait] ${failed.length} sources did not reach the model:`, failed);
    showError(
      `${failed.length} of ${sources.length} sources could not be given to the model ` +
        `(${failed.slice(0, 3).join(', ')}${failed.length > 3 ? ', …' : ''}). ` +
        'It will train on the rest — remove and re-add those, or reload the page.',
    );
  } else if (handed > 0) {
    console.info(`[scriptonait] handed ${handed} sources to the model`);
  }
  await reportDuplicates();
  await refreshPlan();
}

/// Pull each source's current sample count and window-pass progress from
/// the wasm corpus and write them back to SOURCES_STORE, so "which
/// sources has training actually drawn from" and "how far into its own
/// pass over its windows is each one" both survive a reload — the second
/// is what stops a resumed run from replaying the same handful of
/// windows it had just drawn before the reload. Best effort — storage or
/// worker trouble here must not interrupt anything this isn't for.
async function flushSourceSampleCounts() {
  if (!model) return;
  try {
    const [{ sources: stats }, { sources: progress }] = await Promise.all([
      call('corpus-source-stats'),
      call('corpus-window-progress'),
    ]);
    const progressById = new Map(progress.map((p) => [p.id, { epoch: p.epoch, cursor: p.cursor }]));
    for (const { id, sampled } of stats) {
      const windowProgress = progressById.get(id);
      db.updateSourceStats(id, { timesSampled: sampled, windowProgress }).catch(() => {});
    }
  } catch (error) {
    /* best effort */
  }
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
    $('remove-duplicates-btn').hidden = !ids || ids.length === 0;
    if (!ids || ids.length === 0) return;
    const titles = ids
      .map((id) => (sources.find((s) => s.id === id) || {}).title || id)
      .slice(0, 3);
    showError(
      `${ids.length} source${ids.length === 1 ? ' is a copy' : 's are copies'} of another ` +
        `(${titles.join(', ')}${ids.length > 3 ? ', …' : ''}). Remove Copies in Corpus.`,
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
/// Layer sharing's own fields, keyed by mode — which fields exist and
/// which HTML elements hold them, so the render/read/toggle logic below
/// doesn't repeat this list three times.
const LAYER_SHARING_FIELDS = {
  grouped: [['unique-layers', 'uniqueLayers']],
  recurrent: [
    ['prelude-layers', 'preludeLayers'],
    ['coda-layers', 'codaLayers'],
    ['core-loop-min', 'coreLoopMin'],
    ['core-loop-max', 'coreLoopMax'],
  ],
};

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
  if (info) {
    $('cfg-layer-sharing').value = info.layerSharing;
    for (const mode of Object.keys(LAYER_SHARING_FIELDS)) {
      const active = info.layerSharing === mode;
      for (const [suffix, key] of LAYER_SHARING_FIELDS[mode]) {
        if (active) $(`cfg-${suffix}`).value = info[key];
        $(`cfg-${suffix}-field`).hidden = !active;
      }
    }
  }
  $('cfg-layer-sharing').disabled = Boolean(info);
  for (const fieldList of Object.values(LAYER_SHARING_FIELDS)) {
    for (const [suffix] of fieldList) $(`cfg-${suffix}`).disabled = Boolean(info);
  }
  $('shape-hint').textContent = info ? 'Fixed' : 'New model shape:';
  refreshShapeEstimate();
}

/// What the shape in the fields would cost, priced before anything is
/// built from it.
///
/// This exists because choosing a model shape is choosing a number of
/// hours and a quantity of GPU memory, and neither is guessable from
/// "12 layers, 516 hidden". The arithmetic is in `describe_shape` on the
/// wasm side — the same `ModelConfig` the model is actually built from,
/// so the estimate cannot drift from the thing it estimates.
///
/// Recomputed on every keystroke and needs no model, which is the whole
/// point: the moment somebody wants this answer is the moment before
/// there is a model to ask.
let shapeEstimateToken = 0;

/// A starting point for "Unique layers" when Layer sharing is switched to
/// Uniform groups: the largest divisor of the layer count that is at
/// most half of it, so turning sharing on visibly shrinks the model
/// instead of defaulting to a value indistinguishable from sharing being
/// off. A layer count with no such divisor (1, or a prime) has nothing
/// to offer but full sharing, and gets that.
function defaultUniqueLayers(numLayers) {
  for (let d = Math.floor(numLayers / 2); d >= 1; d--) {
    if (numLayers % d === 0) return d;
  }
  return numLayers;
}

/// A starting prelude/coda/loop-range split when Layer sharing is
/// switched to Recurrent core: one non-shared layer on each end, the
/// core looping through whatever depth is left (Geiping et al. 2025's
/// own shape has a small prelude and coda next to a much deeper core).
function defaultRecurrentCoreFields(numLayers) {
  const prelude = 1;
  const coda = 1;
  const coreLoopMax = Math.max(1, numLayers - prelude - coda);
  return { preludeLayers: prelude, codaLayers: coda, coreLoopMin: 1, coreLoopMax };
}

/// The Model Shape panel's current layer-sharing fields, read from
/// whichever mode is selected — the shape passed to `describe-shape` and
/// `create-model` alike, so the estimate can never drift from what
/// Create actually builds.
function currentLayerSharingFields() {
  const layers = Number($('cfg-layers').value) || 0;
  const layerSharing = $('cfg-layer-sharing').value;
  if (layerSharing === 'grouped') {
    return { layerSharing, uniqueLayers: Number($('cfg-unique-layers').value) || layers };
  }
  if (layerSharing === 'recurrent') {
    return {
      layerSharing,
      preludeLayers: Number($('cfg-prelude-layers').value) || 0,
      codaLayers: Number($('cfg-coda-layers').value) || 0,
      coreLoopMin: Number($('cfg-core-loop-min').value) || 1,
      coreLoopMax: Number($('cfg-core-loop-max').value) || 1,
    };
  }
  return { layerSharing: 'off' };
}

function formatBytes(bytes) {
  if (!bytes) return '—';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}

async function refreshShapeEstimate() {
  const box = $('shape-estimate');
  if (!box) return;
  // With a model loaded the fields state its shape and cannot be
  // changed, so there is nothing to price.
  if (model) {
    box.textContent = '';
    return;
  }
  const token = ++shapeEstimateToken;
  let estimate;
  try {
    estimate = await call('describe-shape', {
      layers: Number($('cfg-layers').value) || 0,
      ...currentLayerSharingFields(),
      hidden: Number($('cfg-hidden').value) || 0,
      heads: Number($('cfg-heads').value) || 0,
      kvHeads: Number($('cfg-kv-heads').value) || 0,
      contextLen: Number($('cfg-context').value) || 0,
      window: Number($('cfg-window').value) || 0,
      corpusChars: sources.reduce((sum, s) => sum + (s.rawText || '').length, 0),
    });
  } catch (error) {
    box.textContent = '';
    return;
  }
  // Keystrokes race: only the newest answer may write.
  if (token !== shapeEstimateToken) return;

  if (!estimate.valid) {
    box.textContent = `This shape will not build: ${estimate.problem}`;
    box.className = 'hint shape-estimate invalid';
    return;
  }

  const parts = [
    `${formatCount(estimate.params)} parameters`,
    `${formatBytes(estimate.trainingBytes)} of GPU memory to train ` +
      `(limit ${formatBytes(estimate.memoryLimitBytes)})`,
    `${formatBytes(estimate.inferenceBytes)} to generate`,
    `${estimate.headDim}-wide heads`,
    `${formatCount(estimate.ffnDim)} MLP width`,
    `${formatCount(estimate.vocabSize)}-token vocabulary from the text you have`,
  ];
  // Only worth saying when it is true, and worth saying plainly when it
  // is: a shape off the 64-grid pays this on every step, forever.
  if (estimate.tileEfficiency > 1.02) {
    parts.push(
      `about ${Math.round((estimate.tileEfficiency - 1) * 100)}% of every step wasted — the ` +
        'matmul kernels work in 64x64 blocks, and hidden size and context are the two ' +
        'dimensions that have to be multiples of 64 (they multiply, so being off on both ' +
        'costs both)',
    );
  }
  box.textContent = parts.join(' · ');
  box.className = 'hint shape-estimate';
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
  let sharingRow = '';
  if (info.layerSharing === 'grouped') {
    sharingRow = `<div><dt>Layer sharing</dt><dd>Uniform groups, ${info.uniqueLayers} unique</dd></div>`;
  } else if (info.layerSharing === 'recurrent') {
    sharingRow = `<div><dt>Layer sharing</dt><dd>Recurrent core, ${info.preludeLayers} prelude + ` +
      `${info.coreLoopMin}-${info.coreLoopMax} core loops + ${info.codaLayers} coda</dd></div>`;
  }
  $('model-details').innerHTML = `
    <dl>
      <div><dt>Parameters</dt><dd>${params}</dd></div>
      <div><dt>Layers</dt><dd>${info.layers}</dd></div>
      ${sharingRow}
      <div><dt>Hidden size</dt><dd>${info.hidden}</dd></div>
      <div><dt>Heads</dt><dd>${info.heads} (${info.kvHeads} key/value)</dd></div>
      <div><dt>Context</dt><dd>${info.contextLen} tokens, ${info.window}-token attention window</dd></div>
      <div><dt>Vocabulary</dt><dd>${info.vocabSize} tokens</dd></div>
      <div><dt>Training steps</dt><dd>${info.step.toLocaleString()}</dd></div>
      <div><dt>Training and generating on</dt><dd>${escapeHtml(info.device || 'no GPU — cannot train')}</dd></div>
    </dl>`;
  // Auto-save's suggested file name (Settings tab) is a live default
  // until something's been explicitly chosen — refresh it now that the
  // model its shape-based fallback reads from actually exists.
  if (!autosaveFileName) $('autosave-filename').value = autosaveTargetBaseName();
  if (!$('library-name').value) $('library-name').value = libraryDefaultName();

  // Core loops (Inference settings) only means anything for a Recurrent
  // core model — test-time compute scaling: the same checkpoint answers
  // at any depth in its trained range with no retraining.
  const recurrent = info.layerSharing === 'recurrent';
  $('opt-core-loops-field').hidden = !recurrent;
  if (recurrent) {
    const field = $('opt-core-loops');
    field.min = info.coreLoopMin;
    field.max = info.coreLoopMax;
    if (!field.value) field.value = info.coreLoopMax;
  }
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
// why a run refused to start. The worker failing to parse or throwing
// outside a handler entirely is LocalWasmBackend's concern (see its
// onFatalError), not this file's.
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
      console.warn('[scriptonait] this is a software renderer, not your GPU — training will be slow');
      notice('This browser gave the page a software renderer, not your GPU — training will be slow.', 'info');
    }
  } else {
    console.warn(
      `[scriptonait] no WebGPU (${reason || 'unavailable'}): generation runs on the CPU, ` +
        'and training cannot run at all',
    );
    notice(
      `No WebGPU device (${reason || 'unavailable'}): generation runs on the CPU, and training ` +
        'cannot run at all.',
      'error',
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
  stopGenerateTicker();
  const fraction = targetWords > 0 ? words / targetWords : 0;
  setProgress('generate-progress-bar', fraction);
  const of = targetWords > 0 ? ` of ${targetWords}` : '';
  $('generate-stats').textContent =
    `${words} words${of} · ${tokens} tokens · ${tokensPerSecond.toFixed(0)} tokens/s · ${formatDuration(elapsedSeconds)}`;
  setTitleProgress('Writing', fraction);
});

/// Continuous leaves length to whatever the prompt itself asks for (or
/// the default budget, if it asks for nothing) — today's behavior.
/// Limit hands the field's value straight through as a hard token
/// ceiling, overriding that regardless of what the prompt says.
function applyLengthMode() {
  $('opt-max-tokens').disabled = $('opt-length-mode').value !== 'limit';
}
$('opt-length-mode').addEventListener('change', applyLengthMode);
applyLengthMode();

/// generate-piece/generate-progress only start arriving once the model
/// has produced a first token — parse-prompt and whatever it takes to
/// get the first token out (a busy GPU, a cold wasm/GPU init) both
/// happen before that, with nothing to report yet. Left as a frozen
/// "Starting…" for however long that turns out to take, it reads
/// identically whether it's been one second or sixty. This ticks the
/// elapsed time instead, so a long wait still looks like it's doing
/// something rather than stuck.
let generateTicker = null;
function stopGenerateTicker() {
  if (generateTicker) {
    clearInterval(generateTicker);
    generateTicker = null;
  }
}

$('generate-btn').addEventListener('click', async () => {
  if (generating) return;
  const prompt = $('prompt-input').value.trim();
  if (!prompt) {
    showError('type what you want written first');
    return;
  }
  clearError();
  notice('Generating…', 'info');
  generating = true;
  $('generate-btn').disabled = true;
  $('stop-btn').hidden = false;
  $('stop-btn').disabled = false;
  $('generate-status').hidden = false;
  $('qa-notes').hidden = true;
  $('output').textContent = '';
  setProgress('generate-progress-bar', 0);
  $('generate-stats').textContent = 'Starting…';
  const generateStartedAt = performance.now();
  stopGenerateTicker();
  generateTicker = setInterval(() => {
    $('generate-stats').textContent =
      `Starting… ${Math.round((performance.now() - generateStartedAt) / 1000)}s`;
  }, 1000);

  try {
    const parsed = await call('parse-prompt', { prompt });
    targetWords = parsed.targetWords;

    const result = await call('generate', {
      prompt,
      temperature: Number($('opt-temperature').value),
      topK: Number($('opt-top-k').value),
      topP: Number($('opt-top-p').value),
      minP: Number($('opt-min-p').value),
      repetitionPenalty: Number($('opt-repetition').value),
      seed: Number($('opt-seed').value) || Math.floor(Math.random() * 1e9),
      maxTokens: $('opt-length-mode').value === 'limit' ? Number($('opt-max-tokens').value) : 0,
      coreLoops: Number($('opt-core-loops').value) || 0,
    // A bounded deadline instead of none: the GPU readback generate does
    // before its first token (sync_from_gpu_inner) has no cancellation
    // point wgpu exposes — Stop can't reach it, and neither can anything
    // else short of this timing out. Without one, a stuck readback left
    // `generating` true and both buttons disabled forever, with no way
    // back except reloading the page. Long enough that a real, slow
    // generation is never mistaken for a hang.
    }, [], 5 * 60 * 1000);
    setProgress('generate-progress-bar', 1);
    if (result.stopReason === 'no-data') {
      showError('Not enough text to generate from. Add some in Corpus.');
      return;
    }
    const why = {
      'end-of-text': 'the model ended the piece',
      length: 'reached the length you asked for',
      stopped: 'stopped',
    }[result.stopReason] || result.stopReason;
    $('generate-stats').textContent =
      `${result.wordCount} words in ${formatDuration(result.elapsedSeconds)} · ` +
      `${result.tokensPerSecond.toFixed(0)} tokens/s · ${why}`;
    renderNotes(result.notes);
    notice(`Generated ${result.wordCount} words — ${why}.`, 'success');
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
    stopGenerateTicker();
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

function newId() {
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
  if (button) withNotice('Removing source', 'Removed source', () => removeSource(button.dataset.id));
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
  await withNotice('Removing sources', 'Removed sources', async () => {
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
    await refreshPlan();
    return `Removed ${removed.length.toLocaleString()} source${removed.length === 1 ? '' : 's'}`;
  });
});

$('remove-duplicates-btn').addEventListener('click', async () => {
  let ids;
  try {
    ({ ids } = await call('duplicate-sources'));
  } catch (error) {
    showError(error);
    return;
  }
  if (!ids || ids.length === 0) {
    $('remove-duplicates-btn').hidden = true;
    return;
  }
  if (!confirm(`Remove ${ids.length.toLocaleString()} duplicate source${ids.length === 1 ? '' : 's'}? This can't be undone.`)) {
    return;
  }
  await withNotice('Removing duplicate sources', 'Removed duplicate sources', async () => {
    for (const id of ids) await removeSource(id);
    $('remove-duplicates-btn').hidden = true;
    return `Removed ${ids.length.toLocaleString()} duplicate source${ids.length === 1 ? '' : 's'}`;
  });
});

async function removeSource(id) {
  sources = sources.filter((source) => source.id !== id);
  renderSources();
  await persist('deleting a source', () => db.deleteSource(id));
  try {
    renderModel(await call('remove-source', { id }));
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
///
/// Empty and no note: leaves this line blank rather than repeating
/// what the sources list's own empty state ("Nothing added yet.") right
/// above it already says.
function updateSourceSummary(list, note = '') {
  const stats = $('corpus-stats');
  if (!list || list.length === 0) {
    stats.textContent = note;
    return;
  }
  const chars = list.reduce((sum, s) => sum + (s.rawText || '').length, 0);
  const saved = persistenceWorks ? '' : ' · not saved';
  stats.textContent =
    `${list.length} source${list.length === 1 ? '' : 's'}, ` +
    `${formatCount(chars)} characters${saved}${note ? ` · ${note}` : ''}`;
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
    // The corpus this source just joined is a fresh one — its own count
    // of how many training windows have come from this source starts at
    // 0 regardless of what happened before a reload. Hand back whatever
    // was persisted last time, if anything.
    if (source.timesSampled) {
      call('set-source-sample-count', { id: source.id, count: source.timesSampled }).catch(() => {});
    }
    // Same idea for how far training got through this source's own pass
    // over its windows — without this, a reload would start that pass
    // over from its first window instead of continuing it.
    if (source.windowProgress) {
      call('set-window-progress', {
        id: source.id,
        epoch: source.windowProgress.epoch,
        cursor: source.windowProgress.cursor,
      }).catch(() => {});
    }
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

    // A file added again under the same name is an update, not a second
    // copy of it — re-adding an edited draft used to duplicate it,
    // leaving the only fix a full remove-everything-and-re-add-it-all,
    // which resets every other source's window progress and held-out
    // split along with it. Reusing the id turns this into what
    // upsert_source already is: a replace, not an insert — the corpus
    // keeps this source's sample count and progress, only its text
    // (and, through that, its future windows) changes.
    const existing = sources.find((s) => s.title === entry.title);
    const source = {
      id: existing ? existing.id : newId(),
      title: entry.title,
      kind: entry.kind,
      rawText,
      sourceUrl: entry.sourceUrl || null,
      createdAt: existing ? existing.createdAt : Date.now(),
      // Carried over so the stored record doesn't lose them the moment
      // this overwrites it — the live corpus already keeps its own
      // count regardless (upsert only initializes it if missing), but
      // IndexedDB only knows what's on the record it's given.
      timesSampled: existing ? existing.timesSampled : undefined,
      windowProgress: existing ? existing.windowProgress : undefined,
    };
    if (existing) {
      sources[sources.indexOf(existing)] = source;
    } else {
      sources.push(source);
    }
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
  // The corpus just changed, so every number the plan is built from
  // did too. Without this the plan keeps reporting the corpus it was
  // last computed against.
  await refreshPlan();
  // And the vocabulary a new model would be built with scales with how
  // much text there is, so the shape estimate moved too.
  refreshShapeEstimate();
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
/// The step/loss/held-out/tokens-per-second line, drawn onto the chart
/// itself by drawLossChart rather than kept as separate DOM text.
let chartStats = '';

/// What the most recent step's batch actually trained on — the text
/// itself, not just which document, since a batch can (and, thanks to
/// the source rotation, usually does) draw from several different ones
/// per step, not the same document run after run.
function describeBatchSources(draws, maxChars) {
  if (!draws || draws.length === 0) return '';
  const limit = maxChars > 0 ? maxChars : 200;
  const shown = draws.slice(0, 3).map((draw) => {
    const source = sources.find((s) => s.id === draw.id);
    const title = source ? source.title : draw.id;
    let excerpt = (draw.excerpt || '').replace(/\s+/g, ' ').trim();
    if (excerpt.length > limit) excerpt = `${excerpt.slice(0, limit)}…`;
    return `${title}: "${excerpt}"`;
  });
  const rest = draws.length > 3 ? ` +${draws.length - 3} more` : '';
  return `Training on: ${shown.join(' | ')}${rest}`;
}

onStream('train-progress', (progress) => {
  // Kept live so the model's own step count never lags behind what the
  // worker is reporting — a stale step here is what made every history
  // row look orphaned mid-run (row.step compared against a step frozen
  // at whenever the run started).
  if (model) model.step = progress.step;
  setProgress('train-progress-bar', progress.fractionDone);
  const on = model && model.device ? ` on ${model.device}` : '';
  // Held-out loss is the one that says whether it is learning the
  // language or the sample, so it sits next to the training loss.
  const held =
    typeof progress.validationLoss === 'number'
      ? ` · held-out ${progress.validationLoss.toFixed(3)}`
      : '';
  // Drawn onto the chart itself (see drawLossChart) rather than as a
  // separate line of text — the chart is the thing being read while
  // this updates, so the numbers belong where the eye already is.
  chartStats =
    `step ${progress.step.toLocaleString()} · loss ${progress.smoothedLoss.toFixed(3)}${held} · ` +
    `${progress.tokensPerSecond.toFixed(0)} tokens/s${on}`;
  // Once real numbers are flowing they live on the chart; leaving the
  // last one-off status message ("Starting…") sitting above it would
  // just be a stale leftover.
  $('train-stats').textContent = '';
  $('train-window').textContent =
    $('show-training-window').value === 'off'
      ? ''
      : describeBatchSources(progress.sources, Number($('training-window-chars').value));
  setTitleProgress('Fine-tuning', progress.fractionDone);
  lossHistory.push({ step: progress.step, loss: progress.smoothedLoss });
  if (
    typeof progress.validationLoss === 'number' &&
    (validationHistory.length === 0 ||
      validationHistory[validationHistory.length - 1].loss !== progress.validationLoss)
  ) {
    validationHistory.push({ at: progress.step, loss: progress.validationLoss });
    if (typeof progress.trainingProbe === 'number' && progress.trainingProbe >= 0) {
      probeHistory.push({ at: progress.step, loss: progress.trainingProbe });
    }
  }
  drawLossChart();
});

// Samples from the model as it trains. Exactly one card, rewritten in
// place every time a sample arrives: `replaceChildren` runs on every
// event, so the box holds this card and nothing else no matter what was
// in it before. Never append - a stack of stale samples buries the only
// one worth reading, which is the current one.
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
    $('overview-plan').hidden = true;
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
  // The schedule is anchored to the model's lifetime step, not to when
  // this run happened to start, so there is only one frame to show it
  // in: step N of the plan.
  const progress = [
    n.plannedSteps > n.step
      ? `step ${n.step.toLocaleString()} of ${n.plannedSteps.toLocaleString()} planned`
      : `step ${n.step.toLocaleString()}`,
  ];
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
    progress.push(`${formatDuration(n.etaSeconds)} left in the plan`);
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
  $('overview-plan').hidden = false;
  $('overview-plan').textContent = `${plan.phase.title} — ${progress.join(' · ')}`;

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
    if (lastPhaseKey !== null) {
      notice(`${plan.phase.title} — ${plan.phase.detail}`, 'info');
    }
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
  refreshCorpusStats();
}

/// Per-source token counts and how many times each has actually been
/// sampled, on the Overview tab — updates on the same events refreshPlan
/// does (a source added or removed, a model loaded or created), so it
/// never shows a source list that's drifted from what's on screen above.
async function refreshCorpusStats() {
  const panel = $('overview-corpus-panel');
  if (!model) {
    panel.hidden = true;
    return;
  }
  try {
    const { sources: stats } = await call('corpus-source-stats');
    const titleFor = (id) => (sources.find((s) => s.id === id) || {}).title || id;
    const totalSampled = stats.reduce((sum, s) => sum + s.sampled, 0) || 1;
    const body = $('overview-corpus-body');
    body.replaceChildren();
    for (const s of [...stats].sort((a, b) => b.sampled - a.sampled)) {
      const row = document.createElement('tr');
      const share = Math.round((s.sampled / totalSampled) * 100);
      row.innerHTML =
        `<td>${escapeHtml(titleFor(s.id))}</td><td>${formatCount(s.trainTokens)}</td>` +
        `<td>${formatCount(s.heldOutTokens)}</td><td>${s.sampled.toLocaleString()}${
          s.sampled > 0 ? ` (${share}%)` : ''
        }</td>`;
      body.append(row);
    }
    panel.hidden = stats.length === 0;
  } catch (error) {
    console.warn('[scriptonait] could not read corpus stats', error);
  }
}

// A point-in-time event (a reactive rate cut just fired — see worker.js's
// cosine-cuts branch), not a standing verdict — the Progress panel's own
// action list (planActions, driven by trainingPhase) already covers "what
// does the curve look like right now" persistently, so this is a toast,
// not a second box repeating it.
onStream('train-advice', ({ advice, step }) => {
  notice(advice, 'info');
  console.info(`[scriptonait] advice at step ${step.toLocaleString()}: ${advice}`);
});

// The single overwriting sample card is gone \u2014 the Samples panel (backed
// by train-record events, see renderSampleHistory) is the only place a
// training sample shows up now, and it keeps every one rather than just
// the latest.


// --- Run history -------------------------------------------------------
//
// Everything else on this page shows the present moment. A training run
// is six hours long, and the question worth asking is almost always
// "what did it do between then and now" — which, until this existed, was
// answerable only from whatever console lines had not scrolled away.
//
// Two kinds of record share one timeline: measurements (a row of numbers
// every hundred steps) and events (a run starting, a rate being cut, a
// sample being generated). They are kept together because a loss curve
// with an unexplained bend in it is worse than no curve, and the bend is
// always an event.
//
// It is all copyable, in Markdown for reading and JSON for machines. The
// JSON is also the shape an MCP server would serve if the app is ever
// wired up to one — the format is the interface, so building it now
// costs nothing later.

const history = [];

/// Columns, in order. Each is a label, a reader, and how to render it —
/// kept in one place so the table, the Markdown and the JSON cannot
/// drift into disagreeing about what a run recorded.
const HISTORY_COLUMNS = [
  // Which run a row came from. Without it the table silently splices
  // together timelines that never followed one another: a model
  // recovered from a save is *behind* rows already recorded by a run
  // that got further and was then lost, so the table shows a future
  // that no longer exists next to a present that is behind it.
  ['run', (r) => r.runId, (v) => String(v).replace(/^run-/, '')],
  ['step', (r) => r.step, (v) => v.toLocaleString()],
  ['tokens', (r) => r.tokensSeen, (v) => formatCount(v)],
  ['passes', (r) => r.epochs, (v) => v.toFixed(2)],
  ['loss', (r) => r.loss, (v) => v.toFixed(3)],
  ['probe', (r) => r.probe, (v) => (v >= 0 ? v.toFixed(3) : '—')],
  ['held-out', (r) => r.heldOut, (v) => v.toFixed(3)],
  ['gap', (r) => r.gap, (v) => (v === null ? '—' : v.toFixed(3))],
  ['bits/byte', (r) => r.bitsPerByte, (v) => (v > 0 ? v.toFixed(3) : '—')],
  ['lr', (r) => r.lr, (v) => v.toExponential(2)],
  ['schedule', (r) => r.scheduleMode, (v) => v || '—'],
  ['x sched', (r) => r.plateauScale, (v) => v.toFixed(2)],
  ['|grad|', (r) => r.gradNorm, (v) => v.toFixed(2)],
  ['tok/s', (r) => r.tokensPerSecond, (v) => Math.round(v).toLocaleString()],
  ['real words', (r) => (r.quality ? r.quality.knownWordRate : null),
    (v) => (v === null ? '—' : `${Math.round(v * 100)}%`)],
  ['repeats', (r) => (r.quality ? r.quality.repeated4gramRate : null),
    (v) => (v === null ? '—' : `${Math.round(v * 100)}%`)],
  ['phase', (r) => r.phase, (v) => v || '—'],
];

function historyCell(row, [, read, render]) {
  const value = read(row);
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return typeof value === 'number' || typeof value === 'string' ? render(value) : '—';
}

const measurements = () => history.filter((r) => r.kind === 'measurement');
const samples = () => history.filter((r) => r.kind === 'sample');

/// Rebuild the chart's three curves from persisted history, so a
/// reload (or a project import) shows the run's actual shape instead of
/// a blank chart that only starts filling in once training resumes —
/// the numbers to draw it were sitting in storage the whole time, just
/// never read back into the arrays drawLossChart reads from.
///
/// Measurements come far sparser than the live per-tick training curve
/// did (once every `metricsEvery` steps, not every progress tick) —
/// each point here carries its own step number rather than relying on
/// array position to stand in for one, which is what let a resumed
/// run's much denser live ticks warp the x-axis against this coarser
/// rebuilt portion (500 steps per point here, roughly 1 per point once
/// live training resumed — the same index space, two very different
/// meanings).
function rebuildChartFromHistory() {
  lossHistory.length = 0;
  validationHistory.length = 0;
  probeHistory.length = 0;
  const rows = measurements()
    .filter((r) => typeof r.loss === 'number')
    .sort((a, b) => a.step - b.step);
  for (const row of rows) {
    lossHistory.push({ step: row.step, loss: row.loss });
    if (typeof row.heldOut === 'number') {
      validationHistory.push({ at: row.step, loss: row.heldOut });
    }
    if (typeof row.probe === 'number' && row.probe >= 0) {
      probeHistory.push({ at: row.step, loss: row.probe });
    }
  }
  if (lossHistory.length > 0) $('loss-chart').hidden = false;
  drawLossChart();
}

function renderHistory() {
  const table = $('history-table');
  if (!table) return;
  const rows = measurements();
  $('history-count').textContent = rows.length
    ? `· ${rows.length} measurement${rows.length === 1 ? '' : 's'}, ` +
      `${samples().length} sample${samples().length === 1 ? '' : 's'}`
    : '· nothing recorded yet';

  const head = `<thead><tr>${HISTORY_COLUMNS.map(([label]) => `<th>${label}</th>`).join('')}` +
    '</tr></thead>';
  // Newest last, so the table reads in the direction the run ran and the
  // bottom row is the present.
  const body = rows
    .map((row) => {
      const orphan = model && row.step > model.step + 1;
      return `<tr${orphan ? ' class="orphan"' : ''}>` +
        `${HISTORY_COLUMNS.map((col) => `<td>${historyCell(row, col)}</td>`).join('')}</tr>`;
    })
    .join('');
  table.innerHTML = `${head}<tbody>${body}</tbody>`;
  // Keep the newest row in view, unless the user has scrolled up to look
  // at something — in which case leave them where they are.
  const wrap = table.parentElement;
  if (wrap && wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 60) {
    wrap.scrollTop = wrap.scrollHeight;
  }

  const events = history.filter((r) => r.kind && r.kind !== 'measurement' && r.kind !== 'sample');
  $('history-events').innerHTML = events
    .slice(-40)
    .map(
      (e) =>
        `<div><span class="step">step ${Number(e.step || 0).toLocaleString()}</span>${
          escapeHtml(String(e.text || e.kind))
        }</div>`,
    )
    .join('');

  renderSampleHistory();
}

/// Which stored sample is on screen. -1 means "the newest", and it stays
/// meaning that as new ones arrive, so a panel left alone keeps up while
/// one somebody has paged back through does not jump.
let sampleCursor = -1;

function renderSampleHistory() {
  const all = samples();
  const box = $('sample-history');
  if (all.length === 0) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const index = sampleCursor < 0 ? all.length - 1 : Math.min(sampleCursor, all.length - 1);
  const sample = all[index];
  const quality = sample.quality;
  $('sample-index').value = String(index + 1);
  $('sample-index').max = String(all.length);
  $('sample-total').textContent = String(all.length);
  $('sample-step').textContent = Number(sample.step).toLocaleString();
  $('sample-history-head').textContent = [
    `step ${Number(sample.step).toLocaleString()}`,
    typeof sample.loss === 'number' ? `loss ${sample.loss.toFixed(3)}` : null,
    quality && quality.words ? `${Math.round(quality.knownWordRate * 100)}% real words` : null,
    quality && quality.repeated4gramRate > 0.05
      ? `${Math.round(quality.repeated4gramRate * 100)}% repeated runs`
      : null,
  ].filter(Boolean).join(' · ');
  $('sample-history-text').textContent = sample.text || '';
  $('sample-prev').disabled = index === 0;
  $('sample-next').disabled = index === all.length - 1;
}

/// The whole history as Markdown: a header of what was being trained and
/// on what, then the table, then the events, then the samples.
///
/// Written to be pasted into a conversation. That is a real use — the
/// person running this cannot read a loss curve as fast as they can ask
/// somebody about it, and a screenshot of one line is not enough to
/// answer with.
function historyAsMarkdown() {
  const rows = measurements();
  const start = history.find((r) => r.kind === 'run-started');
  const out = ['# scriptonait run history', ''];

  if (start && start.model) {
    const m = start.model;
    const c = start.corpus || {};
    const s = start.settings || {};
    out.push('## Model', '');
    out.push(`- ${formatCount(m.params)} parameters — ${m.layers} layers, ${m.hidden} hidden, ` +
      `${m.heads} heads (${m.kvHeads} key/value), context ${m.contextLen}, window ${m.window}, ` +
      `vocabulary ${m.vocabSize}`);
    out.push(`- Corpus: ${c.sources} sources, ${formatCount(c.chars)} characters, ` +
      `${formatCount(c.trainingTokens)} training tokens, ` +
      `${formatCount(c.validationTokens)} held out`);
    out.push(`- Run: ${s.plannedSteps} planned steps, batch ${s.batchSize}, ` +
      `${s.tokensPerStep} tokens/step, peak rate ${s.peakLr}, warm-up ${s.warmupSteps}, ` +
      `weight decay ${s.weightDecay}, grad clip ${s.gradClip}`);
    out.push(`- Device: ${start.device}`);
    out.push('');
  }

  if (rows.length) {
    out.push('## Measurements', '');
    out.push(`| ${HISTORY_COLUMNS.map(([label]) => label).join(' | ')} |`);
    out.push(`|${HISTORY_COLUMNS.map(() => '---').join('|')}|`);
    for (const row of rows) {
      out.push(`| ${HISTORY_COLUMNS.map((col) => historyCell(row, col)).join(' | ')} |`);
    }
    out.push('');
  }

  const events = history.filter((r) => r.kind && r.kind !== 'measurement' && r.kind !== 'sample');
  if (events.length) {
    out.push('## Events', '');
    for (const e of events) {
      out.push(`- **step ${Number(e.step || 0).toLocaleString()}** — ${e.text || e.kind}`);
    }
    out.push('');
  }

  const all = samples();
  if (all.length) {
    out.push('## Samples', '');
    // The first, a few through the middle, and the last: enough to see
    // the trajectory without pasting fifty of them.
    const wanted = all.length <= 6
      ? all
      : [0, 1, Math.floor(all.length / 3), Math.floor((2 * all.length) / 3),
        all.length - 2, all.length - 1].map((i) => all[i]);
    for (const sample of wanted) {
      const q = sample.quality;
      out.push(`### step ${Number(sample.step).toLocaleString()}` +
        (typeof sample.loss === 'number' ? ` — loss ${sample.loss.toFixed(3)}` : '') +
        (q && q.words ? `, ${Math.round(q.knownWordRate * 100)}% real words` : ''));
      out.push('');
      out.push('```');
      out.push((sample.text || '').trim());
      out.push('```');
      out.push('');
    }
  }
  return out.join('\n');
}

/// Put the text where it can be copied, by whatever means works.
///
/// `navigator.clipboard.writeText` needs a secure context, a focused
/// document and an unexpired user gesture, and it fails by doing
/// nothing — which is what happened: a button that appears to work and
/// leaves an empty clipboard is worse than one that plainly cannot.
///
/// So the text always lands in a visible box first, selected and ready
/// for Ctrl+C. The clipboard write is then attempted as a convenience
/// on top of that, and its failure costs nothing because the text is
/// already on screen.
async function copyToClipboard(text, label) {
  const box = $('history-output');
  box.value = text;
  box.hidden = false;
  box.focus();
  box.select();

  let copied = false;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (error) {
    console.info(`[scriptonait] the clipboard refused (${error && error.message}); ` +
      'the text is in the box on the page');
  }
  if (!copied) {
    // The older path, which works in places the async one does not.
    try {
      copied = document.execCommand('copy');
    } catch (error) {
      copied = false;
    }
  }
  notice(
    copied ? `Copied ${label}.` : `${label} is in the box below, selected — press Ctrl+C.`,
    copied ? 'success' : 'info',
  );
}

// A run that ends before you asked it to has to say so, loudly. It used
// to end in silence: the page kept showing the last sample it received
// and nothing said the steps had stopped.
onStream('train-stopped', ({ step, reason }) => {
  // A local run's own click handler already does this reset once its
  // `call('train', ...)` promise settles — this runs again there
  // (idempotent) and is the only place a remote run's ever gets it,
  // since nothing awaits a remote run to completion the way the click
  // handler awaits a local one.
  training = false;
  $('train-btn').disabled = false;
  $('train-stop-btn').hidden = true;
  $('live-controls').hidden = true;
  setTitleProgress(null);
  updateGuidance();
  refreshPlan().catch(() => {});
  if (reason === 'finished') {
    $('train-stats').textContent = `Finished at step ${step.toLocaleString()} — press Train to continue`;
    notice(`Training finished at step ${step.toLocaleString()}.`, 'success');
    return;
  }
  console.error(`[scriptonait] training stopped at step ${step.toLocaleString()}: ${reason}`);
  showError(
    `Training stopped at step ${step.toLocaleString()}: ${reason}. ` +
      'The model is saved — press Train to continue from where it stopped.',
  );
  $('train-stats').textContent = `Stopped at step ${step.toLocaleString()}: ${reason}`;
});

onStream('train-autosave', ({ step, bytes }) => {
  autosave(step, { bytes });
});

/// The bridge for "train remote, infer local": a `RemoteBackend`'s own
/// periodic and end-of-run checkpoint pulls arrive here rather than as
/// `train-autosave` directly, since a remote run's bytes also need
/// loading into this browser's own WASM model before anything local
/// (Generate, a later local Train) sees them — a local run's own worker
/// already holds that model, so it never needs this step.
onStream('remote-checkpoint', async ({ step, bytes }) => {
  try {
    const info = await call('import-checkpoint', { bytes });
    if (info && !info.error) renderModel(info);
  } catch (error) {
    console.warn('[scriptonait] could not load a synced remote checkpoint locally', error);
  }
  autosave(step, { bytes });
});

onStream('train-record', async (record) => {
  // Seed the live control from what the run actually started with, so
  // pressing Apply without editing it is a no-op rather than a surprise.
  if (record.kind === 'run-started' && record.settings && record.settings.peakLr) {
    $('live-lr').value = record.settings.peakLr;
  }
  // The message carries `type` as well, from `post`; drop it so a stored
  // record is exactly what it claims to be.
  const { type, ...row } = record;
  history.push(row);
  renderHistory();
  try {
    // appendHistory synthesizes the store's id; backfill it onto the
    // same object already sitting in `history` (not a copy — this
    // mutates what's there) so a mid-session Export Project carries a
    // real id on every row instead of crashing the next Import on
    // "key path did not yield a value".
    const stored = await db.appendHistory(row);
    row.id = stored.id;
  } catch (error) {
    console.warn('[scriptonait] could not store a history record', error);
    notice(`Could not save a history record: ${(error && error.message) || error}.`, 'error');
  }
});

/// Build the text, then hand it over. Wrapped because a throw inside
/// the builder used to take the click with it and leave no trace: the
/// button did nothing, said nothing, and looked broken for a reason
/// nobody could see.
function copyHistory(build, label) {
  let text;
  try {
    text = build();
  } catch (error) {
    console.error('[scriptonait] could not build the history text', error);
    showError(`the history could not be assembled: ${(error && error.message) || error}`);
    return;
  }
  copyToClipboard(text, label).catch((error) => showError(error));
}

$('history-copy-btn').addEventListener('click', () =>
  copyHistory(historyAsMarkdown, 'as Markdown'));

$('history-json-btn').addEventListener('click', () =>
  copyHistory(() => JSON.stringify(history, null, 2), 'as JSON'));

$('history-clear-btn').addEventListener('click', async () => {
  history.length = 0;
  sampleCursor = -1;
  renderHistory();
  try {
    await db.clearHistory();
  } catch (error) {
    console.warn('[scriptonait] could not clear the history', error);
    notice(`Could not clear the stored history: ${(error && error.message) || error}.`, 'error');
  }
});

$('sample-prev').addEventListener('click', () => {
  const all = samples();
  const index = sampleCursor < 0 ? all.length - 1 : sampleCursor;
  sampleCursor = Math.max(0, index - 1);
  renderSampleHistory();
});

$('sample-next').addEventListener('click', () => {
  const all = samples();
  const index = sampleCursor < 0 ? all.length - 1 : sampleCursor;
  // Stepping onto the newest goes back to following it, rather than
  // pinning to whatever index the newest happens to be right now.
  sampleCursor = index + 1 >= all.length - 1 ? -1 : index + 1;
  renderSampleHistory();
});

$('sample-index').addEventListener('change', (event) => {
  const all = samples();
  if (all.length === 0) return;
  const typed = Math.round(Number(event.target.value));
  const requested = Number.isFinite(typed) ? Math.min(all.length, Math.max(1, typed)) : all.length;
  // Same convention as "Later": landing on the newest resumes following
  // it, rather than pinning to whatever index happens to be newest now.
  sampleCursor = requested >= all.length ? -1 : requested - 1;
  renderSampleHistory();
});

$('reset-schedule-btn').addEventListener('click', async () => {
  try {
    const result = await trainCall('reset-schedule');
    notice(
      `Schedule restored from ${result.was.toFixed(2)}x to full strength — rate in force ` +
        `${result.lrNow.toExponential(2)}.`,
      'success',
    );
  } catch (error) {
    showError(error);
  }
});

$('live-lr-btn').addEventListener('click', async () => {
  const rate = Number($('live-lr').value);
  if (!(rate > 0)) return;
  try {
    // autoLearningRate: false, or Auto mode would recompute and silently
    // overwrite this override on the very next settings push — the
    // point of an override is that it actually takes effect.
    const result = await trainCall('update-training-settings', { peakLearningRate: rate, autoLearningRate: false });
    notice(`Peak learning rate is now ${result.peakLr}.`, 'success');
  } catch (error) {
    showError(error);
  }
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
    s.layerSharing === model.layerSharing &&
    s.uniqueLayers === model.uniqueLayers &&
    s.preludeLayers === model.preludeLayers &&
    s.codaLayers === model.codaLayers &&
    s.coreLoopMax === model.coreLoopMax &&
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
  if (!text) return;
  if (benchmarking) {
    text.textContent = 'Measuring this machine…';
    return;
  }
  // Auto-benchmark off only means this shape won't be *measured*
  // automatically — chosenBatchSize() still uses an existing matching
  // measurement regardless of this toggle, so "falls back to what is
  // typed, or 1" is only true when there is no matching measurement to
  // fall back on either; used to be shown whenever the toggle was off,
  // even with a perfectly good matching profile already in hand.
  if (!benchmarkAutoEnabled && (!machineProfile || !profileShapeMatches(machineProfile))) {
    text.textContent = 'Auto-benchmark is off. Batch size falls back to what is typed, or 1.';
    return;
  }
  if (!machineProfile) {
    text.textContent = gpuReport
      ? 'Machine profile: not measured yet. The first training run measures it once.'
      : 'Machine profile: needs a GPU.';
    return;
  }
  const p = machineProfile;
  const stale = model && !profileShapeMatches(p);
  text.textContent =
    `Machine profile: ${p.adapter} — ${p.dispatchesPerSubmit} dispatches per command buffer, ` +
    `batch ${p.batchSize}, ${Math.round(p.msPerStep)} ms per step ` +
    `(${Math.round(p.tokensPerSecond)} tokens/s)` +
    (stale ? '. Measured on a differently shaped model — measure again for its batch size.' : '.');
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
  renderMachineProfile();
  try {
    const result = await call('benchmark', {}, [], 0);
    if (result.error) {
      console.warn(`[scriptonait] benchmark: ${result.error}`);
      notice(`Machine benchmark failed: ${result.error}. Falling back to safe defaults.`, 'error');
      return machineProfile;
    }
    machineProfile = await db.putMachineProfile(result.profile);
    console.info('[scriptonait] machine profile stored', machineProfile);
    return machineProfile;
  } finally {
    benchmarking = false;
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

/// Auto disables Batch/Effort/Learning-rate and lets the existing
/// auto-pick logic choose them; Manual hands them back and, per the Train
/// button handler, refuses to start at learning rate 0 — in Manual mode 0
/// means "no rate set", not "pick one". Steps stays editable in both
/// modes: it's the project's planned length, not a choice about how
/// training happens, so it means the same thing whichever mode picks the
/// rest.
function applyTrainMode() {
  const manual = $('train-mode').value === 'manual';
  for (const id of ['train-batch', 'train-effort', 'train-lr']) {
    $(id).disabled = !manual;
  }
  // Batch and Effort each have their own "pick one" sentinel (0, "auto")
  // that the auto-pick logic already reads correctly — but only if a
  // leftover typed value from a previous Manual session isn't still
  // sitting in the field. Steps and Learning rate need no such reset:
  // Steps 0 means "until stopped" in both modes, and the Train button
  // reads Learning rate only when Manual is selected.
  if (!manual) {
    $('train-batch').value = '0';
    $('train-effort').value = 'auto';
  }
  updateGuidance();
}
$('train-mode').addEventListener('change', applyTrainMode);
// Not called eagerly here: it reaches into updateGuidance() ->
// renderMachineProfile(), which reads module state (benchmarkAutoEnabled)
// declared later in this file. Calling it at this point in top-level
// module evaluation throws "Cannot access before initialization" on
// every page load. start() calls it once, unconditionally, after every
// module-level `let` above has run — see applyLoadedSettings().

/// The wire value worker.js/llm.set_schedule_kind still expect — 'wsd'
/// only for a deferred cooldown, 'cosine-cuts' for reactive plateau-cuts
/// (which always run against an immediate cooldown; see
/// applySchedulerCompatibility), 'cosine' otherwise. Kept as one string
/// rather than widening the wasm/worker protocol, since Rust only ever
/// distinguishes "wsd" from "not wsd" and worker.js only ever asks "is
/// this 'cosine-cuts'" — see llm-core::train::ScheduleKind and this
/// file's own set_schedule_kind doc comment.
function scheduleModeFromAxes({ stablePhase, cooldownShape }) {
  if (stablePhase === 'reactive-cuts') return 'cosine-cuts';
  return cooldownShape === 'deferred' ? 'wsd' : 'cosine';
}

/// The reverse of scheduleModeFromAxes, for a project saved before the
/// two axes existed — its only record of the schedule is this string.
function axesFromScheduleMode(scheduleMode) {
  if (scheduleMode === 'cosine-cuts') {
    return { stablePhase: 'reactive-cuts', cooldownShape: 'immediate' };
  }
  if (scheduleMode === 'cosine') {
    return { stablePhase: 'flat', cooldownShape: 'immediate' };
  }
  return { stablePhase: 'flat', cooldownShape: 'deferred' };
}

/// Which of the other Manual-mode scheduler selects a given choice rules
/// out, and what to coerce them to rather than leave them on a value
/// that no longer means anything. Reactive plateau-cuts combined with a
/// deferred (WSD) cool-down, or with an adaptively-extended plan, is a
/// documented failure mode — this session's own plateau-cut death
/// spiral — so it forces the cool-down back to Immediate and the plan
/// length back to Fixed, the simple, well-tested shape today's
/// "cosine-cuts" already is, rather than leaving either combination
/// reachable. Decay start only means anything once there is a deferred
/// cool-down to place within.
function applySchedulerCompatibility() {
  const reactive = $('stable-phase').value === 'reactive-cuts';

  $('cooldown-shape').disabled = reactive;
  if (reactive) $('cooldown-shape').value = 'immediate';

  const deferred = $('cooldown-shape').value === 'deferred' && !reactive;
  $('decay-start').disabled = !deferred;
  if (!deferred) $('decay-start').value = 'fixed';

  $('plan-length').disabled = reactive;
  if (reactive) $('plan-length').value = 'fixed';
}

/// Auto owns every scheduler axis; Manual hands them back, subject to
/// applySchedulerCompatibility's rules. Mirrors applyTrainMode: Auto
/// doesn't just disable the five selects, it resets them to this app's
/// own considered defaults (plan-based warm-up, flat stable phase,
/// deferred/WSD cool-down at the fixed fraction, a fixed plan length) —
/// otherwise "Auto" would silently keep locking in whatever a previous
/// Manual session happened to leave typed into these same controls,
/// which is Manual with the door locked, not Auto.
function applySchedulerMode() {
  const manual = $('scheduler-mode').value === 'manual';
  for (const id of ['warmup-strategy', 'stable-phase', 'cooldown-shape', 'decay-start', 'plan-length']) {
    $(id).disabled = !manual;
  }
  if (manual) {
    applySchedulerCompatibility();
  } else {
    $('warmup-strategy').value = 'plan';
    $('stable-phase').value = 'flat';
    $('cooldown-shape').value = 'deferred';
    $('decay-start').value = 'fixed';
    $('plan-length').value = 'fixed';
  }
  updateGuidance();
}
$('scheduler-mode').addEventListener('change', applySchedulerMode);
for (const id of ['stable-phase', 'cooldown-shape']) {
  $(id).addEventListener('change', applySchedulerCompatibility);
}

/// Everything on this tab that used to reset to the markup's hardcoded
/// defaults on every reload. Written through on change, loaded back at
/// startup (see `start()`) — the same immediately-applied pattern the
/// other Settings-tab controls already use.
async function persistTrainingPlanSettings() {
  const axes = {
    stablePhase: $('stable-phase').value,
    cooldownShape: $('cooldown-shape').value,
  };
  await db.putTrainingPlanSettings({
    mode: $('train-mode').value,
    plannedSteps: Number($('train-steps').value) || 0,
    effort: $('train-effort').value,
    batchSize: Number($('train-batch').value) || 0,
    learningRate: Number($('train-lr').value) || 0,
    sampleToggle: $('sample-toggle').checked,
    sampleEvery: Number($('sample-every').value) || 0,
    boundarySampleRate: Number($('opening-rate').value) / 100,
    metricsEvery: Number($('metrics-every').value) || 0,
    showTrainingWindow: $('show-training-window').value !== 'off',
    trainingWindowChars: Number($('training-window-chars').value) || 0,
    schedulerMode: $('scheduler-mode').value,
    warmupStrategy: $('warmup-strategy').value,
    stablePhase: axes.stablePhase,
    cooldownShape: axes.cooldownShape,
    decayStartAdaptive: $('decay-start').value === 'adaptive',
    planLengthAdaptive: $('plan-length').value === 'adaptive-extend',
    // Kept alongside the axes above (not only derivable from them) so a
    // project this file writes still opens correctly in a build from
    // before the axes existed.
    scheduleMode: scheduleModeFromAxes(axes),
  });
}
for (const id of [
  'train-mode', 'train-steps', 'train-effort', 'train-batch', 'train-lr', 'sample-toggle',
  'sample-every', 'opening-rate', 'metrics-every', 'show-training-window', 'training-window-chars',
  'scheduler-mode', 'warmup-strategy', 'stable-phase', 'cooldown-shape', 'decay-start', 'plan-length',
]) {
  $(id).addEventListener('change', () =>
    withNotice('Saving setting', 'Setting saved', persistTrainingPlanSettings));
}

/// Every field the Settings tab's Inference panel owns beyond the two
/// device selects (inference-device/training-device already have their
/// own persistence): sampling, seed, and the length-mode/max-tokens pair.
async function persistInferenceOptions() {
  await db.putInferenceOptions({
    temperature: Number($('opt-temperature').value),
    topK: Number($('opt-top-k').value),
    topP: Number($('opt-top-p').value),
    minP: Number($('opt-min-p').value),
    repetitionPenalty: Number($('opt-repetition').value),
    seed: Number($('opt-seed').value) || 0,
    lengthMode: $('opt-length-mode').value,
    maxTokens: Number($('opt-max-tokens').value) || 0,
  });
}
for (const id of [
  'opt-temperature', 'opt-top-k', 'opt-top-p', 'opt-min-p', 'opt-repetition', 'opt-seed',
  'opt-length-mode', 'opt-max-tokens',
]) {
  $(id).addEventListener('change', () =>
    withNotice('Saving setting', 'Setting saved', persistInferenceOptions));
}

/// Every setting a training run reads off the Training-Settings and
/// Inference tabs, read fresh off the controls in one place — used both
/// to start a run (the Train button) and to push a change into one
/// already going (`pushLiveTrainingSettings`), so the two can never
/// drift into two different ideas of what a run's settings are. Model
/// shape (layers/hidden/heads/context/window) is deliberately not here:
/// it only ever takes effect on a freshly built model, there is no live
/// version of it.
function readTrainingSettings() {
  const manualMode = $('train-mode').value === 'manual';
  return {
    batchSize: chosenBatchSize(),
    maxSteps: Number($('train-steps').value) || 0,
    effort: chosenEffort(),
    // Manual mode uses the typed rate as-is. Auto mode picks its own,
    // judged against this model's *live* token count in the worker
    // (AUTO_LR_FROM_SCRATCH/AUTO_LR_TRAINED in worker.js) rather than
    // here: `peakLearningRate` below is simply unused in that case.
    // This used to be decided here, off `model.pretrained` — a fact
    // about whether the weights came from a checkpoint, which becomes
    // true the moment this model is ever reloaded from storage and
    // stays true forever after, regardless of how little of it had
    // actually trained. That silently capped Auto mode at the small
    // fine-tuning rate for any model that had survived a single page
    // reload.
    peakLearningRate: manualMode ? Number($('train-lr').value) : 0,
    autoLearningRate: !manualMode,
    scheduleMode: scheduleModeFromAxes({
      stablePhase: $('stable-phase').value,
      cooldownShape: $('cooldown-shape').value,
    }),
    warmupVariance: $('warmup-strategy').value === 'variance',
    decayStartAdaptive: $('decay-start').value === 'adaptive',
    planLengthAdaptive: $('plan-length').value === 'adaptive-extend',
    boundarySampleRate: Number($('opening-rate').value) / 100,
    autosaveFrequencySteps: Math.max(1, Number($('autosave-frequency').value) || 1000),
    metricsEvery: Number($('metrics-every').value) || 0,
    // 0 turns sampling off; anything else is a step interval.
    sampleEvery: $('sample-toggle').checked ? Number($('sample-every').value) : 0,
    // Training samples are generated with the Inference tab's own
    // prompt, length and sampling settings, not a second hidden set —
    // the same fields Generate reads. An empty prompt falls back to
    // that field's own placeholder example rather than a second,
    // separately hardcoded one.
    samplePrompt: $('prompt-input').value.trim() || $('prompt-input').placeholder,
    sampleMaxTokens: $('opt-length-mode').value === 'limit' ? Number($('opt-max-tokens').value) : 0,
    sampling: {
      temperature: Number($('opt-temperature').value),
      topK: Number($('opt-top-k').value),
      topP: Number($('opt-top-p').value),
      minP: Number($('opt-min-p').value),
      repetitionPenalty: Number($('opt-repetition').value),
      coreLoops: Number($('opt-core-loops').value) || 0,
    },
  };
}

/// Pushed to a run already in flight the moment any of these change —
/// batch size, learning rate, planned steps, effort, source-opening
/// windows, autosave/metrics cadence, and the sample prompt/length/
/// sampling settings. A run started at step 0 should not have to be
/// stopped and restarted just to pick up a value changed at step 3,400.
/// A no-op before any model exists — the worker just keeps the values
/// for whenever Train is next pressed.
function pushLiveTrainingSettings() {
  trainCall('update-training-settings', readTrainingSettings()).catch(() => {});
}
for (const id of [
  'train-mode', 'train-steps', 'train-effort', 'train-batch', 'train-lr', 'opening-rate',
  'sample-toggle', 'sample-every', 'metrics-every', 'autosave-frequency', 'prompt-input',
  'scheduler-mode', 'warmup-strategy', 'stable-phase', 'cooldown-shape', 'decay-start', 'plan-length',
  'opt-temperature', 'opt-top-k', 'opt-top-p', 'opt-min-p', 'opt-repetition', 'opt-length-mode',
  'opt-max-tokens',
]) {
  $(id).addEventListener('change', pushLiveTrainingSettings);
}

onStream('bench-progress', ({ stage, dispatchesPerSubmit }) => {
  if (!benchmarking || stage !== 'chunk') return;
  $('machine-profile-text').textContent =
    `Measuring this machine — ${dispatchesPerSubmit} dispatches per command buffer is ` +
    'fastest so far; now finding the batch size…';
});

/// Whether the first training run on a new shape auto-measures the
/// machine. On by default; loaded from settings at startup.
let benchmarkAutoEnabled = true;

$('benchmark-enabled').addEventListener('change', async (event) => {
  await withNotice('Saving setting', 'Setting saved', async () => {
    const wasEnabled = benchmarkAutoEnabled;
    benchmarkAutoEnabled = event.target.value !== 'off';
    await db.putBenchmarkConfig({ autoEnabled: benchmarkAutoEnabled });
    // Turning it off and back on is the only way to clear a bad stored
    // profile now that there is no dedicated button for it: the toggle
    // itself is the escape hatch.
    if (benchmarkAutoEnabled && !wasEnabled && gpuReport) {
      await db.deleteMachineProfile(gpuReport);
      machineProfile = null;
      notice('Machine profile cleared — the next training run measures it again.', 'info');
    }
    renderMachineProfile();
    updateGuidance();
  });
});

/// Loaded from Settings at startup, and kept live from there.
let inferenceDevicePref = 'gpu';

/// 'gpu' | 'remote' — which backend `trainingBackend` currently points
/// at (the CPU option is disabled in the markup: training has no CPU
/// path). Loaded from Settings at startup; see the `#training-device`
/// change handler below for what happens when it changes live.
let trainingBackendPref = 'gpu';

$('training-device').addEventListener('change', async (event) => {
  await withNotice('Saving training backend', 'Training backend saved', () =>
    applyTrainingBackendPref(event.target.value === 'remote' ? 'remote' : 'gpu', { persist: true }),
  );
});

async function currentRemoteServerConfig() {
  return {
    url: $('remote-server-url').value.trim(),
    token: $('remote-server-token').value,
  };
}

/// Swaps `trainingBackend` to match `pref` ('gpu' or 'remote'), building
/// a fresh `RemoteBackend` from whatever Server URL/token are currently
/// in Settings when switching to remote, and replaying every stream
/// handler this file has registered onto it — see `onStream` above.
/// Refuses while a run is in flight, the same guard every other
/// training-affecting Settings change would need.
async function applyTrainingBackendPref(pref, { persist } = {}) {
  if (persist && training) {
    $('training-device').value = trainingBackendPref;
    throw new Error('a training run is going — press Stop first');
  }
  trainingBackendPref = pref;
  $('remote-server-settings').hidden = pref !== 'remote';
  if (pref === 'remote') {
    const { url, token } = await currentRemoteServerConfig();
    trainingBackend = new RemoteBackend({ baseUrl: url, token, onFatalError: (error) => showError(error) });
    replayStreamRegistrations(trainingBackend);
  } else {
    trainingBackend = localBackend;
  }
  if (persist) {
    await db.putDevicePreference({ trainingDevice: pref, inferenceDevice: inferenceDevicePref });
  }
}

$('remote-server-url').addEventListener('change', async (event) => {
  const url = event.target.value.trim();
  event.target.value = url;
  await withNotice('Saving remote server setting', 'Remote server setting saved', async () => {
    if (trainingBackendPref === 'remote' && training) {
      throw new Error('a remote training run is going — press Stop first');
    }
    const { token } = await currentRemoteServerConfig();
    await db.putRemoteServerConfig({ url, token });
    if (trainingBackendPref === 'remote') await applyTrainingBackendPref('remote');
  });
});

$('remote-server-token').addEventListener('change', async (event) => {
  await withNotice('Saving remote server setting', 'Remote server setting saved', async () => {
    if (trainingBackendPref === 'remote' && training) {
      throw new Error('a remote training run is going — press Stop first');
    }
    const { url } = await currentRemoteServerConfig();
    await db.putRemoteServerConfig({ url, token: event.target.value });
    if (trainingBackendPref === 'remote') await applyTrainingBackendPref('remote');
  });
});

$('remote-test-btn').addEventListener('click', async () => {
  $('remote-test-btn').disabled = true;
  $('remote-server-status').textContent = 'Testing…';
  try {
    const { url, token } = await currentRemoteServerConfig();
    const probe = new RemoteBackend({ baseUrl: url, token });
    const info = await probe.call('health');
    $('remote-server-status').textContent =
      `Connected — ${info.adapter} (${info.backend}${info.isSoftware ? ', software' : ''}).`;
    notice('Remote server reachable.', 'success');
  } catch (error) {
    $('remote-server-status').textContent = `Not connected — ${(error && error.message) || error}.`;
    showError(error);
  } finally {
    $('remote-test-btn').disabled = false;
  }
});

$('inference-device').addEventListener('change', async (event) => {
  inferenceDevicePref = event.target.value === 'cpu' ? 'cpu' : 'gpu';
  await withNotice('Saving inference device', 'Inference device saved', async () => {
    await db.putDevicePreference({ trainingDevice: trainingBackendPref, inferenceDevice: inferenceDevicePref });
    await call('set-inference-device', { device: inferenceDevicePref });
  });
});

$('train-btn').addEventListener('click', async () => {
  if (training) return;
  clearError();
  if ($('train-mode').value === 'manual' && !(Number($('train-lr').value) > 0)) {
    notice('Set a learning rate before training in Manual mode.', 'error');
    return;
  }
  notice('Training…', 'info');
  training = true;
  lastPhaseKey = null;
  $('live-controls').hidden = false;
  $('train-btn').disabled = true;
  $('import-input').disabled = true;
  $('train-stop-btn').hidden = false;
  $('train-stop-btn').disabled = false;
  $('train-status').hidden = false;
  $('loss-chart').hidden = false;
  $('train-stats').textContent = 'Starting…';

  let runningRemotely = false;
  try {
    // One button, two jobs. With no model, make one first — nobody
    // should have to know that "create an untrained model" is a separate
    // step from "train it", because it never isn't.
    if (!model) {
      $('train-stats').textContent = 'Making a new model…';
      renderModel(
        await call('create-model', {
          layers: Number($('cfg-layers').value),
          ...currentLayerSharingFields(),
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
      // No maxVocabSize here: no UI control owns that ceiling, so the
      // worker asks the wasm side for it instead of this call
      // re-declaring the same number a third time.
      const learned = await call('learn-vocabulary', {}, [], 0);
      if (learned && learned.model) renderModel(learned.model);
    }

    // Measure the machine once, before the first run on it. The
    // settings that follow are read off that measurement, so it has to
    // happen first — and it only ever happens once per adapter and
    // model shape, because the answer is stored.
    if (benchmarkAutoEnabled && (!machineProfile || !profileShapeMatches(machineProfile))) {
      $('train-stats').textContent =
        'Measuring this machine — timing a few steps to pick the settings…';
      await runBenchmark().catch((error) => {
        // A benchmark that fails is not a reason not to train: the
        // fallbacks are one sequence per batch and the default command
        // buffer, which are the safe values.
        console.warn('[scriptonait] the machine benchmark failed', error);
      });
    }

    if (trainingBackendPref === 'remote') {
      await startRemoteTraining(readTrainingSettings());
      runningRemotely = true;
      $('train-stats').textContent = 'Started on the remote GPU…';
      return;
    }

    activeTrainingCall = call('train', readTrainingSettings(), [], 0);
    const result = await activeTrainingCall;

    if (result.stopReason === 'already-training') {
      showError('A training run is already going. Press Stop first if you want to change it.');
    } else if (result.stopReason === 'no-gpu') {
      showError(
        'Training runs on your GPU, and this browser did not give the page one. ' +
          'Try a browser with WebGPU (Chrome or Edge 113+, Safari 18+), or enable it in ' +
          "your browser's flags.",
      );
    } else if (result.stopReason === 'no-data') {
      showError('Not enough text to train on. Add more in Corpus.');
    } else {
      setProgress('train-progress-bar', 1);
      const loss = typeof result.loss === 'number' ? result.loss.toFixed(3) : '—';
      $('train-stats').textContent =
        `${result.steps} steps in ${formatDuration(result.elapsedSeconds)} · loss ${loss}` +
        ' · press Train again to continue';
      notice(`Training finished — ${result.steps} steps, loss ${loss}.`, 'success');
    }
    if (result.model) renderModel(result.model);
    await saveModel();
    training = false;
    await refreshPlan();
  } catch (error) {
    showError(error);
  } finally {
    // A remote run that has genuinely started keeps `training` true and
    // the Stop button live — this run isn't over, it just isn't
    // something this click handler waits on. `onStream('train-stopped',
    // ...)` below does this same reset once it actually ends.
    if (!runningRemotely) {
      training = false;
      $('train-btn').disabled = false;
      $('train-stop-btn').hidden = true;
      $('live-controls').hidden = true;
      setTitleProgress(null);
      updateGuidance();
    }
    activeTrainingCall = null;
  }
});

$('train-batch').addEventListener('input', updateGuidance);
// Every field that changes the price, priced as it is typed.
for (const id of ['cfg-layers', 'cfg-hidden', 'cfg-heads', 'cfg-kv-heads', 'cfg-context',
  'cfg-window', 'cfg-unique-layers', 'cfg-prelude-layers', 'cfg-coda-layers',
  'cfg-core-loop-min', 'cfg-core-loop-max']) {
  $(id).addEventListener('input', () => refreshShapeEstimate());
}
$('cfg-layer-sharing').addEventListener('change', () => {
  const mode = $('cfg-layer-sharing').value;
  const layers = Number($('cfg-layers').value) || 1;
  for (const [m, fields] of Object.entries(LAYER_SHARING_FIELDS)) {
    for (const [suffix] of fields) $(`cfg-${suffix}-field`).hidden = mode !== m;
  }
  if (mode === 'grouped') {
    $('cfg-unique-layers').value = defaultUniqueLayers(layers);
  } else if (mode === 'recurrent') {
    const defaults = defaultRecurrentCoreFields(layers);
    $('cfg-prelude-layers').value = defaults.preludeLayers;
    $('cfg-coda-layers').value = defaults.codaLayers;
    $('cfg-core-loop-min').value = defaults.coreLoopMin;
    $('cfg-core-loop-max').value = defaults.coreLoopMax;
  }
  refreshShapeEstimate();
});

$('train-stop-btn').addEventListener('click', () => {
  $('train-stop-btn').disabled = true;
  $('train-stats').textContent = 'Stopping after this step…';
  trainCall('stop', {}, [], 0).catch(() => {});
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
  ctx.font = '11px system-ui, sans-serif';

  // A top band for the live numbers and the legend, so the step/loss/
  // held-out/tokens-per-second line lives on the chart itself instead
  // of as a separate paragraph competing for space above it.
  const topBand = 28;
  if (chartStats) {
    ctx.fillStyle = '#c0caf5';
    ctx.fillText(chartStats, 4, 12);
  }
  // Right-aligned by measured width rather than fixed offsets, so the
  // three pieces get a real gap between them instead of guessing at
  // how wide "· same windows" renders and overlapping when it's wider
  // than guessed.
  const legend = [
    { text: 'held-out', colour: '#e0af68' },
    { text: '· same windows', colour: '#7aa2f7' },
    { text: 'training', colour: '#7aa2f7' },
  ];
  let legendX = width - 4;
  for (const { text, colour } of legend) {
    legendX -= ctx.measureText(text).width;
    ctx.fillStyle = colour;
    ctx.fillText(text, legendX, topBand - 4);
    legendX -= 8;
  }

  if (lossHistory.length < 2) return;

  const plotTop = topBand;
  const points = lossHistory
    .map((p) => p.loss)
    .concat(validationHistory.map((p) => p.loss))
    .concat(probeHistory.map((p) => p.loss));
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const plotSpan = height - plotTop - 12;
  const yFor = (loss) => height - ((loss - min) / span) * plotSpan - 6;

  // By step number, not array position: a rebuilt-from-history point
  // (one every `metricsEvery` steps) and a live progress tick (roughly
  // one per step) sit at wildly different densities in the same array,
  // and index-based x-positions stretched whichever portion was denser
  // across most of the chart's width regardless of how many actual
  // steps it covered.
  const steps = lossHistory.map((p) => p.step);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const stepSpan = maxStep - minStep || 1;
  const xFor = (step) => ((step - minStep) / stepSpan) * width;

  ctx.strokeStyle = '#7aa2f7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((point, i) => {
    const x = xFor(point.step);
    const y = yFor(point.loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  /// Both fixed-set curves, positioned by the step they were measured
  /// at so they line up in time with the curve above rather than by
  /// index.
  const drawMeasured = (series, colour, dashed) => {
    if (series.length < 2) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.beginPath();
    series.forEach((point, i) => {
      const x = xFor(point.at);
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
  ctx.fillText(max.toFixed(3), 4, plotTop + 12);
  ctx.fillText(min.toFixed(3), 4, height - 4);
}

// --- Saving and loading models ----------------------------------------

/// Clears the model, corpus and history — everything Import Project
/// replaces, just replaced with nothing instead of an imported project.
/// Settings (auto-save, device, benchmark, training-plan) are left as
/// they are: those are how you like the page to behave, not part of any
/// one project.
///
/// Reloads the page rather than trying to reset the worker's in-memory
/// model and corpus from here: there's no "unload" call today, and a
/// fresh page load already does exactly this — starts with nothing,
/// waits — for free.
$('new-project-btn').addEventListener('click', async () => {
  if (!confirm('Start a new project? This clears the current model, corpus and history.')) {
    return;
  }
  await withNotice('Starting new project', 'New project ready', async () => {
    // Asked for before any await, same reason Save/Export Project do —
    // the browser only honors showSaveFilePicker while it can still trace
    // the call back to this click. Cancelling or a browser without the
    // API isn't fatal to starting the project; it just starts without an
    // auto-save file set yet, same as it always could — but the *old*
    // project's target must not survive into it: without this, a
    // cancelled picker left the previous project's file/folder connected,
    // and the new, blank project's first autosave silently overwrote it.
    if (autosaveSupported()) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: autosaveFileName || 'scriptonait.snp',
          types: [{ description: 'scriptonait project', accept: { 'application/octet-stream': ['.snp'] } }],
        });
        await establishProjectFile(handle);
      } catch (error) {
        if (error && error.name !== 'AbortError') {
          console.warn('[scriptonait] could not set the new project\'s autosave file', error);
        }
        await clearAutosaveTarget();
      }
    } else {
      await clearAutosaveTarget();
    }
    await db.replaceAllSources([]);
    await db.replaceAllHistory([]);
    await db.clearModels();
    window.location.reload();
  });
});

$('export-btn').addEventListener('click', async () => {
  // The picker has to be asked for before any await — export-checkpoint
  // is a real wait (a full GPU-to-CPU weight readback), and the browser
  // only honors showSaveFilePicker while it can still trace the call
  // back to this click. Ask first, while that's still true; do the slow
  // work after, into the handle already granted.
  let handle = null;
  if (typeof window.showSaveFilePicker === 'function') {
    try {
      handle = await window.showSaveFilePicker({
        suggestedName: 'scriptonait.ckpt',
        types: [{ description: 'scriptonait checkpoint', accept: { 'application/octet-stream': ['.ckpt'] } }],
      });
    } catch (error) {
      if (error && error.name === 'AbortError') return;
      showError(error);
      return;
    }
  }
  await withNotice('Exporting checkpoint', 'Exported checkpoint', async () => {
    const { bytes } = await call('export-checkpoint');
    const blob = new Blob([bytes], { type: 'application/octet-stream' });
    if (handle) {
      const writable = await handle.createWritable();
      try {
        await writable.write(blob);
      } finally {
        await writable.close();
      }
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'scriptonait.ckpt';
    link.click();
    URL.revokeObjectURL(url);
  });
});

$('import-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (training) {
    showError('a training run is in flight — press Stop, then import a checkpoint');
    event.target.value = '';
    return;
  }
  clearError();
  notice(`Loading ${file.name}…`, 'info');
  try {
    const buffer = await file.arrayBuffer();
    renderModel(await call('import-checkpoint', { bytes: buffer }, [buffer]));
    await syncAllSources();
    notice(`Loaded ${file.name}.`, 'success');
    } catch (error) {
    showError(`that file didn't load: ${error.message}`);
    setModelStatus('absent', 'No model loaded.');
  }
  event.target.value = '';
});

/// Gather the current model + corpus + settings into the same project
/// blob format Export Project, Branch, and the Library all build from —
/// one function so the three can't quietly drift into disagreeing about
/// what a snapshot contains. Waits for persistLater's fire-and-forget
/// save chain first, so a source added moments ago is never missed.
async function snapshotProjectBlob() {
  await persistChain;
  const checkpointBytes = model ? (await call('export-checkpoint')).bytes : null;
  const optimizerBytes = model
    ? await call('export-optimizer').then((r) => r.bytes).catch(() => null)
    : null;
  const exportedSources = await db.listSources();
  const exportedHistory = await db.listHistory();
  const blob = project.buildProjectFile({
    checkpointBytes,
    optimizerBytes,
    sources: exportedSources,
    history: exportedHistory,
    settings: {
      autosaveConfig: await db.getAutosaveConfig(),
      devicePreference: await db.getDevicePreference(),
      benchmarkConfig: await db.getBenchmarkConfig(),
      trainingPlan: await db.getTrainingPlanSettings(),
      remoteServerConfig: await db.getRemoteServerConfig(),
      inferenceOptions: await db.getInferenceOptions(),
    },
  });
  console.info(
    `[scriptonait] project snapshot: ${exportedSources.length} source(s), ` +
      `${exportedHistory.length} history row(s), ` +
      `checkpoint ${checkpointBytes ? `${checkpointBytes.byteLength} bytes` : 'none (no model)'}, ` +
      `${blob.size} bytes total`,
  );
  return blob;
}

/// Branch and Save to Library both want a *quiescent* snapshot — one
/// that isn't racing the last few steps of a run still writing to the
/// checkpoint — not just whatever `snapshotProjectBlob` reads right now
/// the way Export Project is content to. Stops training if it's running
/// and waits for the run to actually end before taking the snapshot;
/// returns `wasTraining` so the caller can resume it afterward, the same
/// way a second Train press would.
async function stopAndSnapshotProject() {
  const wasTraining = training;
  if (wasTraining) {
    trainCall('stop', {}, [], 0).catch(() => {});
    await activeTrainingCall.catch(() => {});
  }
  return { blob: await snapshotProjectBlob(), wasTraining };
}

$('export-project-btn').addEventListener('click', async () => {
  clearError();
  // Asked for immediately, before any await: the browser only honors
  // showSaveFilePicker while the call can still be traced back to this
  // click, and everything below (a full checkpoint export among it) is
  // a real wait that would otherwise burn through that window before
  // the picker was ever shown — exactly what "must be handling a user
  // gesture" was reporting.
  let handle = null;
  if (typeof window.showSaveFilePicker === 'function') {
    try {
      handle = await window.showSaveFilePicker({
        // Whatever the project's file is already called, so saving
        // again offers the same name back instead of the generic
        // default — this is one of the three places (with New Project
        // and "Choose…") that can set it.
        suggestedName: autosaveFileName || 'scriptonait.snp',
        types: [{ description: 'scriptonait project', accept: { 'application/octet-stream': ['.snp'] } }],
      });
      // Saving a project this way establishes it as the file auto-save
      // continues writing into — the whole point of asking "what is the
      // project's file" once instead of separately in Settings.
      await establishProjectFile(handle);
    } catch (error) {
      // A cancelled picker is not an error.
      if (error && error.name === 'AbortError') return;
      showError(error);
      return;
    }
  }
  await withNotice('Exporting project', 'Exported project', async () => {
    const blob = await snapshotProjectBlob();
    if (handle) {
      const writable = await handle.createWritable();
      try {
        await writable.write(blob);
      } finally {
        await writable.close();
      }
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'scriptonait.snp';
    link.click();
    URL.revokeObjectURL(url);
  });
});

/// One click for what was, until now, four manual steps: stop, export a
/// copy under a name that says what step and schedule it's from, and
/// keep training the original unaffected. Unlike Export Project, this
/// deliberately never calls establishProjectFile — a branch's file must
/// never become the running project's own auto-save target.
$('branch-btn').addEventListener('click', async () => {
  clearError();
  if (!model) {
    showError('No model to branch yet.');
    return;
  }
  if (trainingBackendPref === 'remote') {
    showError('Branching a remote training run is not supported yet.');
    return;
  }
  const scheduleMode = scheduleModeFromAxes({
    stablePhase: $('stable-phase').value,
    cooldownShape: $('cooldown-shape').value,
  });
  // Asked for immediately, before any await — same reason Export Project
  // does: the browser only honors showSaveFilePicker while it can still
  // trace the call back to this click, and the stop-and-wait below is a
  // real wait that would otherwise burn through that window.
  let handle = null;
  if (typeof window.showSaveFilePicker === 'function') {
    try {
      handle = await window.showSaveFilePicker({
        suggestedName: branchFileName(model.step, scheduleMode),
        types: [{ description: 'scriptonait project', accept: { 'application/octet-stream': ['.snp'] } }],
      });
    } catch (error) {
      if (error && error.name === 'AbortError') return;
      showError(error);
      return;
    }
  }
  $('branch-btn').disabled = true;
  try {
    await withNotice('Branching project', 'Branched project', async () => {
      const { blob, wasTraining } = await stopAndSnapshotProject();
      if (handle) {
        const writable = await handle.createWritable();
        try {
          await writable.write(blob);
        } finally {
          await writable.close();
        }
      } else {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = branchFileName(model.step, scheduleMode);
        link.click();
        URL.revokeObjectURL(url);
      }
      // Resume the original run exactly as a second Train press would —
      // by this point activeTrainingCall has settled, so the Train
      // handler's own state (training/button/live-controls) is already
      // back to "stopped," and pressing it again runs the same, already-
      // exercised "a model exists, go straight to training" path.
      if (wasTraining) $('train-btn').dispatchEvent(new Event('click'));
      return wasTraining ? 'Branched — continuing the original run.' : 'Branched.';
    });
  } finally {
    $('branch-btn').disabled = false;
  }
});

/// Replace the live model + corpus + history + settings with what's in
/// `buffer` — a whole project's bytes, whichever store they came from (an
/// imported .snp file, or a Switch from the Library). The one place both
/// agree on what "loading a project" means, so they can't quietly drift
/// apart on it. `label` names the source for the status line and the
/// console log. Returns whether it worked.
async function applyProjectBuffer(buffer, label) {
  try {
    const { header, checkpointBytes, optimizerBytes } = project.parseProjectFile(buffer);
    console.info(
      `[scriptonait] loading ${label} (${buffer.byteLength} bytes): ` +
        `${(header.sources || []).length} source(s), ${(header.history || []).length} history row(s), ` +
        `checkpoint ${checkpointBytes ? `${checkpointBytes.byteLength} bytes` : 'none'}, ` +
        `optimizer ${optimizerBytes ? `${optimizerBytes.byteLength} bytes` : 'none'}`,
    );

    // Same reason New Project clears it (see that handler's own comment):
    // the file/folder handle lives in this browser, not in the project
    // file, so it survives a load untouched unless told otherwise. Left
    // alone, applyLoadedSettings below would find the *previous* project's
    // still-permitted handle, reconnect autosave to it, and the next
    // autosave would silently overwrite that other project's file with
    // this one's content. Cleared before the loaded settings are written
    // so its persisted fileName (display-only) isn't lost by the same call.
    await clearAutosaveTarget();
    await db.replaceAllSources(header.sources || []);
    await db.replaceAllHistory(header.history || []);
    const settings = header.settings || {};
    if (settings.autosaveConfig) await db.putAutosaveConfig(settings.autosaveConfig);
    if (settings.devicePreference) await db.putDevicePreference(settings.devicePreference);
    if (settings.benchmarkConfig) await db.putBenchmarkConfig(settings.benchmarkConfig);
    if (settings.trainingPlan) await db.putTrainingPlanSettings(settings.trainingPlan);
    if (settings.remoteServerConfig) await db.putRemoteServerConfig(settings.remoteServerConfig);
    if (settings.inferenceOptions) await db.putInferenceOptions(settings.inferenceOptions);

    // Before the corpus gets rebuilt, not after: syncAllSources (called
    // below, both directly and inside restoreModel-shaped paths) reads
    // #opening-rate straight off the DOM to push the boundary-sample
    // rate to the freshly created corpus. Restoring settings afterward
    // meant that read still saw whatever was on screen before the
    // load — so a project's own opening-window rate (and every other
    // field this restores) never actually reached the rebuilt corpus,
    // even though it was sitting correctly in IndexedDB and the field
    // itself updated a moment later.
    await applyLoadedSettings();

    if (checkpointBytes) {
      notice('Restoring model…', 'info');
      renderModel(await call('import-checkpoint', { bytes: checkpointBytes }, [checkpointBytes]));
      if (optimizerBytes) {
        await call('import-optimizer', { bytes: optimizerBytes }, [optimizerBytes]).catch(() => {});
      }
    } else {
      // No checkpoint in this project: the old model belongs to the
      // project just replaced, not this one — leaving it in IndexedDB
      // would resume an auto-save mode's rotating snapshots from a
      // project that's gone.
      await db.clearModels();
      model = null;
      setModelStatus('absent', 'No model yet.');
      $('model-details').replaceChildren();
    }

    // A full replace, not a merge: refreshSources only ever adds sources
    // it doesn't already know about, so the in-memory list has to be
    // cleared first or a source dropped by the load would linger.
    // syncAllSources hands each one to the model and restores its
    // persisted sample count (syncSource does that per source already).
    sources = [];
    history.length = 0;
    notice('Restoring corpus…', 'info');
    await refreshSources();
    await syncAllSources();
    notice('Restoring history…', 'info');
    history.push(...(await db.listHistory()));
    renderHistory();
    rebuildChartFromHistory();
    updateGuidance();
    notice(`Loaded ${label}.`, 'success');
    return true;
  } catch (error) {
    showError(`${label} didn't load: ${(error && error.message) || error}`);
    setModelStatus('absent', 'No model loaded.');
    return false;
  }
}

$('import-project-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (!confirm(`Import "${file.name}"? This replaces the current model, corpus, history and settings.`)) {
    event.target.value = '';
    return;
  }
  clearError();
  // The same staged progress the page's own startup restore shows —
  // several real seconds of work (parsing the file, then the model,
  // corpus and history each round-tripping through the worker and
  // IndexedDB) with nothing on screen in between otherwise.
  notice(`Reading ${file.name}…`, 'info');
  const buffer = await file.arrayBuffer();
  await applyProjectBuffer(buffer, file.name);
  event.target.value = '';
});

// --- The model library ---------------------------------------------------
//
// Named snapshots of a whole project (see db.js's own header on the
// record shape) — what turns Branch's "fork a copy" into several models
// a click apart instead of a file picker apart: train one specialist
// corpus, Branch it, keep training; Save to Library, switch corpora,
// train the next; Switch back to any of them for Generate at any time.

/// A default name for the next Save: the project's own base name (an
/// Export/Import/New Project's file, or the model's shape if none has
/// ever been chosen) plus this model's step, so saving progress on the
/// same project more than once doesn't produce indistinguishable entries
/// by default.
function libraryDefaultName() {
  if (!model) return '';
  const stepPart = String(Math.max(0, Math.round(model.step))).padStart(10, '0');
  return `${autosaveBaseName()}-step${stepPart}`;
}

async function renderLibrary() {
  const entries = await db.listLibrary();
  const list = $('library-list');
  if (entries.length === 0) {
    list.innerHTML = '<p class="empty-hint">Nothing saved yet.</p>';
    return;
  }
  list.innerHTML = entries
    .map(
      (entry) => `
    <div class="source-item" data-id="${entry.id}">
      <div class="meta">
        <span class="title">${escapeHtml(entry.name)}</span>
        <span class="stats">${formatCount(entry.params)} params · step ${entry.step.toLocaleString()} · ${new Date(entry.savedAt).toLocaleString()}</span>
      </div>
      <div class="actions">
        <button type="button" class="secondary switch-library" data-id="${entry.id}">Switch</button>
        <button type="button" class="secondary delete-library" data-id="${entry.id}">Delete</button>
      </div>
    </div>`,
    )
    .join('');
}

$('library-save-btn').addEventListener('click', async () => {
  if (!model) {
    showError('No model to save yet.');
    return;
  }
  const name = $('library-name').value.trim() || libraryDefaultName();
  clearError();
  $('library-save-btn').disabled = true;
  try {
    await withNotice('Saving to library', 'Saved to library', async () => {
      const { blob, wasTraining } = await stopAndSnapshotProject();
      await db.putLibraryEntry({
        id: newId(),
        name,
        step: model.step,
        params: model.params,
        savedAt: Date.now(),
        blob,
      });
      // Resume the original run exactly as a second Train press would —
      // same reason Branch does this.
      if (wasTraining) $('train-btn').dispatchEvent(new Event('click'));
      return wasTraining ? 'Saved — continuing the original run.' : 'Saved to library.';
    });
  } finally {
    $('library-save-btn').disabled = false;
  }
  $('library-name').value = '';
  await renderLibrary();
});

// One listener on the container instead of one per row — same reason
// sources-list does it this way.
$('library-list').addEventListener('click', async (event) => {
  const switchBtn = event.target.closest('.switch-library');
  const deleteBtn = event.target.closest('.delete-library');
  if (switchBtn) {
    const entry = await db.getLibraryEntry(switchBtn.dataset.id);
    if (!entry) {
      showError('That library entry is gone.');
      await renderLibrary();
      return;
    }
    if (!confirm(`Switch to "${entry.name}"? This replaces the current model, corpus, history and settings.`)) {
      return;
    }
    clearError();
    notice(`Loading ${entry.name}…`, 'info');
    const buffer = await entry.blob.arrayBuffer();
    await applyProjectBuffer(buffer, entry.name);
    await renderLibrary();
  } else if (deleteBtn) {
    const entry = await db.getLibraryEntry(deleteBtn.dataset.id);
    if (!entry) {
      await renderLibrary();
      return;
    }
    if (!confirm(`Delete "${entry.name}" from the library? This can't be undone.`)) return;
    await withNotice('Removing from library', 'Removed from library', () => db.deleteLibraryEntry(entry.id));
    await renderLibrary();
  }
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
        'It is still loaded — use Save on Overview to keep it.',
    );
  }
}

// --- Auto-save ---------------------------------------------------------
//
// A model is hours of somebody's GPU. Until now the only thing that
// saved it was the end of a run, so a crash — or a closed laptop, or a
// browser deciding to reclaim a background tab — took all of it.
//
// Two layers, because they fail differently. The browser copy is
// automatic and costs nothing to keep, but lives in storage a browser
// may clear and a page may exhaust. A file on disk survives everything,
// and needs the File System Access API, which not every browser has.
// Where it is missing the browser copy still runs and the page says so
// rather than pretending.

/// Loaded from Settings at startup, and kept live from there — every
/// field writes through to db.js immediately on change.
let autosaveEnabled = true;
let autosaveFrequencySteps = 1000;
let autosaveMode = 'overwrite';

/// The file the run writes itself to, once somebody has chosen one. The
/// handle itself is restored across reloads from IndexedDB when the
/// browser still grants it (see applyLoadedSettings) — it can't travel
/// in the project file, since it only ever means anything on the
/// machine and origin that granted it. The name can, and does: kept
/// here so "Choose…" can offer it back as the suggested name (picking
/// up where a project's own auto-save name left off) even before
/// there's a live handle to match it to.
let autosaveHandle = null;
/// Add mode's target: a granted folder, written into instead of
/// overwriting a single file. Independent of `autosaveHandle` so
/// switching modes back and forth doesn't make either forget the grant
/// it was given.
let autosaveDirHandle = null;
let autosaveFileName = null;
let lastAutosaveStep = 0;
let autosaveInFlight = false;
/// Set when `autosave()` is called while a save is already in flight —
/// a fast machine can reach the next scheduled save before the previous
/// one's file write has finished. Recorded rather than dropped, so the
/// in-flight save's `finally` can run exactly one catch-up save for
/// whatever the most recent call asked for; a second overlapping call
/// just replaces this, since only the latest state is worth catching up
/// to.
let autosavePending = null;

function autosaveSupported() {
  return typeof window.showSaveFilePicker === 'function';
}

function autosaveDirectorySupported() {
  return typeof window.showDirectoryPicker === 'function';
}

/// A name for the model that doesn't depend on it ever having been
/// saved anywhere: its own shape, in the same order those fields are
/// entered on the Overview tab (layers, hidden size, heads, key/value
/// heads, context, attention window) — e.g. 16_640_16_8_512_512. Two
/// models trained with different shapes get different default names
/// instead of colliding on the same generic one.
function modelShapeName() {
  if (!model) return null;
  let sharing = '';
  if (model.layerSharing === 'grouped') sharing = `_u${model.uniqueLayers}`;
  else if (model.layerSharing === 'recurrent') {
    sharing = `_p${model.preludeLayers}c${model.coreLoopMax}c${model.codaLayers}`;
  }
  return `${model.layers}${sharing}_${model.hidden}_${model.heads}_${model.kvHeads}_${model.contextLen}_${model.window}`;
}

/// The project's own base name: whatever file was last connected — an
/// Export, an Import, or New Project's own picker already gave the
/// project one — else the model's own shape, which is at least specific
/// to what's actually running rather than a generic placeholder. Used
/// where a name is wanted for the *project* (Branch's own filenames,
/// reflecting back a just-picked autosave file's name); see
/// autosaveTargetBaseName for what auto-save's own file should be
/// called, which is not always the same thing.
function autosaveBaseName() {
  return (autosaveFileName || modelShapeName() || 'scriptonait').replace(/\.snp$/i, '');
}

/// What auto-save specifically should call its own file: whatever has
/// actually already been chosen for it — typed into the Settings field,
/// or connected through "Choose file…"/"Choose folder…", including a
/// name inherited from an Export/Import/New Project that happened to
/// establish one too — used exactly as given, so reconnecting after a
/// lost permission still offers the same name back. Only before any of
/// that has ever happened does this make one up: the model's own shape,
/// with '_autosave' appended so this default is never mistaken for a
/// manually-named export.
function autosaveTargetBaseName() {
  // Stripped the same way autosaveBaseName is: every caller uses this as a
  // base to append its own suffix onto (-stepN.snp, the picker's own .snp),
  // and autosaveFileName already carries .snp from handle.name — without
  // stripping it here first, those calls doubled it (name.snp-stepN.snp,
  // name.snp.snp).
  return autosaveFileName
    ? autosaveFileName.replace(/\.snp$/i, '')
    : `${modelShapeName() || 'scriptonait'}_autosave`;
}

function autosaveStepFileName(step) {
  return `${autosaveTargetBaseName()}-step${String(Math.max(0, Math.round(step))).padStart(10, '0')}.snp`;
}

/// The suggested name for a Branch export — the step-suffixed pattern
/// above, plus the schedule mode, since that's the setting most likely
/// to be the reason a branch exists at all.
function branchFileName(step, scheduleMode) {
  const stepPart = String(Math.max(0, Math.round(step))).padStart(10, '0');
  return `${autosaveBaseName()}-branch-step${stepPart}-${scheduleMode}.snp`;
}

/// The whole project — model, corpus, history, settings — as one blob,
/// not just the trained weights: a file that is the only copy left
/// after a crash has to be enough on its own to get back to where
/// things were, not just the trained parameters with everything else
/// stranded in this browser's IndexedDB.
async function buildProjectBlob(checkpointBytes, optimizerBytes) {
  return project.buildProjectFile({
    checkpointBytes,
    optimizerBytes,
    sources,
    history,
    settings: {
      autosaveConfig: await db.getAutosaveConfig(),
      devicePreference: await db.getDevicePreference(),
      benchmarkConfig: await db.getBenchmarkConfig(),
      trainingPlan: await db.getTrainingPlanSettings(),
      remoteServerConfig: await db.getRemoteServerConfig(),
      inferenceOptions: await db.getInferenceOptions(),
    },
  });
}

async function writeBlobTo(handle, blob) {
  const writable = await handle.createWritable();
  try {
    await writable.write(blob);
  } finally {
    await writable.close();
  }
}

/// Overwrite mode: replace the one chosen file every time.
async function writeProjectToFile(checkpointBytes, optimizerBytes) {
  await writeBlobTo(autosaveHandle, await buildProjectBlob(checkpointBytes, optimizerBytes));
}

/// Add mode: one more step-suffixed file in the chosen folder every
/// time, never overwriting an earlier one. That's what makes it safe
/// against a fast machine reaching the next autosave before an earlier
/// write's promise has settled — each step gets its own filename, so
/// there is nothing for two in-flight writes to collide on even if
/// `autosaveInFlight`'s own guard (see `autosave`) somehow let that
/// happen.
async function writeProjectToNewFile(step, checkpointBytes, optimizerBytes) {
  const fileHandle = await autosaveDirHandle.getFileHandle(autosaveStepFileName(step), { create: true });
  await writeBlobTo(fileHandle, await buildProjectBlob(checkpointBytes, optimizerBytes));
}

/// Save without interrupting anything, on a step boundary.
///
/// Skipped rather than queued when one is already in flight: an export
/// pulls the weights off the GPU, and stacking two of those is the exact
/// thing that exhausted the heap and lost a model in the first place.
/// Export a checkpoint, retrying briefly if the GPU was busy with a
/// training step at the exact moment this asked for it.
///
/// The refusal itself is deliberate (see `acquire()` on the wasm side):
/// nothing should poll forever waiting for training to let go of the
/// device. But for an autosave — the thing crash-resilience depends on —
/// giving up after exactly one attempt turns "the model was mid-step"
/// into "this save never happened", and the collision is common at step
/// 0 specifically, the one step slow enough to matter (it pays for
/// allocating every GPU training buffer). A handful of short retries
/// covers that without ever blocking indefinitely.
async function exportCheckpointWithRetry() {
  const maxAttempts = 6;
  const delayMs = 700;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return (await call('export-checkpoint', {}, [], 0)).bytes;
    } catch (error) {
      const busy = error && typeof error.message === 'string' && error.message.includes('already busy');
      if (!busy || attempt === maxAttempts) throw error;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
}

async function autosave(step, { force = false, bytes: given = null } = {}) {
  if (!autosaveEnabled && !force) return;
  if (autosaveInFlight) {
    autosavePending = { step, force, given };
    return;
  }
  if (!force && !given && step - lastAutosaveStep < autosaveFrequencySteps) return;
  autosaveInFlight = true;
  lastAutosaveStep = step;
  try {
    await withNotice('Auto-saving', 'Auto-saved', async () => {
      const started = performance.now();
      // During a run the worker exports between slices and hands the
      // bytes over, because asking for them from here means asking while
      // a training step is running — and both take the same GPU guard.
      const bytes = given || (await exportCheckpointWithRetry());
      // Keep whatever optimizer state is already stored rather than
      // writing null over it.
      //
      // Exporting the moments here is not an option: they are twice the
      // size of the model, and pulling that across every thousand steps is
      // the memory pressure that lost a run in the first place. But
      // nulling them means this safety net quietly destroys the momentum
      // in the saved copy — so a crash-and-recover would resume with Adam
      // reset, which is most of what the saved copy was for.
      //
      // Slightly stale moments are fine. They are running averages of
      // gradient statistics and do not reference particular weights, so
      // moments from a thousand steps ago paired with current weights cost
      // almost nothing. Zero costs a visible bump and a few hundred steps.
      let optimizer = null;
      try {
        const stored = await db.getModel();
        optimizer = (stored && stored.optimizer) || null;
      } catch (error) {
        /* no stored copy yet: there is nothing to preserve */
      }
      const params = model ? model.params : 0;
      await db.putModel({ bytes, step, params, optimizer });
      let wroteFile = false;
      let fileDescription = '';
      if (autosaveMode === 'add' && autosaveDirHandle) {
        await writeProjectToNewFile(step, bytes, optimizer);
        wroteFile = true;
        fileDescription = `${autosaveStepFileName(step)} in ${autosaveDirHandle.name}`;
      } else if (autosaveMode !== 'add' && autosaveHandle) {
        await writeProjectToFile(bytes, optimizer);
        wroteFile = true;
        fileDescription = autosaveHandle.name;
      }
      console.info(
        `[scriptonait] auto-saved at step ${step.toLocaleString()} ` +
          `(${formatCount(bytes.byteLength)} bytes${wroteFile ? `, to ${fileDescription}` : ''}) in ` +
          `${(performance.now() - started).toFixed(0)} ms`,
      );
      $('autosave-status').textContent =
        `Last save: step ${step.toLocaleString()}` +
        (wroteFile ? ` — ${fileDescription} and this browser.` : ' — this browser.');
      flushSourceSampleCounts();
      return `Auto-saved at step ${step.toLocaleString()}`;
    });
  } catch (error) {
    // withNotice already posted the failure notice; this is just the
    // console trail. The status line keeps showing the last save that
    // actually succeeded — a second failure notice would bury that fact
    // the moment the next save works and the line moves on.
    console.warn('[scriptonait] auto-save failed', error);
  } finally {
    autosaveInFlight = false;
    // A fast machine can reach the next scheduled autosave before this
    // one's file write has finished — see the guard at the top of this
    // function. Rather than silently dropping every save that arrives
    // while one is in flight (which, on a machine where writes routinely
    // take longer than the interval between them, would mean autosave
    // never actually progresses past the first one), run exactly one
    // catch-up save for whatever the most recent overlapping call asked
    // for.
    if (autosavePending) {
      const pending = autosavePending;
      autosavePending = null;
      autosave(pending.step, { force: pending.force, bytes: pending.given });
    }
  }
}

// --- Crash resilience ---------------------------------------------------
//
// The step-interval autosave above only catches steps at its own
// cadence — a run stopped 400 steps after its last save loses those 400
// if the tab dies before the next one. That's exactly what happened
// switching to another site while training: the tab went to the
// background and the browser reclaimed it before the next interval hit.
// Saving whenever the tab is hidden or about to unload closes most of
// that window; it can't do anything for a hard OS-level kill (nothing
// downstream of that can, mid-write or not), but it turns "however many
// steps since the last thousand-step checkpoint" into "however many
// steps since you last switched away."

let lastCrashSaveAt = 0;

function saveBeforeTabDisappears() {
  const now = performance.now();
  // A tab can fire visibilitychange repeatedly in quick succession
  // (switching tabs back and forth); this isn't a save worth repeating
  // more than about once per ten seconds.
  if (now - lastCrashSaveAt < 10_000) return;
  // A run in flight already has its own crash-resilience: the worker
  // exports between slices and hands the bytes straight to autosave(),
  // no GPU contention involved. Asking for a checkpoint from here
  // instead means asking while training (or the model-creation/vocab-
  // learning/benchmark work that precedes its first step) is legitimately
  // holding the GPU — which is exactly what turning a tab away or
  // switching windows while a run is starting used to do: the export
  // retried against a busy GPU for several seconds and then failed with
  // an "already busy" notice, on a run that was never actually at risk.
  if (!autosaveEnabled || !model || autosaveInFlight || training) return;
  lastCrashSaveAt = now;
  // `force` only bypasses autosave()'s step-count throttle, not its
  // enabled check — already made above — so this still honors the
  // Settings-tab toggle.
  autosave(model.step, { force: true });
  flushSourceSampleCounts();
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') saveBeforeTabDisappears();
});
window.addEventListener('pagehide', saveBeforeTabDisappears);

$('autosave-file-btn').addEventListener('click', async () => {
  if (autosaveMode === 'add') {
    if (!autosaveDirectorySupported()) {
      notice(
        'This browser cannot write to a folder on its own (Chrome and Edge can). The browser ' +
          'copy still saves; "Save" on the Overview tab works anywhere.',
        'error',
      );
      return;
    }
    let handle;
    try {
      handle = await window.showDirectoryPicker({ mode: 'readwrite' });
    } catch (error) {
      // A cancelled picker is not an error, and not something to notify
      // about either.
      if (error && error.name !== 'AbortError') showError(error);
      return;
    }
    await withNotice('Choosing autosave folder', 'Autosave folder set', async () => {
      await establishAutosaveDirectory(handle);
      // Write immediately, so the folder gets its first file now rather
      // than in six hours when it matters.
      if (model) await autosave(model.step, { force: true });
      return `Autosave folder set to ${handle.name}`;
    });
    return;
  }
  if (!autosaveSupported()) {
    notice(
      'This browser cannot write to a file on its own (Chrome and Edge can). The browser ' +
        'copy still saves; "Save" on the Overview tab works anywhere.',
      'error',
    );
    return;
  }
  let handle;
  try {
    handle = await window.showSaveFilePicker({
      // Whatever the project's file is already called — from an
      // imported project's settings, an earlier Export/New Project, or
      // earlier this session — rather than always the generic default,
      // so reconnecting after an import or a lost permission is one
      // click to the right name. Nothing established yet suggests the
      // auto-save-specific default instead (project name, or the
      // model's own shape, plus '_autosave') rather than the plain
      // "scriptonait.snp" this picker used to fall back to.
      suggestedName: `${autosaveTargetBaseName()}.snp`,
      types: [{ description: 'scriptonait project', accept: { 'application/octet-stream': ['.snp'] } }],
    });
  } catch (error) {
    // A cancelled picker is not an error, and not something to notify
    // about either.
    if (error && error.name !== 'AbortError') showError(error);
    return;
  }
  await withNotice('Choosing autosave file', 'Autosave file set', async () => {
    await establishProjectFile(handle);
    // Write immediately, so the file exists and the permission is proven
    // now rather than in six hours when it matters.
    if (model) await autosave(model.step, { force: true });
    return `Autosave file set to ${handle.name}`;
  });
});

/// Wherever a project's file gets chosen — New Project, Export
/// Project's own picker, or this "Choose…" button — it becomes the
/// same thing: the file auto-save writes into from now on. One
/// function so all three agree on what "the project's file" means,
/// instead of each keeping its own idea of it and "scriptonait.snp"
/// resurfacing anywhere a name was already set.
async function establishProjectFile(handle) {
  autosaveHandle = handle;
  autosaveFileName = handle.name;
  $('autosave-filename').value = autosaveBaseName();
  $('autosave-status').textContent = `File: ${handle.name}.`;
  // The handle for this browser (can't travel — see putAutosaveFileHandle);
  // the name through persistAutosaveConfig, since that's the part that
  // does travel, in the project file.
  await db.putAutosaveFileHandle(handle).catch((error) => {
    console.warn('[scriptonait] could not remember the project file', error);
  });
  await persistAutosaveConfig().catch((error) => {
    console.warn('[scriptonait] could not remember the project file name', error);
  });
}

/// The directory-handle counterpart of `establishProjectFile`, for Add
/// mode — same idea (this is where the Settings "Choose…" button's
/// grant becomes "the folder auto-save writes into from now on"), a
/// separate function because a folder and a single file need different
/// live-handle state (`autosaveDirHandle` vs `autosaveHandle`) and a
/// different status line.
async function establishAutosaveDirectory(handle) {
  autosaveDirHandle = handle;
  $('autosave-status').textContent = `Folder: ${handle.name} (files named ${autosaveTargetBaseName()}-step<N>.snp).`;
  await db.putAutosaveDirectoryHandle(handle).catch((error) => {
    console.warn('[scriptonait] could not remember the auto-save folder', error);
  });
  await persistAutosaveConfig().catch((error) => {
    console.warn('[scriptonait] could not remember the auto-save settings', error);
  });
}

/// Forget whatever file or folder autosave was pointed at, in this
/// session and in storage. Used when a project that owned that target
/// is going away (New Project without a completed picker) — the point
/// is that nothing new can inherit it and overwrite what it points to.
async function clearAutosaveTarget() {
  autosaveHandle = null;
  autosaveDirHandle = null;
  autosaveFileName = null;
  $('autosave-filename').value = '';
  refreshAutosaveTargetDisplay();
  await db.putAutosaveFileHandle(null).catch(() => {});
  await db.putAutosaveDirectoryHandle(null).catch(() => {});
  await persistAutosaveConfig().catch(() => {});
}

/// The one place that writes autosaveConfig, so the file name never
/// gets silently dropped by a handler that only meant to change one of
/// the other three fields — putAutosaveConfig replaces the whole
/// record, it doesn't merge.
function persistAutosaveConfig() {
  return db.putAutosaveConfig({
    enabled: autosaveEnabled, frequencySteps: autosaveFrequencySteps, mode: autosaveMode,
    fileName: autosaveFileName,
  });
}

$('autosave-enabled').addEventListener('change', async (event) => {
  autosaveEnabled = event.target.value !== 'off';
  await withNotice('Saving setting', 'Setting saved', () => persistAutosaveConfig());
});

$('autosave-frequency').addEventListener('change', async (event) => {
  autosaveFrequencySteps = Math.max(1, Number(event.target.value) || 1000);
  await withNotice('Saving setting', 'Setting saved', () => persistAutosaveConfig());
});

/// Overwrite writes into a single file, Add into a chosen folder — two
/// different kinds of grant, so switching modes shows what's actually
/// set for the mode now selected rather than leaving the other mode's
/// status line on screen.
function refreshAutosaveTargetDisplay() {
  $('autosave-file-btn').textContent = autosaveMode === 'add' ? 'Choose folder…' : 'Choose file…';
  if (autosaveMode === 'add') {
    $('autosave-status').textContent = autosaveDirHandle
      ? `Folder: ${autosaveDirHandle.name} (files named ${autosaveTargetBaseName()}-step<N>.snp).`
      : 'Folder: not set.';
  } else {
    $('autosave-status').textContent = autosaveHandle ? `File: ${autosaveHandle.name}.` : 'File: not set.';
  }
}

$('autosave-mode').addEventListener('change', async (event) => {
  autosaveMode = event.target.value === 'add' ? 'add' : 'overwrite';
  refreshAutosaveTargetDisplay();
  await withNotice('Saving setting', 'Setting saved', () => persistAutosaveConfig());
});

$('autosave-filename').addEventListener('change', async (event) => {
  // Clearing the field back to empty means "go back to the default,"
  // not "call it literally scriptonait" — null keeps autosaveTargetBaseName's
  // own default live rather than freezing today's computed name in.
  autosaveFileName = event.target.value.trim() || null;
  event.target.value = autosaveTargetBaseName();
  refreshAutosaveTargetDisplay();
  await withNotice('Saving setting', 'Setting saved', () => persistAutosaveConfig());
});

/// Load the model saved by an earlier visit, if there is one.
async function restoreModel() {
  let stored = null;
  try {
    stored = await db.getModel();
  } catch (error) {
    console.warn('[scriptonait] could not read the saved model:', error);
    notice(`Could not read your saved model: ${(error && error.message) || error}.`, 'error');
    return;
  }
  if (!stored || !stored.bytes) return;
  try {
    renderModel(await call('load-model', { bytes: stored.bytes }, [], 0));
    await syncAllSources();
      if (stored.optimizer) {
      // Momentum, so training continues where it left off instead of
      // restarting Adam from nothing.
      try {
        await call('import-optimizer', { bytes: stored.optimizer }, [], 0);
        console.info('[scriptonait] restored the optimizer state with the model');
      } catch (error) {
        console.warn('[scriptonait] optimizer state did not fit this model:', error);
        notice('Momentum from your last session did not fit this model — resuming without it.', 'info');
      }
    }
  } catch (error) {
    console.warn('[scriptonait] the saved model would not load:', error);
    notice(`Your saved model would not load: ${(error && error.message) || error}.`, 'error');
    setModelStatus('absent', 'No model yet.');
  }
}

// --- Start -------------------------------------------------------------

/// Pull every Settings-tab and Training-tab record back from db.js and
/// apply it to both the module state and the control showing it — every
/// field involved writes through to db.js on change, so this is the one
/// place that has to do the reverse. Called at startup, and again after
/// a project import replaces what's stored (see the Import Project
/// handler) so the two paths can't drift apart.
async function applyLoadedSettings() {
  try {
    const autosaveConfig = await db.getAutosaveConfig();
    if (autosaveConfig) {
      autosaveEnabled = autosaveConfig.enabled !== false;
      autosaveFrequencySteps = autosaveConfig.frequencySteps > 0 ? autosaveConfig.frequencySteps : 1000;
      autosaveMode = autosaveConfig.mode === 'add' ? 'add' : 'overwrite';
      $('autosave-enabled').value = autosaveEnabled ? 'on' : 'off';
      $('autosave-frequency').value = String(autosaveFrequencySteps);
      $('autosave-mode').value = autosaveMode;
      // The name travels with the project even where the handle can't —
      // shown provisionally here so it's never blank after an import;
      // the handle checks right below override it with a confirmed
      // "connected" status when a live, permitted handle also exists.
      if (autosaveConfig.fileName) autosaveFileName = autosaveConfig.fileName;
      $('autosave-filename').value = autosaveTargetBaseName();
      $('autosave-status').textContent = autosaveMode === 'add'
        ? 'Folder: not set.'
        : autosaveFileName
          ? `File: ${autosaveFileName} (not connected — choose it to resume auto-save-to-file).`
          : 'File: not set.';
    }
    $('autosave-file-btn').textContent = autosaveMode === 'add' ? 'Choose folder…' : 'Choose file…';
  } catch (error) {
    console.warn('[scriptonait] could not read auto-save settings', error);
  }
  try {
    const handle = autosaveSupported() ? await db.getAutosaveFileHandle() : null;
    if (handle) {
      // queryPermission only checks, it never prompts — safe to call
      // without a click behind it, unlike requestPermission. A browser
      // that already granted readwrite here keeps it across reloads;
      // one that didn't gets told to pick the file again rather than
      // silently falling back to browser-only saves.
      const permission = await handle.queryPermission({ mode: 'readwrite' });
      autosaveFileName = handle.name;
      if (permission === 'granted') {
        autosaveHandle = handle;
        if (autosaveMode !== 'add') $('autosave-status').textContent = `File: ${handle.name}.`;
      } else if (autosaveMode !== 'add') {
        $('autosave-status').textContent = `File: ${handle.name} (permission needed — choose it again).`;
      }
    }
  } catch (error) {
    console.warn('[scriptonait] could not restore the auto-save file', error);
  }
  try {
    const dirHandle = autosaveDirectorySupported() ? await db.getAutosaveDirectoryHandle() : null;
    if (dirHandle) {
      const permission = await dirHandle.queryPermission({ mode: 'readwrite' });
      if (permission === 'granted') {
        autosaveDirHandle = dirHandle;
        if (autosaveMode === 'add') {
          $('autosave-status').textContent =
            `Folder: ${dirHandle.name} (files named ${autosaveTargetBaseName()}-step<N>.snp).`;
        }
      } else if (autosaveMode === 'add') {
        $('autosave-status').textContent = `Folder: ${dirHandle.name} (permission needed — choose it again).`;
      }
    }
  } catch (error) {
    console.warn('[scriptonait] could not restore the auto-save folder', error);
  }
  try {
    const devicePref = await db.getDevicePreference();
    if (devicePref) {
      inferenceDevicePref = devicePref.inferenceDevice === 'cpu' ? 'cpu' : 'gpu';
      $('inference-device').value = inferenceDevicePref;
      await call('set-inference-device', { device: inferenceDevicePref });
    }
  } catch (error) {
    console.warn('[scriptonait] could not read device settings', error);
  }
  try {
    const remoteConfig = await db.getRemoteServerConfig();
    $('remote-server-url').value = (remoteConfig && remoteConfig.url) || '';
    $('remote-server-token').value = (remoteConfig && remoteConfig.token) || '';
    const devicePref = await db.getDevicePreference();
    const pref = devicePref && devicePref.trainingDevice === 'remote' ? 'remote' : 'gpu';
    $('training-device').value = pref;
    await applyTrainingBackendPref(pref);
  } catch (error) {
    console.warn('[scriptonait] could not read the remote training backend settings', error);
  }
  try {
    const benchmarkConfig = await db.getBenchmarkConfig();
    if (benchmarkConfig) {
      benchmarkAutoEnabled = benchmarkConfig.autoEnabled !== false;
      $('benchmark-enabled').value = benchmarkAutoEnabled ? 'on' : 'off';
    }
  } catch (error) {
    console.warn('[scriptonait] could not read benchmark settings', error);
  }
  try {
    const planSettings = await db.getTrainingPlanSettings();
    if (planSettings) {
      $('train-mode').value = planSettings.mode === 'manual' ? 'manual' : 'auto';
      if (planSettings.plannedSteps > 0) $('train-steps').value = String(planSettings.plannedSteps);
      if (planSettings.effort) $('train-effort').value = planSettings.effort;
      if (planSettings.batchSize > 0) $('train-batch').value = String(planSettings.batchSize);
      if (planSettings.learningRate > 0) $('train-lr').value = String(planSettings.learningRate);
      $('sample-toggle').checked = !!planSettings.sampleToggle;
      if (planSettings.sampleEvery > 0) $('sample-every').value = String(planSettings.sampleEvery);
      if (typeof planSettings.boundarySampleRate === 'number') {
        $('opening-rate').value = String(Math.round(planSettings.boundarySampleRate * 100));
      }
      if (planSettings.metricsEvery > 0) $('metrics-every').value = String(planSettings.metricsEvery);
      if (typeof planSettings.showTrainingWindow === 'boolean') {
        $('show-training-window').value = planSettings.showTrainingWindow ? 'on' : 'off';
      }
      if (planSettings.trainingWindowChars > 0) {
        $('training-window-chars').value = String(planSettings.trainingWindowChars);
      }
      // The two axes directly, if this project was saved after they
      // existed; otherwise derived from the old scheduleMode string, so
      // nothing about the schedule resets on a project saved before them.
      const axes = planSettings.stablePhase
        ? { stablePhase: planSettings.stablePhase, cooldownShape: planSettings.cooldownShape }
        : axesFromScheduleMode(planSettings.scheduleMode);
      $('scheduler-mode').value = planSettings.schedulerMode === 'manual' ? 'manual' : 'auto';
      $('warmup-strategy').value = planSettings.warmupStrategy === 'variance' ? 'variance' : 'plan';
      $('stable-phase').value = axes.stablePhase;
      $('cooldown-shape').value = axes.cooldownShape;
      $('decay-start').value = planSettings.decayStartAdaptive ? 'adaptive' : 'fixed';
      $('plan-length').value = planSettings.planLengthAdaptive ? 'adaptive-extend' : 'fixed';
    }
  } catch (error) {
    console.warn('[scriptonait] could not read training-plan settings', error);
  }
  try {
    const inferenceOptions = await db.getInferenceOptions();
    if (inferenceOptions) {
      if (typeof inferenceOptions.temperature === 'number') {
        $('opt-temperature').value = String(inferenceOptions.temperature);
      }
      if (typeof inferenceOptions.topK === 'number') $('opt-top-k').value = String(inferenceOptions.topK);
      if (typeof inferenceOptions.topP === 'number') $('opt-top-p').value = String(inferenceOptions.topP);
      if (typeof inferenceOptions.minP === 'number') $('opt-min-p').value = String(inferenceOptions.minP);
      if (typeof inferenceOptions.repetitionPenalty === 'number') {
        $('opt-repetition').value = String(inferenceOptions.repetitionPenalty);
      }
      if (inferenceOptions.seed > 0) $('opt-seed').value = String(inferenceOptions.seed);
      if (inferenceOptions.lengthMode) $('opt-length-mode').value = inferenceOptions.lengthMode;
      if (inferenceOptions.maxTokens > 0) $('opt-max-tokens').value = String(inferenceOptions.maxTokens);
      applyLengthMode();
    }
  } catch (error) {
    console.warn('[scriptonait] could not read inference options', error);
  }
  // Unconditional: the Batch/Effort/Learning-rate disabled state has to
  // be set correctly on every load, not only when saved settings exist.
  applyTrainMode();
  applySchedulerMode();
}

(async function start() {
  // Loading a project restored from the last visit is several separate
  // waits on IndexedDB and the worker (sources, model, history) with
  // nothing to show for it in between — it reads as a hang rather than
  // as several seconds of real work. The one notification bar the page
  // already has for transient status carries it, one line per stage —
  // but only when there is actually something being restored. A first
  // visit with nothing saved yet has to stay silent about it; narrating
  // a restore that didn't happen is worse than saying nothing.
  const [hasModel, sourceCount, historyCount] = await Promise.all([
    db.getModel().then((m) => Boolean(m && m.bytes)).catch(() => false),
    db.listSources().then((rows) => (rows || []).length).catch(() => 0),
    db.listHistory().then((rows) => (rows || []).length).catch(() => 0),
  ]);
  const hasProject = hasModel || sourceCount > 0 || historyCount > 0;
  if (hasProject) notice('Restoring your project…', 'info');

  // Settings, before anything reads them.
  await applyLoadedSettings();

  // Guidance first, before anything that can be slow. Reading saved
  // sources waits on IndexedDB, and until it answers the page would
  // otherwise sit there saying nothing at all — which is the state this
  // whole line of text exists to prevent.
  renderSources();
  updateGuidance();
  renderLibrary();

  // Nothing is fetched from the network here. The page loads, shows what
  // you already have — including the model your last visit trained — and
  // waits. No model is downloaded, ever.
  if (sourceCount > 0) notice('Restoring corpus…', 'info');
  await refreshSources();
  setModelStatus('absent', 'No model yet.');
  updateGuidance();
  if (hasModel) notice('Restoring model…', 'info');
  await restoreModel();
  updateGuidance();
  // What to do next, before anything has been trained: with a model and
  // a corpus already restored from the last visit, the plan is readable
  // immediately and is the most useful thing on the page.
  await refreshPlan();
  // And with no model, what the default shape in the fields would cost.
  refreshShapeEstimate();

  // What every previous run measured. This is why the history is in
  // IndexedDB and not in a variable: the run worth understanding is
  // usually the one from yesterday.
  if (historyCount > 0) notice('Restoring history…', 'info');
  try {
    history.push(...(await db.listHistory()));
    renderHistory();
    rebuildChartFromHistory();
  } catch (error) {
    console.warn('[scriptonait] could not read the run history', error);
  }
  if (hasProject) notice('Project restored.', 'success');
})();
