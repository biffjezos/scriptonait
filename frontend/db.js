// IndexedDB storage for the sources you add.
//
// Every operation goes through `withStore`, which exists because of a
// specific bug: the previous version resolved a promise *to an object
// store* and then issued the request after awaiting it.
//
//     const store = await tx('sources', 'readwrite');   // don't
//     await wrapRequest(store.add(record));
//
// An IndexedDB transaction commits as soon as control returns to the
// event loop with no request outstanding. Awaiting in between sometimes
// leaves the transaction alive (if the microtask happens to drain inside
// the same task) and sometimes doesn't — so `add` would silently do
// nothing, or throw TransactionInactiveError, or hang the caller's
// promise forever. That is why adding thirty files stored some of them,
// displayed one, and needed a page reload to show the rest.
//
// So: open the database first, then create the transaction and issue the
// request in the same synchronous block, and resolve when the
// transaction completes.

const DB_NAME = 'scriptonait-llm';
// Version 3 added the history store: one row per measurement, kept so a
// run can be looked at after it happened rather than only while it is
// scrolling past.
const DB_VERSION = 3;
const SOURCES_STORE = 'sources';
const MODELS_STORE = 'models';
const SETTINGS_STORE = 'settings';
const HISTORY_STORE = 'history';

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(SOURCES_STORE)) {
        db.createObjectStore(SOURCES_STORE, { keyPath: 'id' });
      }
      // Holds the trained model between visits. A training run is hours
      // of the user's GPU; keeping it only in the tab meant a reload
      // threw all of it away, silently.
      if (!db.objectStoreNames.contains(MODELS_STORE)) {
        db.createObjectStore(MODELS_STORE, { keyPath: 'id' });
      }
      // What this machine measured about itself. Nothing here is a
      // preference; it is all the result of a timed run on this GPU.
      if (!db.objectStoreNames.contains(SETTINGS_STORE)) {
        db.createObjectStore(SETTINGS_STORE, { keyPath: 'id' });
      }
      // Every measurement a run takes, kept. A training run is hours of
      // numbers that scroll past once; without them the only record of
      // what a setting did is somebody's memory of a console line.
      if (!db.objectStoreNames.contains(HISTORY_STORE)) {
        const store = db.createObjectStore(HISTORY_STORE, { keyPath: 'id' });
        store.createIndex('runId', 'runId', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('could not open the database'));
    req.onblocked = () =>
      reject(new Error('the database is blocked by another tab — close it and reload'));
  });
  return dbPromise;
}

/// Run `work(store)` inside one transaction and resolve with whatever
/// the request it returns produced.
///
/// `work` must be synchronous and must issue its request immediately —
/// that's the whole point (see the header).
///
/// Resolution is on the request's own success, not on the transaction
/// commit. Committing is the browser's business and can lag; waiting for
/// it makes every caller wait on storage flush, and in at least one
/// environment (headless Chrome under virtual time) `oncomplete` never
/// arrives at all. A failed commit still rejects, via `onabort`/`onerror`
/// below — it just does so after the caller has moved on, which for a
/// list of pasted scripts is the right trade.
function withStore(storeName, mode, work) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        let transaction;
        try {
          transaction = db.transaction(storeName, mode);
        } catch (error) {
          reject(error);
          return;
        }
        const store = transaction.objectStore(storeName);
        let result;
        try {
          const request = work(store);
          if (request) {
            request.onsuccess = () => {
              result = request.result;
              resolve(result);
            };
            request.onerror = () => reject(request.error);
          }
        } catch (error) {
          reject(error);
          return;
        }
        // Only reached when `work` issued no request at all.
        transaction.oncomplete = () => resolve(result);
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () =>
          reject(transaction.error || new Error('the database transaction was aborted'));
      }),
  );
}

// --- Sources -----------------------------------------------------------
// A source: { id, title, kind: 'file'|'paste', rawText, sourceUrl, tags,
//             createdAt, updatedAt, timesSampled, windowProgress }
//
// `timesSampled` and `windowProgress` (`{epoch, cursor}`, how far this
// source has gotten through its own pass over its training windows — see
// worker.js's `window-progress` command) are periodically pulled from the
// wasm corpus and written back (see `updateSourceStats`); absent on a
// source added before they existed, which callers treat the same as 0 /
// no progress yet.
//
// `kind: 'url'` no longer gets created (URL fetch was removed), but a
// record written by an older version can still carry it, and `sourceUrl`
// stays in the shape so those records still load.
//
// Genre/tone tagging is gone: the model takes a real instruction (form,
// length, subject, what to echo) parsed from the prompt itself, which is
// what those tags were a crude stand-in for. `tags` stays in the record
// shape so databases written by the old version still load.

/// Store a record the caller already built, id and all.
///
/// The caller owns the id because the caller owns the list: the page
/// keeps its sources in memory and shows them immediately, and this is
/// only how they survive a reload. If this minted its own id the two
/// copies would disagree about what to remove.
export async function putSource(record) {
  await withStore(SOURCES_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function deleteSource(id) {
  await withStore(SOURCES_STORE, 'readwrite', (store) => store.delete(id));
}

export async function listSources() {
  const all = (await withStore(SOURCES_STORE, 'readonly', (store) => store.getAll())) || [];
  return all.sort((a, b) => a.createdAt - b.createdAt);
}

export async function getSource(id) {
  return withStore(SOURCES_STORE, 'readonly', (store) => store.get(id));
}

/// Replace every stored source with `list`, id and all — a project
/// import's job, not an edit to any one source. One request per record
/// (see the module header on why `withStore` only ever issues one).
export async function replaceAllSources(list) {
  const existing = await listSources();
  for (const record of existing) {
    await withStore(SOURCES_STORE, 'readwrite', (store) => store.delete(record.id));
  }
  for (const record of list) {
    await withStore(SOURCES_STORE, 'readwrite', (store) => store.put(record));
  }
}

/// How many training windows have been drawn from this source, and how
/// far it's gotten through its own pass over them (`windowProgress:
/// {epoch, cursor}`, see worker.js's `window-progress` command) — both as
/// of the last time they were pulled from the wasm corpus and written
/// back. Doesn't touch `updatedAt`, since this isn't an edit to the
/// source itself. Missing on a source added before these fields existed;
/// callers treat that the same as 0 / no progress yet.
export async function updateSourceStats(id, changes) {
  const existing = await getSource(id);
  if (!existing) return;
  await withStore(SOURCES_STORE, 'readwrite', (store) =>
    store.put({ ...existing, ...changes }));
}

// --- The trained model -------------------------------------------------
//
// One record, always under the same key: the checkpoint bytes plus what
// the page needs to describe it before loading. A checkpoint carries its
// own tokenizer and shape, so nothing else has to be stored beside it.

const CURRENT_MODEL = 'current';

export async function putModel({ bytes, step, params, optimizer = null }) {
  // The optimizer's moment buffers ride along with the weights: a model
  // restored without them resumes with Adam's momentum reset, and the
  // loss jumps at every visit.
  const record = { id: CURRENT_MODEL, bytes, step, params, optimizer, savedAt: Date.now() };
  await withStore(MODELS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getModel() {
  return withStore(MODELS_STORE, 'readonly', (store) => store.get(CURRENT_MODEL));
}

/// The stored model record. A New Project's job: leaving it behind means
/// a reload could restore a model from a project that's gone.
export async function clearModels() {
  await withStore(MODELS_STORE, 'readwrite', (store) => store.clear());
}

// --- The machine profile -----------------------------------------------
//
// One record per adapter the browser has handed this page. Keyed by the
// adapter's own name and backend rather than by a single 'machine' key,
// because the same profile is meaningless on a laptop that has both an
// integrated and a discrete GPU and gives the page whichever one it
// feels like.

function machineKey({ adapter, backend }) {
  return `machine:${backend || '?'}:${adapter || 'unknown'}`;
}

export async function putMachineProfile(profile) {
  const record = { ...profile, id: machineKey(profile), savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getMachineProfile(device) {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(machineKey(device)));
}

export async function deleteMachineProfile(device) {
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.delete(machineKey(device)));
}

// --- App settings --------------------------------------------------------
//
// Three fixed-id records in the same store the machine profile lives in.
// Fixed id, not per-adapter, because these are the user's own choices
// (whether to auto-save, which device to prefer) rather than something
// measured about the hardware.

const AUTOSAVE_CONFIG = 'autosave-config';
const DEVICE_PREFERENCE = 'device-preference';
const BENCHMARK_CONFIG = 'benchmark-config';
const TRAINING_PLAN_SETTINGS = 'training-plan-settings';
const AUTOSAVE_FILE_HANDLE = 'autosave-file-handle';
const REMOTE_SERVER_CONFIG = 'remote-server-config';
const INFERENCE_OPTIONS = 'inference-options';

/// The `FileSystemFileHandle` the "Choose…" picker returned — not the
/// name, the live handle, so autosave-to-file survives a reload without
/// asking again. IndexedDB structured-clones a handle just fine; nothing
/// else can carry it. Left out of the exported .snp for the same reason
/// it's browser-local to begin with: a handle from `showSaveFilePicker`
/// only ever means anything on the machine and origin that granted it —
/// there is no bytes-on-disk form of "write access to this exact file"
/// that a project opened somewhere else could use. What the project file
/// does carry is `autosaveConfig` (enabled/frequency/mode) below, which
/// is the part that travels.
export async function putAutosaveFileHandle(handle) {
  await withStore(SETTINGS_STORE, 'readwrite', (store) =>
    store.put({ id: AUTOSAVE_FILE_HANDLE, handle, savedAt: Date.now() }));
}

export async function getAutosaveFileHandle() {
  const record = await withStore(SETTINGS_STORE, 'readonly', (store) => store.get(AUTOSAVE_FILE_HANDLE));
  return (record && record.handle) || null;
}

const AUTOSAVE_DIR_HANDLE = 'autosave-dir-handle';

/// The `FileSystemDirectoryHandle` Add mode writes step-suffixed files
/// into — the directory counterpart of `putAutosaveFileHandle`, stored
/// under its own key so switching between Overwrite and Add doesn't
/// make either mode forget the handle it was given. Same rationale for
/// staying out of the exported .snp: a directory grant only ever means
/// anything on the machine and origin that granted it.
export async function putAutosaveDirectoryHandle(handle) {
  await withStore(SETTINGS_STORE, 'readwrite', (store) =>
    store.put({ id: AUTOSAVE_DIR_HANDLE, handle, savedAt: Date.now() }));
}

export async function getAutosaveDirectoryHandle() {
  const record = await withStore(SETTINGS_STORE, 'readonly', (store) => store.get(AUTOSAVE_DIR_HANDLE));
  return (record && record.handle) || null;
}

/// { enabled, frequencySteps, mode: 'overwrite'|'add', fileName }
/// `fileName` is just the string the picker returned last — the part of
/// "which file" that can travel in the project file, unlike the handle
/// itself (see putAutosaveFileHandle/putAutosaveDirectoryHandle).
export async function putAutosaveConfig(config) {
  const record = { ...config, id: AUTOSAVE_CONFIG, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getAutosaveConfig() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(AUTOSAVE_CONFIG));
}

/// { trainingDevice: 'gpu', inferenceDevice: 'gpu'|'cpu' }. trainingDevice
/// is always 'gpu' today (there is no CPU training path) but is stored
/// as its own key, not hardcoded into the shape, so a future
/// training-backend selector can use it without a rename.
export async function putDevicePreference(preference) {
  const record = { ...preference, id: DEVICE_PREFERENCE, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getDevicePreference() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(DEVICE_PREFERENCE));
}

/// { url, token }. The token travels through IndexedDB in plain text,
/// same as everything else in this store — this page has no server of
/// its own to keep a secret from, and the token only ever gets sent
/// onward to the one remote-training URL the user typed in themselves.
export async function putRemoteServerConfig(config) {
  const record = { ...config, id: REMOTE_SERVER_CONFIG, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getRemoteServerConfig() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(REMOTE_SERVER_CONFIG));
}

/// { temperature, topK, topP, minP, repetitionPenalty, seed, lengthMode,
/// maxTokens } — every field the Settings tab's Inference panel owns
/// beyond the two device selects (which already have their own
/// devicePreference record). Both Generate and a training run's own
/// in-training samples read these live off the DOM rather than this
/// store directly; this only exists so a reload puts the same values
/// back in the DOM instead of the markup's hardcoded defaults.
export async function putInferenceOptions(options) {
  const record = { ...options, id: INFERENCE_OPTIONS, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getInferenceOptions() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(INFERENCE_OPTIONS));
}

/// { autoEnabled }
export async function putBenchmarkConfig(config) {
  const record = { ...config, id: BENCHMARK_CONFIG, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getBenchmarkConfig() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(BENCHMARK_CONFIG));
}

/// { mode: 'auto'|'manual', plannedSteps, effort, batchSize, learningRate,
///   sampleToggle, sampleEvery, boundarySampleRate, metricsEvery,
///   showTrainingWindow, trainingWindowChars, schedulerMode: 'auto'|'manual',
///   warmupStrategy: 'plan'|'variance', stablePhase: 'flat'|'reactive-cuts',
///   cooldownShape: 'deferred'|'immediate', scheduleMode }
///
/// The Training tab's own settings, previously DOM-only: they reset to
/// the markup's hardcoded defaults on every reload, which is the gap
/// meant here — in particular `plannedSteps`, which since the schedule
/// rework is the project's planned length, not a per-run number, and is
/// worth even less lost on a reload than the others.
///
/// `boundarySampleRate` (0-1) is how often a sampled training window
/// starts exactly at a source's beginning rather than at the next window
/// due in its rotation — see `Corpus::boundary_sample_rate`. Missing on
/// a project saved before this setting existed, which callers treat as
/// the wasm side's own default.
///
/// `metricsEvery` is the step interval for measuring held-out loss and
/// recording a Metrics row — see worker.js's `VALIDATE_EVERY`. Missing
/// or 0 falls back to that default, the same as `boundarySampleRate`.
///
/// `schedulerMode`, `warmupStrategy`, `stablePhase` and `cooldownShape`
/// are the Settings tab's Scheduler panel — see app.js's
/// `scheduleModeFromAxes`/`axesFromScheduleMode`. `scheduleMode`
/// ('wsd'|'cosine-cuts'|'cosine') is kept alongside them, not only
/// derivable from them, purely for a project written by this build to
/// still open correctly in one from before the two axes existed; on a
/// project saved before either existed, callers derive the axes from
/// `scheduleMode` instead of resetting to a hardcoded default.
export async function putTrainingPlanSettings(settings) {
  const record = { ...settings, id: TRAINING_PLAN_SETTINGS, savedAt: Date.now() };
  await withStore(SETTINGS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getTrainingPlanSettings() {
  return withStore(SETTINGS_STORE, 'readonly', (store) => store.get(TRAINING_PLAN_SETTINGS));
}

// --- Run history -------------------------------------------------------
//
// One record per measurement, plus one per event worth remembering (a
// run starting, a learning-rate cut, a corpus change). Rows are keyed
// so that a plain `getAll` comes back in order: run id, then a
// zero-padded step, then a counter for records that share a step.
//
// This is the thing that makes a training run reviewable. Everything
// else on the page shows the present moment; a run is six hours long and
// the question is almost always "what did it do between then and now".

let historySequence = 0;

function historyKey(runId, step) {
  historySequence = (historySequence + 1) % 1000;
  const paddedStep = String(Math.max(0, Math.round(step))).padStart(10, '0');
  const paddedSeq = String(historySequence).padStart(3, '0');
  return `${runId}:${paddedStep}:${paddedSeq}`;
}

export async function appendHistory(record) {
  const stored = { ...record, id: historyKey(record.runId, record.step || 0) };
  await withStore(HISTORY_STORE, 'readwrite', (store) => store.put(stored));
  return stored;
}

/// Every record, oldest first. The store is keyed to sort this way, so
/// no sort is needed and none is done — a run of 50,000 steps writes a
/// few hundred rows, and reading them all is the point.
export async function listHistory() {
  return (await withStore(HISTORY_STORE, 'readonly', (store) => store.getAll())) || [];
}

export async function clearHistory() {
  await withStore(HISTORY_STORE, 'readwrite', (store) => store.clear());
}

/// Replace the whole run history with `rows` — a project import's job.
/// Rows keep the ids they were exported with, so ordering (run id, then
/// zero-padded step) survives the round trip untouched. A row exported
/// without one (a project file saved before the in-memory copy backfilled
/// it from appendHistory's return value) gets a fresh id here rather than
/// failing the whole import — the store's keyPath requires one on every
/// put, and a project file otherwise intact shouldn't be unrecoverable
/// over rows whose only fault is a missing id.
export async function replaceAllHistory(rows) {
  await clearHistory();
  for (const row of rows) {
    const stored = row.id ? row : { ...row, id: historyKey(row.runId || 'imported', row.step || 0) };
    await withStore(HISTORY_STORE, 'readwrite', (store) => store.put(stored));
  }
}
