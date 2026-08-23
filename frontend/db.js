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
// Version 2 added the settings store, which holds the machine profile.
const DB_VERSION = 2;
const SOURCES_STORE = 'sources';
const MODELS_STORE = 'models';
const SETTINGS_STORE = 'settings';

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

function newId() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// --- Sources -----------------------------------------------------------
// A source: { id, title, kind: 'url'|'file'|'paste', rawText, sourceUrl,
//             tags, createdAt, updatedAt }
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

export async function addSource({ title, kind, rawText, sourceUrl = null, tags = {} }) {
  const now = Date.now();
  const record = { id: newId(), title, kind, rawText, sourceUrl, tags, createdAt: now, updatedAt: now };
  await withStore(SOURCES_STORE, 'readwrite', (store) => store.add(record));
  return record;
}

export async function updateSource(id, changes) {
  const existing = await getSource(id);
  if (!existing) throw new Error(`no source ${id}`);
  const updated = { ...existing, ...changes, updatedAt: Date.now() };
  await withStore(SOURCES_STORE, 'readwrite', (store) => store.put(updated));
  return updated;
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

// --- The trained model -------------------------------------------------
//
// One record, always under the same key: the checkpoint bytes plus what
// the page needs to describe it before loading. A checkpoint carries its
// own tokenizer and shape, so nothing else has to be stored beside it.

const CURRENT_MODEL = 'current';
/// The model with the lowest held-out loss the run has seen. Training
/// past its own best is normal - the best model of a run is rarely its
/// last - so the best one is kept separately and never overwritten by a
/// later, worse one.
const BEST_MODEL = 'best';

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

export async function deleteModel() {
  await withStore(MODELS_STORE, 'readwrite', (store) => store.delete(CURRENT_MODEL));
}

export async function putBestModel({ bytes, step, params, validationLoss }) {
  const record = {
    id: BEST_MODEL,
    bytes,
    step,
    params,
    validationLoss,
    savedAt: Date.now(),
  };
  await withStore(MODELS_STORE, 'readwrite', (store) => store.put(record));
  return record;
}

export async function getBestModel() {
  return withStore(MODELS_STORE, 'readonly', (store) => store.get(BEST_MODEL));
}

// --- The machine profile -----------------------------------------------
//
// One record per adapter the browser has handed this page. Keyed by the
// adapter's own name and backend rather than by a single 'machine' key,
// because the same profile is meaningless on a laptop that has both an
// integrated and a discrete GPU and gives the page whichever one it
// feels like.

export function machineKey({ adapter, backend }) {
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
