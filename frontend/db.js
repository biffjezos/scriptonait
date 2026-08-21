// IndexedDB wrapper for training sources and saved model checkpoints.
// Everything here is promise-based; IndexedDB's native API is
// callback/event-based, so each function wraps exactly one request.

const DB_NAME = 'scriptonait-llm';
const DB_VERSION = 1;
const SOURCES_STORE = 'sources';
const MODELS_STORE = 'models';

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
      if (!db.objectStoreNames.contains(MODELS_STORE)) {
        db.createObjectStore(MODELS_STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(storeName, mode) {
  return openDb().then((db) => db.transaction(storeName, mode).objectStore(storeName));
}

function wrapRequest(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function newId() {
  return (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);
}

// --- Sources -----------------------------------------------------------
// A source: { id, title, kind: 'url'|'file'|'paste', rawText, sourceUrl,
//             tags: { genre, tone }, createdAt, updatedAt }
//
// `tags` are plain metadata the frontend prepends as a short "[GENRE:
// x] [TONE: y]" preamble when feeding this source's text to the model
// (see app.js's buildTaggedText) — llm-core itself has no concept of
// tags, they're just ordinary text from its point of view.

export async function addSource({ title, kind, rawText, sourceUrl = null, tags = {} }) {
  const store = await tx(SOURCES_STORE, 'readwrite');
  const now = Date.now();
  const record = { id: newId(), title, kind, rawText, sourceUrl, tags, createdAt: now, updatedAt: now };
  await wrapRequest(store.add(record));
  return record;
}

export async function updateSource(id, changes) {
  const store = await tx(SOURCES_STORE, 'readwrite');
  const existing = await wrapRequest(store.get(id));
  if (!existing) throw new Error(`source not found: ${id}`);
  const updated = { ...existing, ...changes, id, updatedAt: Date.now() };
  await wrapRequest(store.put(updated));
  return updated;
}

export async function deleteSource(id) {
  const store = await tx(SOURCES_STORE, 'readwrite');
  await wrapRequest(store.delete(id));
}

export async function listSources() {
  const store = await tx(SOURCES_STORE, 'readonly');
  const all = await wrapRequest(store.getAll());
  return all.sort((a, b) => a.createdAt - b.createdAt);
}

export async function getSource(id) {
  const store = await tx(SOURCES_STORE, 'readonly');
  return wrapRequest(store.get(id));
}

// --- Model checkpoints ---------------------------------------------------
// A model: { id, name, config: {numLayers,hiddenDim,numHeads,contextLen,
//            localWindow}, weightBytes: ArrayBuffer, step, createdAt }

export async function saveModel({ name, config, weightBytes, step }) {
  const store = await tx(MODELS_STORE, 'readwrite');
  const record = { id: newId(), name, config, weightBytes, step, createdAt: Date.now() };
  await wrapRequest(store.add(record));
  return record;
}

export async function listModels() {
  const store = await tx(MODELS_STORE, 'readonly');
  const all = await wrapRequest(store.getAll());
  return all.sort((a, b) => b.createdAt - a.createdAt);
}

export async function getModel(id) {
  const store = await tx(MODELS_STORE, 'readonly');
  return wrapRequest(store.get(id));
}

export async function deleteModel(id) {
  const store = await tx(MODELS_STORE, 'readwrite');
  await wrapRequest(store.delete(id));
}
