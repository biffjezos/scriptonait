// A whole-project file: model + corpus + history + settings in one
// downloadable .snp, alongside — not instead of — the single-file .ckpt
// export/import and the Settings-tab auto-save-to-file feature. Those
// stay exactly as they are; this is the "move to another machine, or
// take one explicit full backup" path neither of them covers.
//
// Custom container, no zip dependency — the same hand-rolled framing
// checkpoint.rs already uses, for the same reason: no build step, no new
// dependency for one file format.
//
//   bytes 0..4      magic "SNPJ"
//   bytes 4..8      u32 LE format version (1)
//   bytes 8..12     u32 LE JSON header length H
//   bytes 12..12+H  JSON header, UTF-8
//   bytes 12+H..    binary sections back to back (checkpoint, then
//                   optimizer — offsets in the header say which is where)
//
// Sources/history/settings are small enough as text to sit directly in
// the JSON header, the same way the existing "Copy as JSON" history
// feature already treats a whole run's history as one JSON blob; only
// the checkpoint and optimizer blobs — the only genuinely large parts —
// go in the binary tail.

const MAGIC = 'SNPJ';
const FORMAT_VERSION = 1;

/// `checkpointBytes`/`optimizerBytes` are ArrayBuffers, or null (no model
/// yet, or a model that has never been trained so there's no optimizer
/// state to carry). `sources`/`history` are the plain record arrays
/// db.js already stores; `settings` is
/// `{ autosaveConfig, devicePreference, benchmarkConfig, trainingPlan }`.
export function buildProjectFile({ checkpointBytes, optimizerBytes, sources, history, settings }) {
  const checkpointLen = checkpointBytes ? checkpointBytes.byteLength : 0;
  const optimizerLen = optimizerBytes ? optimizerBytes.byteLength : 0;
  const header = {
    version: FORMAT_VERSION,
    exportedAt: Date.now(),
    checkpoint: checkpointBytes ? { offset: 0, length: checkpointLen } : null,
    optimizer: optimizerBytes ? { offset: checkpointLen, length: optimizerLen } : null,
    sources: sources || [],
    history: history || [],
    settings: settings || {},
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(8);
  new DataView(prefix.buffer).setUint32(0, FORMAT_VERSION, true);
  new DataView(prefix.buffer).setUint32(4, headerBytes.length, true);

  const parts = [MAGIC, prefix, headerBytes];
  if (checkpointBytes) parts.push(checkpointBytes);
  if (optimizerBytes) parts.push(optimizerBytes);
  return new Blob(parts, { type: 'application/octet-stream' });
}

/// `bytes` is the ArrayBuffer of a whole .snp file. Returns
/// `{ header, checkpointBytes, optimizerBytes }` — the latter two are
/// standalone ArrayBuffers (copied out, not views into `bytes`), so each
/// can be handed to the worker as its own transferable independently of
/// the others.
export function parseProjectFile(bytes) {
  if (bytes.byteLength < 12) throw new Error('not a scriptonait project file (too short)');
  const magic = new TextDecoder().decode(new Uint8Array(bytes, 0, 4));
  if (magic !== MAGIC) throw new Error('not a scriptonait project file');
  const view = new DataView(bytes, 4, 8);
  const version = view.getUint32(0, true);
  if (version !== FORMAT_VERSION) {
    throw new Error(`project file format version ${version}, expected ${FORMAT_VERSION}`);
  }
  const headerLen = view.getUint32(4, true);
  const headerStart = 12;
  const headerBytes = new Uint8Array(bytes, headerStart, headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const body = headerStart + headerLen;
  const section = (info) =>
    info ? bytes.slice(body + info.offset, body + info.offset + info.length) : null;
  return {
    header,
    checkpointBytes: section(header.checkpoint),
    optimizerBytes: section(header.optimizer),
  };
}
