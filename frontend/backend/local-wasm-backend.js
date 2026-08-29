import { Backend } from './backend.js';

// Long jobs (generation, fine-tuning) legitimately run for minutes and
// opt out with `timeoutMs: 0`. Everything else gets a deadline, because
// a request that never settles is the worst possible failure: the UI
// waits forever, shows nothing, and says nothing. That is exactly what
// happened when the worker's own fetch hung — the page sat on "Loading
// the model..." and every subsequent action queued up behind it in
// silence.
const DEFAULT_TIMEOUT_MS = 60000;

/// This browser's own WASM+WebGPU, run in a Worker so a long CPU
/// generation or a slow GPU readback never blocks the page's own event
/// loop. Requests are promise-shaped; streaming updates arrive
/// out-of-band and are dispatched to whatever handler is currently
/// interested — the same `Backend` contract a remote implementation
/// would fulfil over a network instead of `postMessage`.
export class LocalWasmBackend extends Backend {
  /// `onFatalError`, if given, is called with a message when the worker
  /// itself fails to load or crashes outside any single request —
  /// something no pending `call()` promise can reject, since no request
  /// was necessarily in flight when it happened.
  constructor({ onFatalError } = {}) {
    super();
    this.onFatalError = onFatalError || (() => {});
    // Resolved against this module's own URL, not the page's — a plain
    // './worker.js' here would ask for frontend/backend/worker.js (this
    // file's own directory), which doesn't exist; the real file is one
    // level up at frontend/worker.js. That 404 broke every worker call
    // silently: `restoreModel()`'s startup `load-model` call carries
    // `timeoutMs: 0` (no timeout, see DEFAULT_TIMEOUT_MS's own comment),
    // so instead of erroring it just hung forever, and the page never
    // rendered a restored model at all.
    this.worker = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' });
    this.nextRequestId = 1;
    this.pending = new Map();
    this.streamHandlers = new Map();

    this.worker.onmessage = (event) => {
      const { type, rid: id } = event.data;
      if (type === 'result') {
        const entry = this.pending.get(id);
        if (entry) {
          this.pending.delete(id);
          entry.resolve(event.data.result);
        }
        return;
      }
      if (type === 'error') {
        const error = new Error(event.data.message);
        if (event.data.stack) error.stack = event.data.stack;
        const entry = this.pending.get(id);
        if (entry) {
          this.pending.delete(id);
          entry.reject(error);
        } else {
          this.onFatalError(error);
        }
        return;
      }
      const handler = this.streamHandlers.get(type);
      if (handler) handler(event.data);
    };

    this.worker.onerror = (event) =>
      this.onFatalError(
        `${event.message || 'the worker failed'} (${event.filename || 'worker'}:${event.lineno || '?'})`,
      );

    // A worker that fails to parse or throws outside a handler used to
    // be invisible: no reply, no error, every call timing out after a
    // minute. It reports itself now, and the page says so.
    this.worker.addEventListener('error', (event) => {
      const detail = (event && (event.message || String(event.error))) || 'unknown error';
      console.error('[scriptonait] worker failed to load or crashed:', detail, event);
      this.onFatalError(`the background worker failed: ${detail}`);
    });
  }

  call(type, payload = {}, transfer = [], timeoutMs = DEFAULT_TIMEOUT_MS) {
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      let timer = null;
      const settle = (fn) => (value) => {
        if (timer) clearTimeout(timer);
        this.pending.delete(id);
        fn(value);
      };
      this.pending.set(id, { resolve: settle(resolve), reject: settle(reject) });
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`the worker didn't answer "${type}" within ${Math.round(timeoutMs / 1000)}s`));
        }, timeoutMs);
      }
      // The payload goes in its own field rather than being spread
      // alongside the request id. It used to be `{ id, type, ...payload }`,
      // and a payload carrying its own `id` — every upsert-source does —
      // overwrote the request id with it. The worker then took that as
      // the request id, so the source id vanished ("the source id was
      // missing"), and the reply came back under an id no caller
      // recognised, so that call never settled. One key collision,
      // three symptoms.
      this.worker.postMessage({ rid: id, type, payload }, transfer);
    });
  }

  onStream(type, handler) {
    this.streamHandlers.set(type, handler);
  }
}
