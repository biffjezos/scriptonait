/// The one shape every compute backend speaks: a `{type, ...}` request in,
/// a response or a stream of `{type, ...}` events out. `app.js` depends
/// only on this — never on a `Worker`, a wasm binding, or a network call
/// directly — so which backend actually runs a request (this browser's
/// own WASM+WebGPU today; a remote machine running the same code
/// natively, later) is a setting, not a rewrite.
///
/// `LocalWasmBackend` is the only implementation today, wrapping the
/// existing Worker/postMessage plumbing. A future `RemoteBackend` would
/// open a WebSocket to a remote `llm-server` process and speak the exact
/// same message shape this interface already describes.
export class Backend {
  /// Send one request, get one response. `transfer` names objects (an
  /// ArrayBuffer, say) to move rather than copy, exactly like
  /// `Worker.postMessage`'s own transfer list — a backend that has no
  /// such concept (a network call) is free to ignore it.
  async call(type, payload, transfer) {
    throw new Error('not implemented');
  }

  /// Register `handler` for every unsolicited `{type, ...}` event this
  /// backend delivers outside the request/response flow — training
  /// progress, log lines, GPU status. One handler per type; a second
  /// registration replaces the first, matching `call`'s single-flight
  /// shape.
  onStream(type, handler) {
    throw new Error('not implemented');
  }
}
