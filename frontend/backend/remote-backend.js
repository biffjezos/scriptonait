import { Backend } from './backend.js';

/// Speaks to `llm-server` (a native binary the user runs on their own
/// GPU machine — see crates/llm-server) over plain HTTP + Server-Sent
/// Events, fulfilling the same `Backend` contract `LocalWasmBackend`
/// does so `app.js`'s training call sites don't need to know which one
/// they're talking to. Only training is ever routed here — inference
/// has no remote endpoint (see llm-server's own "non-goals": staying
/// local is what keeps a remote GPU box billed for training only).
///
/// Job-based, not per-step: `call('train', ...)` uploads a checkpoint
/// snapshot and the corpus once, the server runs its own training loop,
/// and progress comes back over one long-lived SSE connection — a
/// network round trip per step would be far slower than the in-process
/// dispatch this codebase already treats training overhead as a
/// first-order cost to minimize.
///
/// `EventSource` can't attach a bearer token to its request (no custom
/// headers), so the SSE stream is read by hand: a plain `fetch` whose
/// body is a `ReadableStream`, split on blank lines the way the
/// text/event-stream format itself delimits events.
export class RemoteBackend extends Backend {
  constructor({ baseUrl, token, onFatalError } = {}) {
    super();
    this.baseUrl = (baseUrl || '').replace(/\/+$/, '');
    this.token = token || '';
    this.onFatalError = onFatalError || (() => {});
    this.streamHandlers = new Map();
    this.sessionId = null;
    this.abortController = null;
    this.smoothedLoss = null;
    this.lastSyncStep = 0;
    this.syncEveryStep = 1000;
  }

  call(type, payload = {}) {
    switch (type) {
      case 'train':
        return this.startTraining(payload);
      case 'stop':
        return this.stopTraining();
      case 'health':
        return this.health();
      default:
        return Promise.reject(
          new Error(`"${type}" is not supported by a remote training backend`),
        );
    }
  }

  onStream(type, handler) {
    this.streamHandlers.set(type, handler);
  }

  emit(type, data) {
    const handler = this.streamHandlers.get(type);
    if (handler) handler({ type, ...data });
  }

  authHeaders() {
    return this.token ? { Authorization: `Bearer ${this.token}` } : {};
  }

  /// Without this, an empty Server URL turns `fetch(\`${this.baseUrl}${path}\`)`
  /// into a same-origin relative request — against the page's own site,
  /// not a server at all — which fails with a confusing "not valid
  /// JSON" error (GitHub Pages' own 404 page starts with `<!DOCTYPE`)
  /// instead of saying what's actually wrong.
  ensureConfigured() {
    if (!this.baseUrl) throw new Error('no remote server URL is set');
  }

  async postJson(path, body) {
    this.ensureConfigured();
    let res;
    try {
      res = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
        body: JSON.stringify(body),
      });
    } catch (error) {
      throw new Error(`could not reach the remote server: ${(error && error.message) || error}`);
    }
    return this.readJsonResponse(res);
  }

  async readJsonResponse(res) {
    const text = await res.text();
    let json;
    try {
      json = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(
        res.ok
          ? 'the remote server sent back something that was not JSON'
          : `remote server returned HTTP ${res.status}`,
      );
    }
    if (!res.ok) throw new Error(json.error || `remote server returned HTTP ${res.status}`);
    return json;
  }

  async health() {
    this.ensureConfigured();
    let res;
    try {
      res = await fetch(`${this.baseUrl}/health`, { headers: this.authHeaders() });
    } catch (error) {
      throw new Error(`could not reach the remote server: ${(error && error.message) || error}`);
    }
    return this.readJsonResponse(res);
  }

  /// `payload.checkpointBase64` is the local model's current checkpoint
  /// (see app.js's `startRemoteTraining` — it always exports one first,
  /// since a model is guaranteed to exist locally by the time Train is
  /// pressed) and `payload.sources` the corpus snapshot, both uploaded
  /// once here — see the module doc on why this is job-based.
  async startTraining(payload) {
    const { checkpointBase64, sources = [], batchSize, peakLearningRate, maxSteps, autosaveFrequencySteps } = payload;
    this.syncEveryStep = Math.max(1, Number(autosaveFrequencySteps) || 1000);
    this.lastSyncStep = 0;
    this.smoothedLoss = null;
    const created = await this.postJson('/session', {
      checkpoint_base64: checkpointBase64,
      sources,
    });
    this.sessionId = created.sessionId;
    try {
      await this.postJson(`/session/${this.sessionId}/train/start`, {
        batch_size: batchSize,
        peak_learning_rate: peakLearningRate,
        max_steps: maxSteps,
      });
    } catch (error) {
      await this.deleteSession(this.sessionId).catch(() => {});
      this.sessionId = null;
      throw error;
    }
    this.connectEvents(this.sessionId);
    return { started: true, sessionId: this.sessionId };
  }

  async stopTraining() {
    if (!this.sessionId) return { stopping: false };
    await this.postJson(`/session/${this.sessionId}/train/stop`, {});
    return { stopping: true };
  }

  async deleteSession(sessionId) {
    await fetch(`${this.baseUrl}/session/${sessionId}`, {
      method: 'DELETE',
      headers: this.authHeaders(),
    });
  }

  /// Pulls the session's current weights and hands them to app.js as a
  /// `remote-checkpoint` event — the bridge that keeps a local, usable
  /// copy of a remotely-trained model, at the same cadence the Settings
  /// tab's own Autosave Frequency already governs for local runs (no
  /// second, hidden copy of that setting).
  async syncCheckpoint(sessionId, step) {
    const res = await fetch(`${this.baseUrl}/session/${sessionId}/checkpoint`, {
      headers: this.authHeaders(),
    });
    if (!res.ok) throw new Error(`could not fetch the remote checkpoint (HTTP ${res.status})`);
    const bytes = await res.arrayBuffer();
    this.emit('remote-checkpoint', { step, bytes });
  }

  async connectEvents(sessionId) {
    this.abortController = new AbortController();
    try {
      const res = await fetch(`${this.baseUrl}/session/${sessionId}/train/events`, {
        headers: this.authHeaders(),
        signal: this.abortController.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`could not open the training-progress stream (HTTP ${res.status})`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buffer.indexOf('\n\n')) >= 0) {
          const rawEvent = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          await this.handleSseEvent(rawEvent, sessionId);
        }
      }
    } catch (error) {
      if (error && error.name !== 'AbortError') {
        this.onFatalError(`the remote training-progress stream failed: ${(error && error.message) || error}`);
      }
    }
  }

  async handleSseEvent(rawEvent, sessionId) {
    const dataLines = rawEvent
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''));
    if (dataLines.length === 0) return; // a ":missed some events" comment line
    let event;
    try {
      event = JSON.parse(dataLines.join('\n'));
    } catch {
      return;
    }
    if (event.type === 'train-progress') this.handleProgress(event, sessionId);
    else if (event.type === 'train-stopped') await this.handleStopped(event, sessionId);
  }

  /// Fills in the fields `app.js`'s existing `onStream('train-progress',
  /// ...)` handler reads that this server's smaller v1 loop doesn't
  /// compute (`smoothedLoss` — the same 0.9/0.1 EMA worker.js itself
  /// uses; `sources`, since the server doesn't report per-batch draws)
  /// so that handler runs unmodified whether the event came from the
  /// worker or from here.
  handleProgress(event, sessionId) {
    this.smoothedLoss = this.smoothedLoss === null ? event.loss : this.smoothedLoss * 0.9 + event.loss * 0.1;
    this.emit('train-progress', {
      step: event.step,
      loss: event.loss,
      smoothedLoss: this.smoothedLoss,
      lr: event.lr,
      gradNorm: event.gradNorm,
      elapsedSeconds: event.elapsedSeconds,
      tokensPerSecond: event.tokensPerSecond,
      fractionDone: event.fractionDone,
      sources: [],
    });
    if (event.step - this.lastSyncStep >= this.syncEveryStep) {
      this.lastSyncStep = event.step;
      this.syncCheckpoint(sessionId, event.step).catch((error) => {
        console.warn('[scriptonait] periodic remote checkpoint sync failed', error);
      });
    }
  }

  async handleStopped(event, sessionId) {
    try {
      await this.syncCheckpoint(sessionId, event.step);
    } catch (error) {
      console.warn('[scriptonait] could not pull the final remote checkpoint', error);
    }
    this.emit('train-stopped', { step: event.step, reason: event.reason });
    if (this.abortController) this.abortController.abort();
    await this.deleteSession(sessionId).catch(() => {});
    if (this.sessionId === sessionId) this.sessionId = null;
  }
}
