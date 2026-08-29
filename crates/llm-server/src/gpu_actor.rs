//! The one thread that ever touches the GPU.
//!
//! `GpuContext` carries `Cell`/`RefCell` fields (`dispatch_count`, the
//! params pool) inherited from its browser-tab origin, where nothing
//! needed to be thread-safe because JS is single-threaded. Sharing it
//! across axum's multi-threaded tokio runtime the way the wasm side
//! shares it across async callbacks on one JS thread would be unsound —
//! so instead of trying to make `GpuContext` `Sync`, this crate never
//! shares it at all: one dedicated OS thread owns it, every request that
//! needs the GPU sends a `Command` and awaits a reply, and the actor
//! processes exactly one thing at a time. That is the same one-thing-
//! at-a-time rule `wasm-app`'s own `busy` guard enforces client-side,
//! just enforced structurally here instead of by a flag anyone could
//! forget to check.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Instant;

use tokio::sync::{broadcast, oneshot};

use llm_core::checkpoint::{Checkpoint, WeightDtype};
use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::model::ModelWeights;
use llm_core::rng::Rng;
use llm_core::tokenizer::Tokenizer;
use llm_core::train::TrainConfig;
use llm_gpu::{GpuContext, GpuTrainer};

use crate::protocol::TrainEvent;

pub type SessionId = String;

/// How many buffered events a training run's broadcast channel holds. A
/// slow or momentarily-disconnected SSE client that falls this far
/// behind just misses the oldest ones and picks up from whatever is
/// current — a dropped progress tick is not something worth stalling
/// the actual training loop over.
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// Where a training run's event sender lives so a `GET .../train/events`
/// request — a separate HTTP call from the one that started the run —
/// can find it and subscribe. Shared directly between the GPU thread
/// and axum's async handlers rather than routed through `Command`,
/// since subscribing needs no GPU involvement at all: any clone of a
/// `broadcast::Sender` can mint new receivers on its own.
type EventRegistry = Arc<Mutex<HashMap<SessionId, broadcast::Sender<TrainEvent>>>>;

struct Session {
    config: ModelConfig,
    weights: ModelWeights,
    corpus: Corpus,
    rng: Rng,
    trainer: Option<GpuTrainer>,
    step: u64,
    tokens_seen: u64,
}

pub struct TrainSettings {
    pub batch_size: u32,
    pub peak_learning_rate: f32,
    pub max_steps: u32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub progress_interval_ms: u64,
}

pub struct HealthInfo {
    pub adapter: String,
    pub backend: String,
    pub device_type: String,
    pub is_software: bool,
}

pub struct NewSession {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub corpus: Corpus,
    pub step: u64,
    pub tokens_seen: u64,
}

pub struct SessionCreated {
    pub session_id: SessionId,
    pub vocab_size: u32,
    pub params: f64,
}

enum Command {
    Health {
        reply: oneshot::Sender<HealthInfo>,
    },
    CreateSession {
        session: NewSession,
        reply: oneshot::Sender<Result<SessionCreated, String>>,
    },
    StartTraining {
        session_id: SessionId,
        settings: TrainSettings,
        /// Sent back once the run has either started for real or failed
        /// to — by the time this arrives, a `GET .../train/events`
        /// subscription (via the registry above) is already able to
        /// find the run's sender.
        reply: oneshot::Sender<Result<(), String>>,
    },
    StopTraining {
        session_id: SessionId,
    },
    GetCheckpoint {
        session_id: SessionId,
        reply: oneshot::Sender<Result<Vec<u8>, String>>,
    },
    DeleteSession {
        session_id: SessionId,
        reply: oneshot::Sender<Result<(), String>>,
    },
}

/// Answer a command that arrived while the actor was mid-training-loop
/// with the same "busy" refusal `wasm-app`'s own guard gives — the GPU
/// is doing exactly one thing, and that thing isn't this request.
fn reject_busy(cmd: Command) {
    let busy = "the GPU is already busy with another operation — wait for it to finish, \
                or stop the training run first"
        .to_string();
    match cmd {
        Command::Health { reply } => {
            // No natural "busy" answer for a health check specifically;
            // dropping the reply just makes the caller's request fail
            // with a generic error, which is an acceptable rough edge —
            // Test Connection isn't a button anyone clicks *during* an
            // active run to begin with.
            drop(reply);
        }
        Command::CreateSession { reply, .. } => {
            let _ = reply.send(Err(busy));
        }
        Command::StartTraining { reply, .. } => {
            let _ = reply.send(Err(busy));
        }
        Command::StopTraining { .. } => {
            // Handled inline by the training loop itself before this
            // function is ever reached for it — see run_training below.
        }
        Command::GetCheckpoint { reply, .. } => {
            let _ = reply.send(Err(busy));
        }
        Command::DeleteSession { reply, .. } => {
            let _ = reply.send(Err(busy));
        }
    }
}

#[derive(Clone)]
pub struct GpuActorHandle {
    tx: std_mpsc::Sender<Command>,
    events: EventRegistry,
}

impl GpuActorHandle {
    async fn call<T>(&self, build: impl FnOnce(oneshot::Sender<T>) -> Command) -> Result<T, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(build(reply_tx))
            .map_err(|_| "the GPU thread has stopped running".to_string())?;
        reply_rx.await.map_err(|_| "the GPU thread dropped the request".to_string())
    }

    pub async fn health(&self) -> Result<HealthInfo, String> {
        self.call(|reply| Command::Health { reply }).await
    }

    pub async fn create_session(&self, session: NewSession) -> Result<SessionCreated, String> {
        self.call(|reply| Command::CreateSession { session, reply }).await?
    }

    pub async fn start_training(&self, session_id: SessionId, settings: TrainSettings) -> Result<(), String> {
        self.call(|reply| Command::StartTraining { session_id, settings, reply }).await?
    }

    pub fn stop_training(&self, session_id: SessionId) {
        let _ = self.tx.send(Command::StopTraining { session_id });
    }

    pub async fn get_checkpoint(&self, session_id: SessionId) -> Result<Vec<u8>, String> {
        self.call(|reply| Command::GetCheckpoint { session_id, reply }).await?
    }

    pub async fn delete_session(&self, session_id: SessionId) -> Result<(), String> {
        self.call(|reply| Command::DeleteSession { session_id, reply }).await?
    }

    /// `None` when nothing has ever started training on this session id
    /// — including "hasn't started yet"; a caller connecting the events
    /// stream before calling train/start sees this and should retry
    /// once it has.
    pub fn subscribe_train_events(&self, session_id: &str) -> Option<broadcast::Receiver<TrainEvent>> {
        self.events.lock().unwrap().get(session_id).map(|tx| tx.subscribe())
    }
}

/// Start the GPU thread and wait for it to actually open a device before
/// returning — a server with no GPU has nothing to serve, and finding
/// that out on the first request instead of at startup would be a worse
/// failure mode than refusing to start at all.
pub fn spawn() -> Result<GpuActorHandle, String> {
    let (tx, rx) = std_mpsc::channel::<Command>();
    let (ready_tx, ready_rx) = std_mpsc::channel::<Result<(), String>>();
    let events: EventRegistry = Arc::new(Mutex::new(HashMap::new()));
    let events_for_actor = Arc::clone(&events);
    thread::Builder::new()
        .name("gpu-actor".to_string())
        .spawn(move || {
            let ctx = match pollster::block_on(GpuContext::new()) {
                Ok(ctx) => ctx,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            run(ctx, rx, events_for_actor);
        })
        .map_err(|err| format!("could not start the GPU thread: {err}"))?;
    ready_rx.recv().map_err(|_| "the GPU thread exited before it was ready".to_string())??;
    Ok(GpuActorHandle { tx, events })
}

fn run(ctx: GpuContext, rx: std_mpsc::Receiver<Command>, events: EventRegistry) {
    let mut sessions: HashMap<SessionId, Session> = HashMap::new();
    let mut next_id: u64 = 1;
    while let Ok(cmd) = rx.recv() {
        handle(&ctx, &mut sessions, &mut next_id, &rx, &events, cmd);
    }
}

fn handle(
    ctx: &GpuContext,
    sessions: &mut HashMap<SessionId, Session>,
    next_id: &mut u64,
    rx: &std_mpsc::Receiver<Command>,
    events: &EventRegistry,
    cmd: Command,
) {
    match cmd {
        Command::Health { reply } => {
            let _ = reply.send(HealthInfo {
                adapter: ctx.adapter_name.clone(),
                backend: ctx.backend.clone(),
                device_type: ctx.device_type.clone(),
                is_software: ctx.is_software,
            });
        }

        Command::CreateSession { session, reply } => {
            let id = format!("s{}", *next_id);
            *next_id += 1;
            let vocab_size = session.config.vocab_size as u32;
            let params = session.config.param_count() as f64;
            let rng = Rng::seed_from_u64(1);
            sessions.insert(
                id.clone(),
                Session {
                    config: session.config,
                    weights: session.weights,
                    corpus: session.corpus,
                    rng,
                    trainer: None,
                    step: session.step,
                    tokens_seen: session.tokens_seen,
                },
            );
            let _ = reply.send(Ok(SessionCreated { session_id: id, vocab_size, params }));
        }

        Command::GetCheckpoint { session_id, reply } => {
            let Some(session) = sessions.get(&session_id) else {
                let _ = reply.send(Err(format!("no such session {session_id}")));
                return;
            };
            let bytes = llm_core::checkpoint::write_checkpoint(
                &session.config,
                &session.weights,
                session.corpus.tokenizer(),
                session.step,
                session.tokens_seen,
                0,
                1.0,
                WeightDtype::Bf16,
            );
            let _ = reply.send(Ok(bytes));
        }

        Command::DeleteSession { session_id, reply } => {
            sessions.remove(&session_id);
            events.lock().unwrap().remove(&session_id);
            let _ = reply.send(Ok(()));
        }

        Command::StopTraining { .. } => {
            // Nothing is training right now (run_training below is the
            // only place that loops), so a stop that arrives outside a
            // run is simply a no-op — there's nothing to stop.
        }

        Command::StartTraining { session_id, settings, reply } => {
            run_training(ctx, sessions, rx, events, session_id, settings, reply);
        }
    }
}

/// Runs an entire training run to completion (or until stopped),
/// blocking this thread the whole time — which is exactly the point:
/// nothing else touches the GPU until this returns. Commands that arrive
/// meanwhile are drained between steps; a stop for this same session
/// ends the run, anything else is told the GPU is busy.
#[allow(clippy::too_many_arguments)]
fn run_training(
    ctx: &GpuContext,
    sessions: &mut HashMap<SessionId, Session>,
    rx: &std_mpsc::Receiver<Command>,
    events: &EventRegistry,
    session_id: SessionId,
    settings: TrainSettings,
    reply: oneshot::Sender<Result<(), String>>,
) {
    let Some(session) = sessions.get_mut(&session_id) else {
        let _ = reply.send(Err(format!("no such session {session_id}")));
        return;
    };

    let t_len = session.config.context_len;
    if session.trainer.is_none() {
        match GpuTrainer::new(ctx, &session.config, &session.weights, t_len) {
            Ok(trainer) => session.trainer = Some(trainer),
            Err(err) => {
                let _ = reply.send(Err(format!("could not start training: {err}")));
                return;
            }
        }
    }

    let event_tx = {
        let (tx, _rx) = broadcast::channel::<TrainEvent>(EVENT_CHANNEL_CAPACITY);
        events.lock().unwrap().insert(session_id.clone(), tx.clone());
        tx
    };
    if reply.send(Ok(())).is_err() {
        // The request that started this run is already gone (the HTTP
        // connection dropped before we replied) - there is no one left
        // to report progress to, so there is no point running at all.
        return;
    }

    let train = TrainConfig {
        lr: settings.peak_learning_rate,
        total_steps: settings.max_steps as u64,
        warmup_steps: TrainConfig::warmup_for(settings.max_steps.max(1) as u64),
        start_step: session.step,
        weight_decay: settings.weight_decay,
        grad_clip: settings.grad_clip,
        ..TrainConfig::default()
    };
    let progress_every = std::time::Duration::from_millis(settings.progress_interval_ms.max(1));

    let started = Instant::now();
    let mut last_emit = Instant::now() - progress_every;
    let mut tokens_total: u64 = 0;
    let mut stop_reason: Option<String> = None;

    loop {
        if settings.max_steps > 0 && session.step >= settings.max_steps as u64 {
            break;
        }

        // Drain whatever arrived since the last step. A stop for this
        // session ends the run; anything else has to wait its turn.
        loop {
            match rx.try_recv() {
                Ok(Command::StopTraining { session_id: id }) if id == session_id => {
                    stop_reason = Some("stopped".to_string());
                }
                Ok(other) => reject_busy(other),
                Err(std_mpsc::TryRecvError::Empty) => break,
                Err(std_mpsc::TryRecvError::Disconnected) => {
                    stop_reason = Some("the server is shutting down".to_string());
                }
            }
            if stop_reason.is_some() {
                break;
            }
        }
        if stop_reason.is_some() {
            break;
        }

        let Some(batch) = session.corpus.sample_batch(settings.batch_size as usize, t_len, &mut session.rng)
        else {
            stop_reason = Some("not enough text to sample a batch".to_string());
            break;
        };
        let lr = train.lr_at(session.step);
        let trainer = session.trainer.as_mut().expect("built above");
        let result = pollster::block_on(trainer.train_step(
            ctx,
            &batch.inputs,
            &batch.targets,
            lr,
            train.weight_decay,
            train.grad_clip,
        ));
        let report = match result {
            Ok(report) => report,
            Err(err) => {
                stop_reason = Some(err);
                break;
            }
        };
        session.step += 1;
        session.tokens_seen += report.tokens as u64;
        tokens_total += report.tokens as u64;

        if last_emit.elapsed() >= progress_every {
            last_emit = Instant::now();
            let elapsed = started.elapsed().as_secs_f64();
            let _ = event_tx.send(TrainEvent::Progress {
                step: session.step,
                loss: report.loss,
                grad_norm: report.grad_norm,
                lr: report.lr,
                tokens_per_second: if elapsed > 0.0 { tokens_total as f64 / elapsed } else { 0.0 },
                elapsed_seconds: elapsed,
                fraction_done: train.progress_at(session.step) as f64,
            });
        }
    }

    // Pull the trained weights back into the session's own copy before
    // this run ends, the same reason `sync_from_gpu` exists client-side:
    // a checkpoint download or a later resumed run needs them here, not
    // still sitting on the device.
    if let Some(trainer) = &session.trainer {
        if let Ok(weights) = pollster::block_on(trainer.download_weights(ctx)) {
            session.weights = weights;
        }
    }

    let _ = event_tx.send(TrainEvent::Stopped {
        step: session.step,
        reason: stop_reason.unwrap_or_else(|| "finished".to_string()),
    });
}

/// Build a fresh model from an uploaded shape, seeded with a
/// freshly-built byte-level tokenizer — the only kind this server
/// builds a from-scratch model with today, matching `wasm-app`'s own
/// `WasmLLM::new`.
pub fn build_session_from_config(config: ModelConfig, seed: u64) -> NewSession {
    let weights = ModelWeights::init(&config, seed);
    NewSession {
        config,
        weights,
        corpus: Corpus::with_tokenizer(Tokenizer::byte_level()),
        step: 0,
        tokens_seen: 0,
    }
}

pub fn build_session_from_checkpoint(bytes: &[u8]) -> Result<NewSession, String> {
    let checkpoint = Checkpoint::from_bytes(bytes)?;
    Ok(NewSession {
        config: checkpoint.config,
        weights: checkpoint.weights,
        corpus: Corpus::with_tokenizer(checkpoint.tokenizer),
        step: checkpoint.step,
        tokens_seen: checkpoint.tokens_seen,
    })
}
