//! HTTP surface. Kept as one file for now — six small handlers plus a
//! couple of shared helpers is nowhere near the size that earns a
//! `routes/` split the rest of this project's larger files got.

use std::convert::Infallible;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};

use base64::Engine;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use llm_core::config::{LayerSharing, ModelConfig};
use llm_core::tokenizer::Tokenizer;

use crate::auth;
use crate::gpu_actor::{self, TrainSettings};
use crate::protocol::{
    CreateSessionRequest, CreateSessionResponse, ErrorResponse, HealthResponse, TrainStartRequest,
};
use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/session", post(create_session))
        .route("/session/:id", delete(delete_session))
        .route("/session/:id/train/start", post(start_training))
        .route("/session/:id/train/stop", post(stop_training))
        .route("/session/:id/train/events", get(train_events))
        .route("/session/:id/checkpoint", get(get_checkpoint))
        .route_layer(axum::middleware::from_fn_with_state(state.clone(), auth::require_token));

    Router::new().route("/health", get(health)).merge(protected).with_state(state)
}

fn error_response(status: StatusCode, message: String) -> Response {
    (status, Json(ErrorResponse { error: message })).into_response()
}

async fn health(State(state): State<AppState>) -> Response {
    match state.gpu.health().await {
        Ok(info) => Json(HealthResponse {
            ok: true,
            adapter: info.adapter,
            backend: info.backend,
            device_type: info.device_type,
            is_software: info.is_software,
        })
        .into_response(),
        Err(err) => error_response(StatusCode::SERVICE_UNAVAILABLE, err),
    }
}

async fn create_session(State(state): State<AppState>, Json(req): Json<CreateSessionRequest>) -> Response {
    // Only plain, `Send`-safe data gets built here — the actual
    // `Corpus`/`ModelWeights` construction happens on the GPU actor
    // thread itself (see `gpu_actor::build_session`), since
    // `llm_core::corpus::Corpus` is `!Send` and must never cross the
    // actor's channel.
    let origin = if let Some(checkpoint_b64) = &req.checkpoint_base64 {
        let bytes = match base64::engine::general_purpose::STANDARD.decode(checkpoint_b64) {
            Ok(bytes) => bytes,
            Err(err) => return error_response(StatusCode::BAD_REQUEST, format!("bad base64: {err}")),
        };
        gpu_actor::SessionOrigin::Checkpoint { bytes }
    } else if let Some(cfg) = &req.config {
        let vocab_size = Tokenizer::byte_level().vocab_size();
        let config = ModelConfig {
            num_layers: cfg.num_layers,
            hidden_dim: cfg.hidden_dim,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            context_len: cfg.context_len,
            local_window: cfg.local_window,
            vocab_size,
            // `RecurrentCore` has no wire field yet (see
            // `ModelConfigUpload::unique_layers`) — a remote client can
            // only ask for `Off` or `UniformGroups`.
            layer_sharing: match cfg.unique_layers {
                Some(unique_layers) if unique_layers != cfg.num_layers => {
                    LayerSharing::UniformGroups { unique_layers }
                }
                _ => LayerSharing::Off,
            },
            ..Default::default()
        };
        if let Err(err) = config.validate() {
            return error_response(StatusCode::BAD_REQUEST, err.to_string());
        }
        gpu_actor::SessionOrigin::Fresh { config, seed: cfg.seed }
    } else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "either config or checkpoint_base64 must be set".to_string(),
        );
    };

    let sources = req
        .sources
        .into_iter()
        .map(|s| gpu_actor::SourceSpec { id: s.id, text: s.text, is_html: s.is_html })
        .collect();
    let spec = gpu_actor::SessionSpec { origin, sources };

    match state.gpu.create_session(spec).await {
        Ok(created) => Json(CreateSessionResponse {
            session_id: created.session_id,
            vocab_size: created.vocab_size,
            params: created.params,
        })
        .into_response(),
        Err(err) => error_response(StatusCode::SERVICE_UNAVAILABLE, err),
    }
}

async fn delete_session(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.gpu.delete_session(id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(err) => error_response(StatusCode::SERVICE_UNAVAILABLE, err),
    }
}

async fn start_training(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<TrainStartRequest>,
) -> Response {
    let settings = TrainSettings {
        batch_size: req.batch_size,
        peak_learning_rate: req.peak_learning_rate,
        max_steps: req.max_steps,
        weight_decay: req.weight_decay,
        grad_clip: req.grad_clip,
        progress_interval_ms: req.progress_interval_ms,
    };
    match state.gpu.start_training(id, settings).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(err) => error_response(StatusCode::CONFLICT, err),
    }
}

async fn stop_training(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    state.gpu.stop_training(id);
    StatusCode::OK.into_response()
}

/// Server-Sent Events: connect after `start_training` has returned
/// successfully. Each event's `data` is the same JSON shape
/// `frontend/worker.js`'s `post('train-progress'|'train-stopped', ...)`
/// already sends, so `remote-backend.js` can hand it straight to
/// `app.js`'s existing `onStream` handlers.
async fn train_events(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let Some(receiver) = state.gpu.subscribe_train_events(&id) else {
        return error_response(
            StatusCode::NOT_FOUND,
            "no training run has started for this session yet".to_string(),
        );
    };
    let stream = BroadcastStream::new(receiver).map(|item| {
        let event = match item {
            Ok(event) => {
                let json = serde_json::to_string(&event).unwrap_or_default();
                Event::default().data(json)
            }
            // A receiver that falls far enough behind the broadcast
            // channel's buffer just skips the gap - a comment line the
            // client ignores, rather than an error that would close the
            // connection over a few missed progress ticks.
            Err(_) => Event::default().comment("missed some events"),
        };
        Ok::<Event, Infallible>(event)
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

async fn get_checkpoint(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.gpu.get_checkpoint(id).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        Err(err) => error_response(StatusCode::NOT_FOUND, err),
    }
}
