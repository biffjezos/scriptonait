//! Entry point: opens a real GPU device (see `gpu_actor.rs` — the one
//! thread that ever touches it) and serves the HTTP API `routes.rs`
//! defines, so a browser page can train or generate on this machine's
//! GPU instead of its own.
//!
//! This is a standalone crate for the same reason `llm-gpu`/`wasm-app`
//! are (see this crate's own `Cargo.toml`): it needs crates.io and a
//! real GPU adapter, neither available in this project's development
//! sandbox, so it cannot be built or tested there — only by CI, and by
//! actually running it.
//!
//! Usage: `llm-server [--bind 127.0.0.1] [--port 8420] [--token SECRET]`
//! (`LLM_SERVER_TOKEN` works instead of `--token`). Pointing a page's
//! Settings-tab "Remote server" at `http://127.0.0.1:8420` while this
//! runs on the same machine is a genuine way to test the whole remote
//! path — including whether `llm-gpu`'s native backend actually works
//! on this hardware — without a second machine.
//!
//! v1 scope, deliberately: no persistent session storage (a session
//! lives in this process's memory only, gone on restart or `DELETE`), a
//! flat-then-cosine learning-rate schedule with no held-out-loss/
//! plateau detection or autosave-to-file (all of which stay client-side
//! concerns for now — the page still owns the periodic checkpoint pull
//! that keeps a local copy current), and no remote-inference endpoint
//! (train-remote + infer-local is the combination that was actually
//! asked for; inference staying local is also what keeps this box's
//! GPU time billed only for training).

mod auth;
mod gpu_actor;
mod protocol;
mod routes;
mod state;

use tower_http::cors::CorsLayer;

use state::AppState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bind = arg_value(&args, "--bind").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = arg_value(&args, "--port").and_then(|v| v.parse().ok()).unwrap_or(8420);
    let token = arg_value(&args, "--token").or_else(|| std::env::var("LLM_SERVER_TOKEN").ok());
    if token.is_none() {
        eprintln!(
            "warning: no --token (or LLM_SERVER_TOKEN) set — anyone who can reach this \
             server can use it. Fine for same-machine testing or a private network, \
             not for anything reachable from the open internet."
        );
    }

    println!("Opening a GPU device (native wgpu — Vulkan/Metal/DX12, not a browser)...");
    let gpu = match gpu_actor::spawn() {
        Ok(gpu) => gpu,
        Err(err) => {
            eprintln!("could not open a GPU device: {err}");
            std::process::exit(1);
        }
    };
    println!("GPU ready.");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build the tokio runtime");
    runtime.block_on(serve(bind, port, token, gpu));
}

async fn serve(bind: String, port: u16, token: Option<String>, gpu: gpu_actor::GpuActorHandle) {
    let state = AppState { gpu, token };
    let app = routes::router(state).layer(CorsLayer::permissive());
    let addr = format!("{bind}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("could not bind {addr}: {err}");
            std::process::exit(1);
        }
    };
    println!("Listening on http://{addr}");
    if let Err(err) = axum::serve(listener, app).await {
        eprintln!("server error: {err}");
    }
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
