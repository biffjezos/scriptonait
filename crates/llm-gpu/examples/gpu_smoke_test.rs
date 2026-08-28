//! Native GPU smoke test: opens a real device outside a browser, compiles
//! every training kernel, and runs a few real steps on a tiny model with
//! synthetic data.
//!
//! Every part of this crate has, until now, only ever run through a
//! browser's WebGPU implementation — CI compiles it for wasm32 and checks
//! the shaders with naga, but nothing has actually executed them (see this
//! crate's own top-level doc comment). This is what proves the native wgpu
//! path — Vulkan on Linux/Windows, Metal on macOS — actually works on real
//! hardware, on whatever machine runs it. It is a hardware/kernel sanity
//! check, not a benchmark: the data is random, so the loss isn't expected
//! to fall by much in twenty steps. What matters is that a device is found,
//! every kernel (forward and backward) compiles, and a step produces a
//! finite loss without erroring.
//!
//! Run from this crate's own directory (it is deliberately not part of the
//! root workspace — see this crate's Cargo.toml):
//!
//!     cargo run --release --example gpu_smoke_test

use llm_core::config::ModelConfig;
use llm_core::model::ModelWeights;
use llm_core::rng::Rng;
use llm_gpu::{supports_training, GpuContext, GpuTrainer};

fn main() {
    if let Err(err) = pollster::block_on(run()) {
        eprintln!("FAILED: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    println!("Requesting a GPU device (native wgpu — Vulkan/Metal/DX12, not a browser)...");
    let ctx = GpuContext::new().await?;
    println!("  adapter: {}", ctx.adapter_summary);
    println!(
        "  backend: {}, device type: {}, software: {}, f16: {}",
        ctx.backend, ctx.device_type, ctx.is_software, ctx.has_f16
    );
    if ctx.is_software {
        println!(
            "  NOTE: this is a software renderer, not real hardware — training will be slow, \
             and this isn't the machine to judge speed on."
        );
    }

    let config = ModelConfig::default();
    println!(
        "\nBuilding a tiny model to train on: {} layers, {} hidden, {} parameters",
        config.num_layers,
        config.hidden_dim,
        config.param_count(),
    );
    if !supports_training(&config) {
        return Err("this backend does not support training a model this shape".to_string());
    }
    let weights = ModelWeights::init(&config, 1);

    let t_len = 64usize.min(config.context_len);
    let batch_size = 4;
    let mut gpu_trainer = GpuTrainer::new(&ctx, &config, &weights, t_len)
        .map_err(|e| format!("could not allocate training buffers: {e}"))?;
    println!(
        "Training buffers allocated ({:.1} MB) — every kernel, forward and backward, just \
         compiled and linked.",
        gpu_trainer.allocated_bytes() as f64 / (1024.0 * 1024.0),
    );

    println!(
        "\nRunning 20 training steps on synthetic data (random tokens — a kernel check, not \
         a training run)..."
    );
    let mut rng = Rng::seed_from_u64(7);
    let vocab = config.vocab_size();
    const STEPS: u32 = 20;
    let mut first_loss = None;
    let mut last_loss = f32::NAN;
    let mut total_ms = 0.0f64;
    for step in 1..=STEPS {
        let mut inputs = Vec::with_capacity(batch_size * t_len);
        let mut targets = Vec::with_capacity(batch_size * t_len);
        for _ in 0..batch_size * t_len {
            inputs.push(rng.gen_range(vocab) as u32);
            targets.push(rng.gen_range(vocab) as u32);
        }
        let started = web_time::Instant::now();
        let report = gpu_trainer
            .train_step(&ctx, &inputs, &targets, 1e-4, 0.0, 1.0)
            .await
            .map_err(|e| format!("step {step} failed: {e}"))?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;
        if first_loss.is_none() {
            first_loss = Some(report.loss);
        }
        last_loss = report.loss;
        if !report.loss.is_finite() {
            return Err(format!("step {step} produced a non-finite loss ({}) — a kernel is wrong", report.loss));
        }
        println!(
            "  step {step:>2}: loss {:.4}  grad_norm {:.3}  {elapsed_ms:.1} ms",
            report.loss, report.grad_norm
        );
    }

    println!("\nDone. {STEPS} steps, {:.1} ms/step average.", total_ms / STEPS as f64);
    println!(
        "Loss went {:.4} -> {last_loss:.4} on random data (not expected to fall much) — the \
         backward pass and Adam update ran to completion {STEPS} times without erroring.",
        first_loss.unwrap_or(f32::NAN),
    );
    Ok(())
}
