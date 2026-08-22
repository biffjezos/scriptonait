//! Throughput benchmark for the CPU engine.
//!
//! Prints tokens/second for a training step (forward + cross-entropy +
//! backward + optimizer) and for generation, at a few model shapes. This
//! exists so "the model is faster now" is a number someone can reproduce
//! rather than a claim in a commit message:
//!
//!     cargo run --release -p llm-bench
//!
//! Build it in release mode; a debug build measures the borrow checker's
//! idea of arithmetic, not the engine's.

use std::time::Instant;

use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::train::Trainer;
use llm_core::{generate, model};

fn shapes() -> Vec<(&'static str, ModelConfig)> {
    vec![
        (
            "tiny  (4L x 128)",
            ModelConfig { num_layers: 4, hidden_dim: 128, num_heads: 4, context_len: 256, local_window: 256 },
        ),
        (
            "small (6L x 256)",
            ModelConfig { num_layers: 6, hidden_dim: 256, num_heads: 8, context_len: 512, local_window: 256 },
        ),
    ]
}

fn bench_training(name: &str, config: &ModelConfig, threads: usize) {
    let mut corpus = Corpus::new();
    corpus.upsert("bench", &sample_text(), false);
    let mut trainer = Trainer::new(*config, 1234);
    trainer.set_threads(threads);
    // A batch of one has nothing to split, so the batch is sized to the
    // thread count - which is what the native trainer does too.
    let batch = threads;

    // One untimed step so allocation and first-touch page faults don't
    // land inside the measurement.
    trainer.train_step(&mut corpus, batch, 1e-4);

    let steps = 3;
    let start = Instant::now();
    for _ in 0..steps {
        trainer.train_step(&mut corpus, batch, 1e-4);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tokens = (steps * batch * config.context_len) as f64;
    println!(
        "  train  {name} x{threads:<2}: {:>8.1} tok/s  ({:.0} ms/step of {} x {} tokens)",
        tokens / elapsed,
        elapsed * 1000.0 / steps as f64,
        batch,
        config.context_len
    );
}

fn bench_generation(name: &str, config: &ModelConfig) {
    let weights = model::ModelWeights::init(config, 7);
    let new_tokens = 64;
    let start = Instant::now();
    let out = generate::generate(&weights, config, "INT. SHIP - NIGHT", new_tokens, 0.8, 99);
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(out);
    println!("  gen    {name}    : {:>8.1} tok/s  ({new_tokens} new tokens)", new_tokens as f64 / elapsed);
}

fn sample_text() -> String {
    // Long enough that batch sampling has real windows to choose from.
    "INT. CAVE - CONTINUOUS\n\nThe prisoners have been here since childhood, \
     chained so they can only see the wall in front of them.\n\nSOCRATES\n\
     And what of the shadows? Would they not take them for the whole of \
     what is real?\n\n"
        .repeat(200)
}

fn main() {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("cores: {cores}");
    for (name, config) in shapes() {
        println!("{name}  ({} params)", config.param_count());
        bench_training(name, &config, 1);
        if cores > 1 {
            bench_training(name, &config, cores);
        }
        bench_generation(name, &config);
    }
}
