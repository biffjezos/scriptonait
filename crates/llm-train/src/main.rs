//! Native pretraining, with a time budget and resumable checkpoints.
//!
//! ```text
//! llm-train --data corpus/build --out frontend/model/scriptonait.ckpt \
//!           --minutes 240 --layers 6 --hidden 320 --heads 8 --kv-heads 2
//! ```
//!
//! This is where the model is actually trained. Not in a browser tab: a
//! tab gets one thread (wasm threads need cross-origin isolation, which
//! GitHub Pages doesn't serve), competes with the compositor for the
//! machine, and loses everything when it's closed. Here there are real
//! threads, native optimization flags, and a checkpoint on disk.
//!
//! Runs are resumable because they have to be — a GitHub Actions job is
//! capped at six hours and useful training is longer than that. Weights
//! and step count travel in the checkpoint; Adam's moment buffers travel
//! in a separate file next to it (three times the size, and only a
//! resuming trainer wants them), so a continued run picks up its
//! momentum and its learning-rate schedule instead of visibly notching
//! the loss curve at every restart.
//!
//! Progress goes to stdout on a fixed interval: step, loss, learning
//! rate, gradient norm, throughput, elapsed, and ETA. That's the same
//! information the browser UI reports, for the same reason — a long job
//! that prints nothing is indistinguishable from a hung one.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use llm_core::checkpoint::{Checkpoint, WeightDtype};
use llm_core::config::ModelConfig;
use llm_core::corpus::Corpus;
use llm_core::dataset::TokenDataset;
use llm_core::generate::SamplingConfig;
use llm_core::instruct;
use llm_core::tokenizer::Tokenizer;
use llm_core::train::{TrainConfig, Trainer};

struct Args {
    data: PathBuf,
    out: PathBuf,
    /// Where to also write a bf16 copy for the browser to download.
    web_out: Option<PathBuf>,
    resume: Option<PathBuf>,
    minutes: Option<f64>,
    steps: Option<u64>,
    total_steps: u64,
    batch: usize,
    threads: Option<usize>,
    seed: u64,
    log_every_secs: f64,
    save_every_secs: f64,
    shape: Shape,
    train: TrainConfig,
    sample_prompt: String,
}

struct Shape {
    layers: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    context: usize,
    window: usize,
    rope_theta: f32,
}

impl Default for Shape {
    fn default() -> Self {
        // Sized for what a CI runner can actually train to a useful
        // point in a handful of hours, not for what sounds impressive.
        // See the README's note on the honest ceiling.
        Self { layers: 6, hidden: 320, heads: 8, kv_heads: 2, context: 512, window: 256, rope_theta: 10000.0 }
    }
}

const HELP: &str = "\
llm-train --data <dir> --out <file.ckpt> [options]

  --data <dir>         directory holding dataset.bin and tokenizer.tok
  --out <file>         checkpoint to write (optimizer state goes to
                       <file>.opt)
  --web-out <file>     also write a bf16 copy here — half the bytes, for
                       the browser to download
  --resume <file>      continue from this checkpoint instead of a fresh
                       model; the shape flags are then ignored, since the
                       checkpoint carries its own
  --minutes <n>        stop after roughly this long (default: no limit)
  --steps <n>          stop after this many steps this run
  --total-steps <n>    steps the whole training plan is for, which shapes
                       the cosine decay (default 60000)
  --batch <n>          sequences per step (default: thread count)
  --threads <n>        worker threads (default: all cores)
  --lr <f>             peak learning rate (default 3e-4)
  --warmup <n>         warmup steps (default 200)
  --weight-decay <f>   AdamW decay (default 0.1)
  --grad-clip <f>      global gradient-norm clip (default 1.0)
  --layers/--hidden/--heads/--kv-heads/--context/--window/--rope-theta
                       model shape for a fresh run
  --seed <n>           RNG seed (default 1)
  --log-every <secs>   progress line interval (default 30)
  --save-every <secs>  checkpoint interval (default 600)
  --sample-prompt <s>  prompt used for the end-of-run sample";

impl Args {
    fn parse() -> Result<Args, String> {
        let mut data = None;
        let mut out = None;
        let mut web_out = None;
        let mut resume = None;
        let mut minutes = None;
        let mut steps = None;
        let mut total_steps = 60_000u64;
        let mut batch = None;
        let mut threads = None;
        let mut seed = 1u64;
        let mut log_every_secs = 30.0;
        let mut save_every_secs = 600.0;
        let mut shape = Shape::default();
        let mut train = TrainConfig::default();
        let mut sample_prompt =
            "Write a 700 word novel about two people in space related to Plato's allegory of the cave"
                .to_string();

        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or(format!("{flag} needs a value"));
            let num = |s: String, what: &str| -> Result<f64, String> {
                s.parse::<f64>().map_err(|_| format!("{what} must be a number"))
            };
            match flag.as_str() {
                "--data" => data = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--web-out" => web_out = Some(PathBuf::from(value()?)),
                "--resume" => resume = Some(PathBuf::from(value()?)),
                "--minutes" => minutes = Some(num(value()?, "--minutes")?),
                "--steps" => steps = Some(num(value()?, "--steps")? as u64),
                "--total-steps" => total_steps = num(value()?, "--total-steps")? as u64,
                "--batch" => batch = Some(num(value()?, "--batch")? as usize),
                "--threads" => threads = Some(num(value()?, "--threads")? as usize),
                "--lr" => train.lr = num(value()?, "--lr")? as f32,
                "--warmup" => train.warmup_steps = num(value()?, "--warmup")? as u64,
                "--weight-decay" => train.weight_decay = num(value()?, "--weight-decay")? as f32,
                "--grad-clip" => train.grad_clip = num(value()?, "--grad-clip")? as f32,
                "--layers" => shape.layers = num(value()?, "--layers")? as usize,
                "--hidden" => shape.hidden = num(value()?, "--hidden")? as usize,
                "--heads" => shape.heads = num(value()?, "--heads")? as usize,
                "--kv-heads" => shape.kv_heads = num(value()?, "--kv-heads")? as usize,
                "--context" => shape.context = num(value()?, "--context")? as usize,
                "--window" => shape.window = num(value()?, "--window")? as usize,
                "--rope-theta" => shape.rope_theta = num(value()?, "--rope-theta")? as f32,
                "--seed" => seed = num(value()?, "--seed")? as u64,
                "--log-every" => log_every_secs = num(value()?, "--log-every")?,
                "--save-every" => save_every_secs = num(value()?, "--save-every")?,
                "--sample-prompt" => sample_prompt = value()?,
                "-h" | "--help" => return Err(HELP.to_string()),
                other => return Err(format!("unknown flag {other}\n\n{HELP}")),
            }
        }
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        train.total_steps = total_steps;
        Ok(Args {
            data: data.ok_or(format!("--data is required\n\n{HELP}"))?,
            out: out.ok_or(format!("--out is required\n\n{HELP}"))?,
            web_out,
            resume,
            minutes,
            steps,
            total_steps,
            batch: batch.unwrap_or(cores),
            threads,
            seed,
            log_every_secs,
            save_every_secs,
            shape,
            train,
            sample_prompt,
        })
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;

    let dataset_path = args.data.join("dataset.bin");
    let dataset = TokenDataset::from_bytes(&read(&dataset_path)?)
        .map_err(|e| format!("{}: {e}", dataset_path.display()))?;

    // Resuming takes the tokenizer from the checkpoint, not from the
    // data directory: the weights are indexed by *that* vocabulary, and
    // a rebuilt corpus could have produced a different one.
    let resumed = match &args.resume {
        Some(path) if path.exists() => {
            Some(Checkpoint::from_bytes(&read(path)?).map_err(|e| format!("{}: {e}", path.display()))?)
        }
        Some(path) => {
            println!("no checkpoint at {} yet — starting fresh", path.display());
            None
        }
        None => None,
    };

    let (config, tokenizer, start_step, weights) = match resumed {
        Some(checkpoint) => {
            println!(
                "resuming from step {} ({} params, vocab {})",
                checkpoint.step,
                checkpoint.config.param_count(),
                checkpoint.config.vocab_size
            );
            (checkpoint.config, checkpoint.tokenizer, checkpoint.step, Some(checkpoint.weights))
        }
        None => {
            let tokenizer_path = args.data.join("tokenizer.tok");
            let tokenizer = Tokenizer::from_bytes(&read(&tokenizer_path)?)
                .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;
            let config = ModelConfig {
                num_layers: args.shape.layers,
                hidden_dim: args.shape.hidden,
                num_heads: args.shape.heads,
                num_kv_heads: args.shape.kv_heads,
                context_len: args.shape.context,
                local_window: args.shape.window,
                vocab_size: tokenizer.vocab_size(),
                rope_theta: args.shape.rope_theta,
                use_ple: false,
            };
            config.validate().map_err(|e| e.to_string())?;
            (config, tokenizer, 0, None)
        }
    };

    let mut corpus = Corpus::with_tokenizer(tokenizer.clone());
    for (i, document) in dataset.documents.iter().enumerate() {
        corpus.upsert_tokens(&format!("d{i}"), document.clone());
    }
    if !corpus.can_sample(config.context_len) {
        return Err(format!(
            "the dataset has {} tokens, which isn't enough to fill even one {}-token window",
            corpus.total_tokens(),
            config.context_len
        ));
    }

    let mut trainer = match weights {
        Some(weights) => Trainer::resume(config, weights, start_step, args.seed),
        None => Trainer::new(config, args.seed),
    };
    if let Some(threads) = args.threads {
        trainer.set_threads(threads);
    }
    let optimizer_path = optimizer_path_for(&args.out);
    if optimizer_path.exists() && start_step > 0 {
        match trainer.load_optimizer(&read(&optimizer_path)?) {
            // A mismatch here means the shape changed under a resume.
            // Carrying on with fresh moments is right (Adam recovers in
            // a few hundred steps); carrying on silently is not.
            Err(e) => eprintln!("ignoring {}: {e}", optimizer_path.display()),
            Ok(()) => println!("resumed optimizer state from {}", optimizer_path.display()),
        }
    }

    println!(
        "model: {} layers x {} hidden, {} heads ({} kv), context {}, window {}, vocab {}",
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.context_len,
        config.local_window,
        config.vocab_size
    );
    println!(
        "       {:.2}M parameters, {:.0} MB to train",
        config.param_count() as f64 / 1e6,
        config.memory_bytes(true) as f64 / 1e6
    );
    println!(
        "data:  {} documents, {:.2}M tokens",
        dataset.documents.len(),
        corpus.total_tokens() as f64 / 1e6
    );
    println!(
        "run:   batch {}, {} threads, lr {:.1e} (warmup {}), plan {} steps\n",
        args.batch,
        trainer.threads(),
        args.train.lr,
        args.train.warmup_steps,
        args.total_steps
    );

    let started = Instant::now();
    let deadline = args.minutes.map(|m| started + Duration::from_secs_f64(m * 60.0));
    let stop_at_step = args.steps.map(|n| start_step + n);
    let mut last_log = Instant::now();
    let mut last_save = Instant::now();
    let mut smoothed_loss: Option<f32> = None;
    let mut tokens_since_log = 0usize;
    let mut steps_this_run = 0u64;

    loop {
        let Some(report) = trainer.train_step_with(&mut corpus, args.batch, &args.train) else {
            return Err("the corpus stopped producing batches".to_string());
        };
        steps_this_run += 1;
        tokens_since_log += report.tokens;
        // An exponential moving average, because a single batch's loss
        // is noisy enough at this batch size to hide the trend entirely.
        smoothed_loss = Some(match smoothed_loss {
            Some(previous) => previous * 0.95 + report.loss * 0.05,
            None => report.loss,
        });

        if !report.loss.is_finite() {
            save(&trainer, &tokenizer, &args.out, &optimizer_path, args.web_out.as_deref())?;
            return Err(format!(
                "loss went to {} at step {} — saved the last good checkpoint; \
                 lower --lr or --grad-clip and resume",
                report.loss, trainer.step
            ));
        }

        let now = Instant::now();
        if now.duration_since(last_log).as_secs_f64() >= args.log_every_secs {
            let elapsed = now.duration_since(last_log).as_secs_f64();
            let tok_per_sec = tokens_since_log as f64 / elapsed;
            log_progress(
                trainer.step,
                args.total_steps,
                smoothed_loss.unwrap_or(report.loss),
                report.lr,
                report.grad_norm,
                tok_per_sec,
                started.elapsed(),
                deadline.map(|d| d.saturating_duration_since(now)),
            );
            tokens_since_log = 0;
            last_log = now;
        }

        if now.duration_since(last_save).as_secs_f64() >= args.save_every_secs {
            save(&trainer, &tokenizer, &args.out, &optimizer_path, args.web_out.as_deref())?;
            println!("  saved {} at step {}", args.out.display(), trainer.step);
            last_save = now;
        }

        if deadline.is_some_and(|d| now >= d) || stop_at_step.is_some_and(|s| trainer.step >= s) {
            break;
        }
    }

    save(&trainer, &tokenizer, &args.out, &optimizer_path, args.web_out.as_deref())?;
    println!(
        "\ndone: {} steps this run ({} total), {:.1} minutes, loss {:.4}",
        steps_this_run,
        trainer.step,
        started.elapsed().as_secs_f64() / 60.0,
        smoothed_loss.unwrap_or(f32::NAN)
    );
    println!("saved {}", args.out.display());

    print_sample(&trainer, &tokenizer, &args.sample_prompt);
    Ok(())
}

/// One progress line. The point is that someone reading the log can tell
/// whether it's working *and* when it will be done, without waiting for
/// it to be done.
#[allow(clippy::too_many_arguments)]
fn log_progress(
    step: u64,
    total_steps: u64,
    loss: f32,
    lr: f32,
    grad_norm: f32,
    tok_per_sec: f64,
    elapsed: Duration,
    remaining_budget: Option<Duration>,
) {
    let progress = (step as f64 / total_steps.max(1) as f64 * 100.0).min(100.0);
    let eta = match remaining_budget {
        Some(budget) => format!("budget left {}", human_duration(budget)),
        None => "no time limit".to_string(),
    };
    println!(
        "step {step:>7} ({progress:>5.1}% of plan)  loss {loss:>7.4}  lr {lr:>8.2e}  |g| {grad_norm:>6.2}  {tok_per_sec:>7.0} tok/s  elapsed {}  {eta}",
        human_duration(elapsed)
    );
    // CI log viewers buffer aggressively; without this a four-hour job
    // looks silent for its first hour.
    let _ = std::io::stdout().flush();
}

fn human_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Generate from the trained model so the log shows what it actually
/// sounds like. A loss number doesn't tell you whether the instruction
/// format took.
fn print_sample(trainer: &Trainer, tokenizer: &Tokenizer, prompt: &str) {
    let request = instruct::parse_prompt(prompt);
    println!("\nsample for {prompt:?}");
    println!("  instruction: {}", request.instruction());
    let response = instruct::generate_response(
        &trainer.weights,
        &trainer.config,
        tokenizer,
        // Keep the sample short regardless of what the prompt asked
        // for; this is a smoke test, not a deliverable.
        &instruct::Request { target_words: Some(80), ..request },
        &SamplingConfig { seed: 7, ..SamplingConfig::default() },
        &mut |_, _| true,
    );
    for line in response.text.lines().take(20) {
        println!("  | {line}");
    }
    println!("  ({} words, {:?})", response.word_count, response.stop_reason);
}

fn optimizer_path_for(checkpoint: &Path) -> PathBuf {
    let mut name = checkpoint.as_os_str().to_os_string();
    name.push(".opt");
    PathBuf::from(name)
}

fn save(
    trainer: &Trainer,
    tokenizer: &Tokenizer,
    out: &Path,
    optimizer_path: &Path,
    web_out: Option<&Path>,
) -> Result<(), String> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let checkpoint = Checkpoint {
        config: trainer.config,
        weights: trainer.weights.clone(),
        tokenizer: tokenizer.clone(),
        step: trainer.step,
    };
    write_atomically(out, &checkpoint.to_bytes())?;
    if let Some(web_out) = web_out {
        if let Some(parent) = web_out.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        write_atomically(web_out, &checkpoint.to_bytes_with(WeightDtype::Bf16))?;
    }
    write_atomically(optimizer_path, &trainer.optimizer_bytes())
}

/// Write to a temporary file and rename over the target.
///
/// A job killed partway through a 40 MB write would otherwise leave a
/// truncated checkpoint, and the next run would resume from it — or,
/// worse, the site would serve it. Rename is atomic, so the file at the
/// target path is always a complete one.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut temp = path.as_os_str().to_os_string();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    fs::write(&temp, bytes).map_err(|e| format!("writing {}: {e}", temp.display()))?;
    fs::rename(&temp, path).map_err(|e| format!("renaming into {}: {e}", path.display()))
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_state_sits_next_to_the_checkpoint() {
        assert_eq!(
            optimizer_path_for(Path::new("frontend/model/scriptonait.ckpt")),
            PathBuf::from("frontend/model/scriptonait.ckpt.opt")
        );
    }

    #[test]
    fn durations_read_as_durations() {
        assert_eq!(human_duration(Duration::from_secs(45)), "45s");
        assert_eq!(human_duration(Duration::from_secs(125)), "2m05s");
        assert_eq!(human_duration(Duration::from_secs(7325)), "2h02m");
    }

    #[test]
    fn an_atomic_write_replaces_the_target_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!("llm-train-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("model.ckpt");
        fs::write(&target, b"old").unwrap();
        write_atomically(&target, b"new contents").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new contents");
        assert!(!dir.join("model.ckpt.tmp").exists(), "temp file was left behind");
        fs::remove_dir_all(&dir).ok();
    }
}
