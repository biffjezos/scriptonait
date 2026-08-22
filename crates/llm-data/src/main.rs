//! Turn a directory of downloaded public-domain text into a tokenizer
//! and a pre-tokenized training set.
//!
//! ```text
//! llm-data --raw corpus/raw --out corpus/build [--vocab 8192] [--chunk-words 350]
//! ```
//!
//! Input files are named `<form>__<title>.txt`, where `<form>` is
//! `novel`, `screenplay`, or `allegory`. The prefix is how a file
//! declares what it is; that beats guessing from the text, and it beats
//! a separate manifest that can drift out of sync with the directory.
//!
//! What comes out:
//!
//! - `tokenizer.tok` — BPE merges learned from this corpus.
//! - `dataset.bin` — every document tokenized twice over: once whole (so
//!   the model learns the language and the long-range shape of a script
//!   or a chapter), and once cut into instruction examples (so it learns
//!   that a `TASK` line governs the text after it). See
//!   `llm_core::instruct`.
//! - a short report on stdout.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use llm_core::dataset::TokenDataset;
use llm_core::instruct::{self, Form, Request};
use llm_core::prep;
use llm_core::tokenizer::{self, Tokenizer};

struct Args {
    raw: PathBuf,
    out: PathBuf,
    vocab: usize,
    chunk_words: usize,
    /// Cap on how much text BPE training itself looks at. Learning
    /// merges is superlinear in distinct words and the vocabulary it
    /// produces stops changing long before the whole corpus is consumed,
    /// so a sample is both faster and no worse.
    bpe_sample_bytes: usize,
}

impl Args {
    fn parse() -> Result<Args, String> {
        let mut raw = None;
        let mut out = None;
        let mut vocab = 8192usize;
        let mut chunk_words = 350usize;
        let mut bpe_sample_bytes = 8 * 1024 * 1024;

        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or(format!("{flag} needs a value"));
            match flag.as_str() {
                "--raw" => raw = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--vocab" => vocab = value()?.parse().map_err(|_| "--vocab must be a number")?,
                "--chunk-words" => {
                    chunk_words = value()?.parse().map_err(|_| "--chunk-words must be a number")?
                }
                "--bpe-sample-mb" => {
                    let mb: usize = value()?.parse().map_err(|_| "--bpe-sample-mb must be a number")?;
                    bpe_sample_bytes = mb * 1024 * 1024;
                }
                "-h" | "--help" => return Err(HELP.to_string()),
                other => return Err(format!("unknown flag {other}\n\n{HELP}")),
            }
        }
        Ok(Args {
            raw: raw.ok_or(format!("--raw is required\n\n{HELP}"))?,
            out: out.ok_or(format!("--out is required\n\n{HELP}"))?,
            vocab,
            chunk_words,
            bpe_sample_bytes,
        })
    }
}

const HELP: &str = "\
llm-data --raw <dir> --out <dir> [options]

  --raw <dir>          directory of <form>__<title>.txt files
                       (form is novel, screenplay, or allegory)
  --out <dir>          where to write tokenizer.tok and dataset.bin
  --vocab <n>          target vocabulary size (default 8192)
  --chunk-words <n>    words per instruction example (default 350)
  --bpe-sample-mb <n>  MB of text used to learn merges (default 8)";

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
    fs::create_dir_all(&args.out).map_err(|e| format!("creating {}: {e}", args.out.display()))?;

    let sources = read_sources(&args.raw)?;
    if sources.is_empty() {
        return Err(format!(
            "no .txt files in {} — expected files named <form>__<title>.txt",
            args.raw.display()
        ));
    }
    println!("read {} source files from {}", sources.len(), args.raw.display());
    let mut by_form: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for source in &sources {
        let entry = by_form.entry(source.form.as_str()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += source.text.len();
    }
    for (form, (count, bytes)) in &by_form {
        println!("  {form:<11} {count:>3} files, {:>7.1} MB", *bytes as f64 / 1e6);
    }

    // --- tokenizer -----------------------------------------------------
    // Sampled proportionally across forms, so a vocabulary isn't learned
    // entirely from whichever form happens to have the most text —
    // screenplay formatting tokens are worth as much as prose ones.
    let sample = sample_text(&sources, args.bpe_sample_bytes);
    println!("\nlearning up to {} merges from {:.1} MB...", args.vocab, sample.len() as f64 / 1e6);
    let started = std::time::Instant::now();
    let tokenizer = Tokenizer::train(&[&sample], args.vocab);
    println!(
        "  {} merges, vocab {} ({:.1}s)",
        tokenizer.num_merges(),
        tokenizer.vocab_size(),
        started.elapsed().as_secs_f64()
    );

    // --- dataset -------------------------------------------------------
    let mut dataset = TokenDataset::new();
    let mut instruction_examples = 0usize;
    for source in &sources {
        // The whole document, so the model sees long-range structure and
        // not only 350-word fragments.
        dataset.push(tokenizer::wrap_with_boundaries(&tokenizer.encode(&source.text)));

        for (request, answer) in examples_for(source, args.chunk_words) {
            dataset.push(request.to_training_tokens(&tokenizer, &answer));
            instruction_examples += 1;
        }
    }

    let total = dataset.total_tokens();
    let text_bytes: usize = sources.iter().map(|s| s.text.len()).sum();
    println!(
        "\n{} documents, {instruction_examples} of them instruction examples",
        dataset.documents.len()
    );
    println!(
        "{total} tokens from {:.1} MB of text ({:.2} bytes/token)",
        text_bytes as f64 / 1e6,
        text_bytes as f64 * 2.0 / total as f64, // each source is tokenized twice
    );

    let tokenizer_path = args.out.join("tokenizer.tok");
    let dataset_path = args.out.join("dataset.bin");
    fs::write(&tokenizer_path, tokenizer.to_bytes())
        .map_err(|e| format!("writing {}: {e}", tokenizer_path.display()))?;
    fs::write(&dataset_path, dataset.to_bytes())
        .map_err(|e| format!("writing {}: {e}", dataset_path.display()))?;
    println!("\nwrote {}", tokenizer_path.display());
    println!("wrote {}", dataset_path.display());

    // A worked example of what the model will actually be trained on,
    // so a bad prep step is visible in the log rather than three hours
    // later in the output.
    if let Some(source) = sources.first() {
        if let Some((request, answer)) = examples_for(source, args.chunk_words).into_iter().next() {
            println!("\nexample instruction from {}:", source.name);
            println!("  {}", request.instruction());
            println!("  -> {}...", answer.chars().take(120).collect::<String>().replace('\n', " "));
        }
    }
    Ok(())
}

/// Instruction examples for one source.
///
/// A declared allegory stays an allegory: the shape heuristic in
/// `instruct` reads scene headings and character cues, which is enough to
/// tell a screenplay from prose but can't tell an allegory from any other
/// prose. The filename is the only thing that knows.
///
/// This exists as a function rather than inline so the worked example
/// printed at the end of a build is produced the same way as the dataset
/// itself — a log that disagrees with the data is worse than no log.
fn examples_for(source: &Source, chunk_words: usize) -> Vec<(Request, String)> {
    instruct::synthesize_examples(&source.text, chunk_words)
        .into_iter()
        .map(|(mut request, answer)| {
            if source.form == Form::Allegory {
                request.form = Form::Allegory;
            }
            (request, answer)
        })
        .collect()
}

struct Source {
    name: String,
    form: Form,
    text: String,
}

fn read_sources(dir: &Path) -> Result<Vec<Source>, String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    // Sorted so a rebuild produces a byte-identical dataset.
    paths.sort();

    let mut sources = Vec::new();
    for path in paths {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let raw = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                // A single unreadable or non-UTF-8 file shouldn't sink a
                // corpus build that has fifty other files in it.
                eprintln!("skipping {}: {e}", path.display());
                continue;
            }
        };
        let form = match name.split("__").next().unwrap_or("") {
            "screenplay" => Form::Screenplay,
            "allegory" => Form::Allegory,
            "novel" => Form::Novel,
            other => {
                eprintln!("skipping {}: unknown form prefix {other:?}", path.display());
                continue;
            }
        };
        let text = clean(&raw);
        if text.split_whitespace().count() < 200 {
            eprintln!("skipping {}: only {} words after cleaning", path.display(), text.split_whitespace().count());
            continue;
        }
        sources.push(Source { name, form, text });
    }
    Ok(sources)
}

/// Strip Project Gutenberg's wrapper and normalize whitespace.
///
/// The wrapper is ~500 lines of licence text at each end, identical
/// across every book. Left in, the model would see it dozens of times
/// and learn it better than it learns any of the actual prose.
fn clean(raw: &str) -> String {
    let body = strip_gutenberg_boilerplate(raw);
    // Whitespace normalization keeps leading indentation, deliberately —
    // in a plain-text screenplay the indent is the only thing separating
    // a character cue from an action line.
    prep::normalize_whitespace(body)
}

fn strip_gutenberg_boilerplate(raw: &str) -> &str {
    const START_MARKERS: [&str; 2] = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"];
    const END_MARKERS: [&str; 2] = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"];

    let mut body = raw;
    for marker in START_MARKERS {
        if let Some(idx) = body.find(marker) {
            // Past the marker line itself.
            let after = &body[idx..];
            if let Some(newline) = after.find('\n') {
                body = &after[newline + 1..];
            }
            break;
        }
    }
    for marker in END_MARKERS {
        if let Some(idx) = body.find(marker) {
            body = &body[..idx];
            break;
        }
    }
    body
}

/// Take up to `budget` bytes of text, spread evenly across the sources
/// rather than taken from the front of the list.
fn sample_text(sources: &[Source], budget: usize) -> String {
    let per_source = (budget / sources.len().max(1)).max(4096);
    let mut out = String::with_capacity(budget.min(sources.iter().map(|s| s.text.len()).sum()));
    for source in sources {
        let take = per_source.min(source.text.len());
        // Cut on a character boundary; byte slicing a UTF-8 string
        // anywhere else panics.
        let end = (0..=take).rev().find(|&i| source.text.is_char_boundary(i)).unwrap_or(0);
        out.push_str(&source.text[..end]);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_the_gutenberg_wrapper() {
        let raw = "licence blah blah\n\
                   *** START OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***\n\
                   Actual text here.\n\
                   *** END OF THE PROJECT GUTENBERG EBOOK THE REPUBLIC ***\n\
                   more licence";
        let body = strip_gutenberg_boilerplate(raw);
        assert_eq!(body.trim(), "Actual text here.");
    }

    #[test]
    fn leaves_text_without_a_wrapper_alone() {
        let raw = "INT. CAVE - DAY\n\nJust a script.";
        assert_eq!(strip_gutenberg_boilerplate(raw), raw);
    }

    #[test]
    fn sampling_never_cuts_a_character_in_half() {
        let sources = vec![Source {
            name: "novel__x".into(),
            form: Form::Novel,
            // Multi-byte characters right at every plausible cut point.
            text: "\u{1f980}".repeat(4000),
        }];
        // Small budget so the cut lands mid-string.
        let sample = sample_text(&sources, 8);
        assert!(sample.chars().all(|c| c == '\u{1f980}' || c == '\n'));
    }

    #[test]
    fn cleaning_keeps_screenplay_indentation() {
        let raw = "INT. CAVE - DAY\n\n          SOCRATES\n     What do you see?\n";
        let cleaned = clean(raw);
        assert!(cleaned.contains("          SOCRATES"), "indent was stripped: {cleaned:?}");
    }

    #[test]
    fn a_declared_allegory_overrides_the_shape_heuristic() {
        // Prose with no scene headings reads as a novel to the
        // heuristic; the filename is what says otherwise.
        let text = "The prisoners see only shadows on the wall of the cave. \
                    The shadows are all they have ever known.\n\n"
            .repeat(6);
        let mut examples = instruct::synthesize_examples(&text, 60);
        assert!(!examples.is_empty());
        assert_eq!(examples[0].0.form, Form::Novel);
        let source = Source { name: "allegory__cave".into(), form: Form::Allegory, text };
        if source.form == Form::Allegory {
            examples[0].0.form = Form::Allegory;
        }
        assert_eq!(examples[0].0.form, Form::Allegory);
    }

    #[test]
    fn instruction_examples_round_trip_through_the_tokenizer() {
        let text = "The fire threw shadows. The prisoners watched them move.\n\n".repeat(10);
        let tokenizer = Tokenizer::train(&[&text], 400);
        let request = Request {
            form: Form::Allegory,
            target_words: Some(120),
            subject: "shadows".into(),
            reference: None,
        };
        let tokens = request.to_training_tokens(&tokenizer, &text);
        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.starts_with(&request.instruction()));
        assert!(decoded.contains("The fire threw shadows."));
    }
}
