//! Turning a prompt into an instruction the model was trained to answer.
//!
//! A plain language model continues text. Given "Write a 700 word novel
//! about two people in space related to Plato's allegory of the cave" it
//! will happily continue *that sentence* — more instructions, a synopsis,
//! anything but the story — because a sentence like that, in the books it
//! read, is followed by more prose about writing, not by the thing being
//! asked for.
//!
//! So the model is trained on a fixed instruction format instead:
//!
//! ```text
//!   BOS TASK form=novel; words=700; about: two people in space;
//!            echoing: Plato's allegory of the cave STORY <the text> EOS
//! ```
//!
//! `TASK` and `STORY` are single tokens (see `tokenizer.rs`), so the
//! boundary between "what was asked" and "the answer" is unambiguous and
//! costs two tokens rather than a paragraph of prompt scaffolding.
//! Training examples in this format are synthesized from the corpus by
//! [`synthesize_examples`]: each chunk of a real script or novel is
//! paired with the instruction that *would have* asked for it. The model
//! never sees a hand-written instruction during training, and doesn't
//! need to — what it has to learn is that the text after `STORY` is
//! shaped by the fields before it.
//!
//! Everything here is deliberately mechanical. Parsing is a keyword scan,
//! not a language model, so a prompt that doesn't match any pattern still
//! produces a usable request (form unspecified, no length target, the
//! whole prompt as the subject) rather than an error.

use crate::config::ModelConfig;
use crate::generate::{self, SamplingConfig, StopReason};
use crate::model::ModelWeights;
use crate::screenplay;
use crate::tokenizer::{self, Tokenizer};

/// The kind of thing being asked for. This is what the model conditions
/// its formatting on: prose paragraphs versus scene headings and
/// character cues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Novel,
    Screenplay,
    Allegory,
    /// Nothing in the prompt said. Training examples get a concrete form
    /// (it's derived from the text), so this only appears at generation
    /// time and means "the model picks".
    Unspecified,
}

impl Form {
    /// The token written into the instruction. Stable: changing these
    /// strings changes what every checkpoint was trained against.
    pub fn as_str(self) -> &'static str {
        match self {
            Form::Novel => "novel",
            Form::Screenplay => "screenplay",
            Form::Allegory => "allegory",
            Form::Unspecified => "any",
        }
    }

    fn from_str(s: &str) -> Form {
        match s {
            "novel" => Form::Novel,
            "screenplay" => Form::Screenplay,
            "allegory" => Form::Allegory,
            _ => Form::Unspecified,
        }
    }
}

/// A parsed generation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub form: Form,
    /// Target length in words, if the prompt asked for one.
    pub target_words: Option<usize>,
    /// What it should be about.
    pub subject: String,
    /// A work or idea it should draw on ("Plato's allegory of the
    /// cave"), kept separate from the subject because they play
    /// different roles: the subject is what happens, this is what it
    /// rhymes with.
    pub reference: Option<String>,
}

impl Request {
    /// The canonical instruction line — exactly what goes between the
    /// `TASK` and `STORY` tokens, in training and at generation time
    /// alike. Any divergence between the two is a silent quality bug, so
    /// there is one function and both paths call it.
    pub fn instruction(&self) -> String {
        let mut out = format!("form={}", self.form.as_str());
        if let Some(words) = self.target_words {
            // Bucketed, not exact. The model cannot count, and training
            // it against exact word counts teaches it a number it can't
            // honour; buckets are a length *register* it can learn -
            // "this is a short one" - while the actual stopping is done
            // by the caller, which can count.
            out.push_str(&format!("; words={}", bucket_words(words)));
        }
        if !self.subject.is_empty() {
            out.push_str(&format!("; about: {}", self.subject));
        }
        if let Some(reference) = &self.reference {
            out.push_str(&format!("; echoing: {reference}"));
        }
        out
    }

    /// Token ids for the instruction, ending with `STORY` — feed this to
    /// the generator and it produces the answer.
    pub fn to_prompt_tokens(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tokens = vec![tokenizer::BOS, tokenizer::TASK];
        tokens.extend(tokenizer.encode(&self.instruction()));
        tokens.push(tokenizer::STORY);
        tokens
    }

    /// One complete training example: the instruction, the text that
    /// answers it, and the document boundary.
    pub fn to_training_tokens(&self, tokenizer: &Tokenizer, answer: &str) -> Vec<u32> {
        let mut tokens = self.to_prompt_tokens(tokenizer);
        tokens.extend(tokenizer.encode(answer));
        tokens.push(tokenizer::EOS);
        tokens
    }
}

/// Length buckets the model is trained on. See `Request::instruction`.
fn bucket_words(words: usize) -> &'static str {
    match words {
        0..=150 => "very-short",
        151..=400 => "short",
        401..=900 => "medium",
        901..=2000 => "long",
        _ => "very-long",
    }
}

/// Words per token, used to turn a word target into a token budget.
///
/// Measured against the trained vocabulary this ships with; it's an
/// estimate, and the generation loop counts actual words rather than
/// trusting it. It only has to be close enough that the budget isn't the
/// thing that ends a generation.
const TOKENS_PER_WORD: f32 = 1.6;

/// Parse a free-text prompt into a request.
///
/// Recognizes the shapes people actually type — "write a 700 word novel
/// about X", "a short screenplay set in Y", "500 words on Z in the style
/// of W" — and degrades gracefully: anything unrecognized becomes the
/// subject.
pub fn parse_prompt(prompt: &str) -> Request {
    let text = prompt.trim();
    let lower = text.to_lowercase();

    let form = detect_form(&lower);
    let target_words = detect_word_count(&lower);
    let (subject, reference) = split_subject_and_reference(text, &lower);

    Request { form, target_words, subject, reference }
}

fn detect_form(lower: &str) -> Form {
    // Screenplay first: "a screenplay about a novelist" is a screenplay.
    const SCREENPLAY: [&str; 8] =
        ["screenplay", "script", "film", "movie", "scene", "teleplay", "pilot", "episode"];
    const ALLEGORY: [&str; 6] = ["allegory", "parable", "fable", "myth", "dialogue", "meditation"];
    const NOVEL: [&str; 7] = ["novel", "story", "chapter", "novella", "prose", "tale", "fiction"];

    if SCREENPLAY.iter().any(|w| contains_word(lower, w)) {
        return Form::Screenplay;
    }
    // A reference to an allegory ("related to Plato's allegory of the
    // cave") shouldn't make the *form* an allegory when the prompt also
    // says "novel", so novel/story wins over the allegory words.
    if NOVEL.iter().any(|w| contains_word(lower, w)) {
        return Form::Novel;
    }
    if ALLEGORY.iter().any(|w| contains_word(lower, w)) {
        return Form::Allegory;
    }
    Form::Unspecified
}

/// Whole-word containment, so "scenery" doesn't read as "scene".
fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(found) = haystack[from..].find(needle) {
        let start = from + found;
        let end = start + needle.len();
        let before_ok = start == 0
            || !haystack.as_bytes()[start - 1].is_ascii_alphanumeric();
        // Allow a trailing "s"/"'s" so "scripts" and "Plato's" match.
        let after = haystack[end..].trim_start_matches('s').trim_start_matches("'s");
        let after_ok = after.is_empty() || !after.as_bytes()[0].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// A number immediately before "word"/"words" — "700 word novel",
/// "700-word", "about 1,200 words".
fn detect_word_count(lower: &str) -> Option<usize> {
    let bytes = lower.as_bytes();
    let mut at = 0usize;
    while let Some(found) = lower[at..].find("word") {
        let idx = at + found;
        // Walk back over the separator and then the digits.
        let mut end = idx;
        while end > 0 && matches!(bytes[end - 1], b' ' | b'-' | b'\t') {
            end -= 1;
        }
        let mut start = end;
        while start > 0 && (bytes[start - 1].is_ascii_digit() || bytes[start - 1] == b',') {
            start -= 1;
        }
        if start < end {
            let digits: String =
                lower[start..end].chars().filter(char::is_ascii_digit).collect();
            if let Ok(n) = digits.parse::<usize>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
        at = idx + 4;
    }
    None
}

/// Markers that introduce the thing a piece should draw on, longest
/// first so "in the style of" wins over "of".
const REFERENCE_MARKERS: [&str; 8] = [
    "related to",
    "in the style of",
    "inspired by",
    "based on",
    "echoing",
    "riffing on",
    "in the manner of",
    "after the fashion of",
];

/// Markers that introduce what a piece is about.
const SUBJECT_MARKERS: [&str; 5] = [" about ", " concerning ", " on the subject of ", " set in ", " featuring "];

fn split_subject_and_reference(text: &str, lower: &str) -> (String, Option<String>) {
    // Reference first, so it can be cut off the end of the subject.
    let mut reference: Option<String> = None;
    let mut reference_at = text.len();
    for marker in REFERENCE_MARKERS {
        if let Some(idx) = lower.find(marker) {
            if idx < reference_at {
                reference_at = idx;
                reference = Some(text[idx + marker.len()..].trim().trim_end_matches('.').to_string());
            }
        }
    }

    let head = &text[..reference_at];
    let head_lower = &lower[..reference_at];
    let subject = SUBJECT_MARKERS
        .iter()
        .filter_map(|m| head_lower.find(m).map(|i| i + m.len()))
        .min()
        .map(|i| head[i..].to_string())
        // No "about": the whole prompt is the subject, minus the
        // instruction verbiage that's already captured in the other
        // fields.
        .unwrap_or_else(|| strip_instruction_verbs(head));

    let subject = subject.trim().trim_end_matches(['.', ',']).trim().to_string();
    (subject, reference.filter(|r| !r.is_empty()))
}

/// Drop the leading "write me a 700 word novel" scaffolding, leaving the
/// rest of the prompt untouched.
///
/// Only the *leading* run is stripped, deliberately. Filtering these
/// words everywhere turns "two astronauts and a locked door" into "two
/// astronauts and locked door" — the subject is meant to read as
/// English, since it's what the model is conditioned on.
fn strip_instruction_verbs(text: &str) -> String {
    const DROP: [&str; 14] = [
        "write", "me", "a", "an", "the", "please", "generate", "create", "compose", "draft",
        "produce", "give", "word", "words",
    ];
    let is_scaffolding = |w: &str| {
        let clean = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        clean.is_empty()
            || clean.chars().all(|c| c.is_ascii_digit())
            || DROP.contains(&clean.as_str())
            || Form::from_str(&clean) != Form::Unspecified
    };
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = words.iter().position(|w| !is_scaffolding(w)).unwrap_or(words.len());
    words[start..].join(" ")
}

/// Tracks how much has been written and decides when to stop.
///
/// Extracted so the CPU and WebGPU generation paths share one copy of
/// the rule rather than two that can drift. The rule: run to the target
/// word count, then keep going only until the next sentence or paragraph
/// boundary so the piece ends on a finished sentence, with a hard
/// ceiling 40% over in case the model never produces one.
pub struct LengthGuard {
    target: Option<usize>,
    words: usize,
    /// Whether the previous piece ended mid-word, so the next one
    /// continues it rather than starting a new word.
    carry: bool,
    stopped_by_length: bool,
}

impl LengthGuard {
    pub fn new(target: Option<usize>) -> Self {
        Self { target, words: 0, carry: false, stopped_by_length: false }
    }

    /// Token budget to give the generator: enough to reach the target
    /// with room to find a boundary, and never unbounded.
    pub fn token_budget(&self) -> usize {
        let words = self.target.unwrap_or(600);
        ((words as f32 * TOKENS_PER_WORD * 1.6) as usize).clamp(32, 8192)
    }

    /// Account for a newly generated piece. Returns false when the
    /// generation should stop.
    pub fn observe(&mut self, piece: &str) -> bool {
        let (counted, ends_mid_word) = count_words(piece, self.carry);
        self.words += counted;
        self.carry = ends_mid_word;
        let Some(target) = self.target else { return true };
        if self.words >= target && ends_a_sentence(piece) {
            self.stopped_by_length = true;
            return false;
        }
        if self.words >= target * 7 / 5 {
            self.stopped_by_length = true;
            return false;
        }
        true
    }

    pub fn words(&self) -> usize {
        self.words
    }

    pub fn stopped_by_length(&self) -> bool {
        self.stopped_by_length
    }
}

/// How a generated answer turned out.
#[derive(Debug, Clone)]
pub struct Response {
    pub text: String,
    pub word_count: usize,
    pub tokens_generated: usize,
    pub stop_reason: StopReason,
}

/// Generate the answer to `request`, stopping near the requested length.
///
/// Length is enforced by the caller, not by the model: once the target
/// word count is reached the generation runs on only until the next
/// sentence or paragraph boundary, so it ends on a finished sentence
/// rather than mid-clause. A hard ceiling above the target catches the
/// case where the model never produces a boundary.
///
/// `on_progress` is called with each new piece of text and the running
/// word count; returning `false` stops early (a Stop button).
pub fn generate_response(
    weights: &ModelWeights,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    request: &Request,
    sampling: &SamplingConfig,
    on_progress: &mut dyn FnMut(&str, usize) -> bool,
) -> Response {
    let prompt_tokens = request.to_prompt_tokens(tokenizer);
    let mut guard = LengthGuard::new(request.target_words);
    let max_new_tokens = guard.token_budget();

    let (text, reason) = generate::generate_stream(
        weights,
        config,
        tokenizer,
        &prompt_tokens,
        max_new_tokens,
        sampling,
        &mut |piece, _| {
            let keep_going = guard.observe(piece);
            if !on_progress(piece, guard.words()) {
                return false;
            }
            keep_going
        },
    );

    Response {
        word_count: text.split_whitespace().count(),
        tokens_generated: tokenizer.encode(&text).len(),
        stop_reason: if guard.stopped_by_length() { StopReason::Caller } else { reason },
        text,
    }
}

/// Words starting in `piece`, given whether the previous piece ended
/// mid-word. Returns `(new words, ends mid-word)`.
fn count_words(piece: &str, continues_a_word: bool) -> (usize, bool) {
    if piece.is_empty() {
        return (0, continues_a_word);
    }
    let mut count = piece.split_whitespace().count();
    if continues_a_word && !piece.starts_with(char::is_whitespace) && count > 0 {
        count -= 1; // the first run finishes the previous word
    }
    let ends_mid_word = !piece.ends_with(char::is_whitespace);
    (count, ends_mid_word)
}

fn ends_a_sentence(piece: &str) -> bool {
    let trimmed = piece.trim_end();
    trimmed.ends_with(['.', '!', '?', '"']) || piece.contains("\n\n")
}

/// Build instruction-format training examples out of a document.
///
/// Each chunk of the source becomes `(instruction, text)` where the
/// instruction is the one that would have asked for that chunk: its
/// actual form (detected from the text's own shape), its actual length
/// bucket, and a subject drawn from its most distinctive words. This is
/// what teaches the model that the fields before `STORY` describe the
/// text after it — without anyone hand-writing a single instruction.
///
/// `words_per_chunk` sets roughly how long each example is; chunks are
/// cut at paragraph boundaries so an example never starts mid-sentence.
pub fn synthesize_examples(text: &str, words_per_chunk: usize) -> Vec<(Request, String)> {
    let mut out = Vec::new();
    for chunk in chunk_by_paragraph(text, words_per_chunk) {
        let words = chunk.split_whitespace().count();
        if words < 20 {
            continue;
        }
        let form = detect_form_from_text(&chunk);
        let subject = subject_from_text(&chunk);
        out.push((Request { form, target_words: Some(words), subject, reference: None }, chunk));
    }
    out
}

/// Split on blank lines, then glue paragraphs back together until each
/// group is about `words_per_chunk` long.
fn chunk_by_paragraph(text: &str, words_per_chunk: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_words = 0usize;
    for paragraph in text.split("\n\n") {
        let words = paragraph.split_whitespace().count();
        if words == 0 {
            continue;
        }
        if current_words > 0 && current_words + words > words_per_chunk {
            chunks.push(std::mem::take(&mut current));
            current_words = 0;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
        current_words += words;
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Which form a piece of *existing* text is, from its shape: any scene
/// heading (`INT.`/`EXT.`/...) means screenplay.
fn detect_form_from_text(text: &str) -> Form {
    if text.lines().any(screenplay::is_scene_heading) {
        Form::Screenplay
    } else {
        Form::Novel
    }
}

/// The most distinctive content words in a chunk, as a short subject
/// phrase.
///
/// Frequency minus a stopword list — not TF-IDF, because a single chunk
/// has no document collection to weigh against, and the point isn't
/// retrieval quality: it's giving the model a subject field whose words
/// genuinely do appear in the text after `STORY`, so the association is
/// learnable at all.
fn subject_from_text(text: &str) -> String {
    const STOPWORDS: [&str; 60] = [
        "the", "and", "for", "that", "with", "this", "from", "they", "have", "been", "were",
        "what", "when", "will", "would", "there", "their", "which", "about", "into", "than",
        "then", "them", "these", "those", "your", "you", "her", "his", "she", "him", "had",
        "has", "was", "are", "not", "but", "all", "one", "out", "who", "get", "got", "can",
        "him", "its", "our", "himself", "herself", "just", "like", "only", "over", "some",
        "such", "very", "well", "back", "down", "here",
    ];
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for word in text.split_whitespace() {
        let clean: String = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if clean.len() < 4 || STOPWORDS.contains(&clean.as_str()) {
            continue;
        }
        *counts.entry(clean).or_insert(0) += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    // Ties broken alphabetically so synthesis is deterministic.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(5);
    ranked.into_iter().map(|(w, _)| w).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_motivating_prompt() {
        let request =
            parse_prompt("Write a 700 word novel about two people in space related to Plato's allegory of the cave");
        assert_eq!(request.form, Form::Novel);
        assert_eq!(request.target_words, Some(700));
        assert_eq!(request.subject, "two people in space");
        assert_eq!(request.reference.as_deref(), Some("Plato's allegory of the cave"));
    }

    #[test]
    fn renders_a_stable_instruction_line() {
        let request =
            parse_prompt("Write a 700 word novel about two people in space related to Plato's allegory of the cave");
        assert_eq!(
            request.instruction(),
            "form=novel; words=medium; about: two people in space; echoing: Plato's allegory of the cave"
        );
    }

    #[test]
    fn word_counts_are_bucketed_not_verbatim() {
        // The model can't count, so it's trained on a register rather
        // than a number it would have to honour exactly.
        let short = parse_prompt("write a 100 word story about rain").instruction();
        let long = parse_prompt("write a 1500 word story about rain").instruction();
        assert!(short.contains("words=very-short"), "{short}");
        assert!(long.contains("words=long"), "{long}");
    }

    #[test]
    fn recognizes_screenplay_requests() {
        for prompt in [
            "write a scene where two astronauts argue",
            "a short film script about a lighthouse",
            "300 word screenplay set in a diner",
        ] {
            assert_eq!(parse_prompt(prompt).form, Form::Screenplay, "{prompt}");
        }
    }

    #[test]
    fn a_novel_about_an_allegory_is_still_a_novel() {
        let request = parse_prompt("a novel about a parable");
        assert_eq!(request.form, Form::Novel);
    }

    #[test]
    fn whole_word_matching_only() {
        // "scenery" must not read as "scene".
        let request = parse_prompt("describe the scenery of a mountain");
        assert_eq!(request.form, Form::Unspecified);
    }

    #[test]
    fn various_word_count_spellings() {
        assert_eq!(parse_prompt("a 700-word novel").target_words, Some(700));
        assert_eq!(parse_prompt("about 1,200 words on grief").target_words, Some(1200));
        assert_eq!(parse_prompt("write 250 words about a dog").target_words, Some(250));
        assert_eq!(parse_prompt("a novel about words").target_words, None);
    }

    #[test]
    fn a_bare_prompt_still_produces_a_usable_request() {
        let request = parse_prompt("two astronauts and a locked door");
        assert_eq!(request.form, Form::Unspecified);
        assert_eq!(request.target_words, None);
        assert_eq!(request.subject, "two astronauts and a locked door");
        assert_eq!(request.reference, None);
    }

    #[test]
    fn instruction_verbiage_is_stripped_from_a_subject() {
        let request = parse_prompt("Write me a 700 word novel: prisoners, firelight");
        assert!(!request.subject.contains("Write"), "{}", request.subject);
        assert!(request.subject.contains("prisoners"), "{}", request.subject);
    }

    #[test]
    fn prompt_tokens_are_bracketed_by_the_special_tokens() {
        let t = Tokenizer::byte_level();
        let request = parse_prompt("a 700 word novel about space");
        let tokens = request.to_prompt_tokens(&t);
        assert_eq!(tokens[0], tokenizer::BOS);
        assert_eq!(tokens[1], tokenizer::TASK);
        assert_eq!(*tokens.last().unwrap(), tokenizer::STORY);
        // The instruction survives a round trip, since it's what the
        // model has to read.
        let body = t.decode(&tokens[2..tokens.len() - 1]);
        assert_eq!(body, request.instruction());
    }

    #[test]
    fn training_tokens_end_the_document() {
        let t = Tokenizer::byte_level();
        let request = parse_prompt("a story about rain");
        let tokens = request.to_training_tokens(&t, "It rained.");
        assert_eq!(*tokens.last().unwrap(), tokenizer::EOS);
        assert!(t.decode(&tokens).ends_with("It rained."));
    }

    #[test]
    fn word_counting_handles_pieces_that_split_a_word() {
        // "hel" + "lo world" is two words, not three.
        let (a, mid_a) = count_words("hel", false);
        let (b, _) = count_words("lo world", mid_a);
        assert_eq!(a + b, 2);
    }

    #[test]
    fn synthesis_labels_a_screenplay_chunk_as_a_screenplay() {
        let script = "INT. CAVE - DAY\n\nFirelight on the wall.\n\nSOCRATES\nWhat do you see?\n\n"
            .repeat(6);
        let examples = synthesize_examples(&script, 60);
        assert!(!examples.is_empty());
        assert!(
            examples.iter().all(|(r, _)| r.form == Form::Screenplay),
            "screenplay chunks should be labelled as such"
        );
        assert!(examples.iter().all(|(r, _)| r.target_words.is_some()));
    }

    #[test]
    fn synthesis_labels_prose_as_a_novel_and_finds_a_subject() {
        let prose = "The prisoners had been in the cave since childhood. The prisoners saw \
                     only shadows. Shadows were, to the prisoners, the whole of the world.\n\n"
            .repeat(4);
        let examples = synthesize_examples(&prose, 80);
        assert!(!examples.is_empty());
        let (request, text) = &examples[0];
        assert_eq!(request.form, Form::Novel);
        assert!(request.subject.contains("prisoners"), "subject was {:?}", request.subject);
        // A synthesized subject must actually describe the text it's
        // paired with, or there's nothing for the model to learn.
        for word in request.subject.split(", ") {
            assert!(text.to_lowercase().contains(word), "subject word {word:?} isn't in the text");
        }
    }

    #[test]
    fn synthesis_is_deterministic() {
        let prose = "Ships passed the station. The station turned. Ships and shadows.\n\n".repeat(8);
        assert_eq!(synthesize_examples(&prose, 50), synthesize_examples(&prose, 50));
    }

    #[test]
    fn chunking_respects_the_word_budget_and_loses_nothing() {
        let text = (1..=20).map(|i| format!("paragraph {i} with several words in it")).collect::<Vec<_>>().join("\n\n");
        let chunks = chunk_by_paragraph(&text, 30);
        assert!(chunks.len() > 1, "should have split");
        let recombined: String =
            chunks.iter().flat_map(|c| c.split_whitespace()).collect::<Vec<_>>().join(" ");
        let original: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(recombined, original, "chunking dropped or reordered text");
    }

    #[test]
    fn generation_stops_near_the_requested_length() {
        // A real length check needs a model that emits words, so this
        // uses the byte-level tokenizer and a tiny model and only
        // asserts the ceiling — the exact stopping point depends on
        // whether the model ever produces a sentence boundary.
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 16,
            num_heads: 2,
            num_kv_heads: 1,
            context_len: 64,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 3);
        let t = Tokenizer::byte_level();
        let request = Request {
            form: Form::Novel,
            target_words: Some(20),
            subject: "space".to_string(),
            reference: None,
        };
        let response = generate_response(
            &weights,
            &config,
            &t,
            &request,
            &SamplingConfig { seed: 1, ..Default::default() },
            &mut |_, _| true,
        );
        assert!(
            response.word_count <= 20 * 7 / 5 + 2,
            "overshot the hard ceiling: {} words",
            response.word_count
        );
    }

    #[test]
    fn a_progress_callback_can_stop_generation() {
        let config = ModelConfig {
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            context_len: 32,
            local_window: 32,
            ..Default::default()
        };
        let weights = ModelWeights::init(&config, 2);
        let request = parse_prompt("a 500 word story about rain");
        let mut pieces = 0;
        let response = generate_response(
            &weights,
            &config,
            &Tokenizer::byte_level(),
            &request,
            &SamplingConfig::default(),
            &mut |_, _| {
                pieces += 1;
                pieces < 3
            },
        );
        assert!(pieces <= 3);
        assert!(response.tokens_generated < 100);
    }
}
