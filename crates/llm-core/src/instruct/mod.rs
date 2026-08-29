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
//! Everything here is deliberately mechanical. Parsing (`parse.rs`) is a
//! keyword scan, not a language model, so a prompt that doesn't match any
//! pattern still produces a usable request (form unspecified, no length
//! target, the whole prompt as the subject) rather than an error.

mod length_guard;
mod parse;
mod synthesize;

pub use length_guard::{generate_response, LengthGuard, Response, ResponseSession};
pub use synthesize::synthesize_examples;

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

/// Parse a free-text prompt into a request.
///
/// Recognizes the shapes people actually type — "write a 700 word novel
/// about X", "a short screenplay set in Y", "500 words on Z in the style
/// of W" — and degrades gracefully: anything unrecognized becomes the
/// subject.
pub fn parse_prompt(prompt: &str) -> Request {
    let text = prompt.trim();
    // ASCII-only casing, not `to_lowercase()`: every marker below is
    // ASCII, and unlike full Unicode case-folding this never changes a
    // character's byte length (e.g. 'İ' U+0130 grows from 2 to 3 UTF-8
    // bytes under `to_lowercase()`), so byte offsets found in `lower`
    // stay valid indices into `text`.
    let lower = text.to_ascii_lowercase();

    let form = parse::detect_form(&lower);
    let target_words = parse::detect_word_count(&lower);
    let (subject, reference) = parse::split_subject_and_reference(text, &lower);

    Request { form, target_words, subject, reference }
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
    fn does_not_panic_on_unicode_that_changes_length_when_lowercased() {
        // 'İ' (U+0130) is 2 UTF-8 bytes but its full-Unicode lowercase
        // mapping "i̇" is 3 bytes — `to_lowercase()` would desync the
        // byte offsets `split_subject_and_reference` reuses across
        // `text` and its lowercased copy, panicking on a non-boundary
        // slice. Regression for that: must not panic, and must not
        // silently misalign the split either.
        let request = parse_prompt("İ");
        assert_eq!(request.subject, "İ");

        let request = parse_prompt("İ story about love");
        assert_eq!(request.subject, "love");
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
}
