//! The hand-rolled NLP that turns free text into a [`super::Form`] and a
//! subject/reference split. A keyword scan, not a language model: a
//! prompt that doesn't match any pattern still produces a usable split
//! (see `super::parse_prompt`) rather than an error.

use super::Form;

pub(super) fn detect_form(lower: &str) -> Form {
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
pub(super) fn detect_word_count(lower: &str) -> Option<usize> {
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

pub(super) fn split_subject_and_reference(text: &str, lower: &str) -> (String, Option<String>) {
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
