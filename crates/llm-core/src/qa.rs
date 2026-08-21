//! Rule-based QA pass over generated text — the "script critic" from the
//! architecture discussion, scoped down to what's actually feasible
//! without a second trained model or an LLM judge: cheap heuristic
//! checks a human can read and act on. Not a quality judgement, just
//! things worth a human's attention.

use crate::screenplay::{self, StoryState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Purely informational — not necessarily a problem (e.g. "this
    /// introduces a new character"), just worth noting.
    Info,
    /// Something that looks like it might actually be wrong.
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QaNote {
    pub severity: Severity,
    pub message: String,
}

fn unbalanced_parens(text: &str) -> Option<QaNote> {
    let open = text.matches('(').count();
    let close = text.matches(')').count();
    if open != close {
        Some(QaNote {
            severity: Severity::Warning,
            message: format!(
                "Unbalanced parentheses ({open} open vs {close} close) — often means a \
                 parenthetical like \"(V.O.)\" got cut off mid-generation."
            ),
        })
    } else {
        None
    }
}

fn new_characters(text: &str, known: &StoryState) -> Option<QaNote> {
    let generated = screenplay::extract_story_state(text);
    let new_names: Vec<&str> = generated
        .characters
        .iter()
        .filter(|c| !known.characters.contains(c))
        .map(String::as_str)
        .collect();
    if new_names.is_empty() {
        None
    } else {
        Some(QaNote {
            severity: Severity::Info,
            message: format!("Introduces character(s) not seen in any source yet: {}.", new_names.join(", ")),
        })
    }
}

fn length_vs_target(text: &str, target_word_count: Option<usize>) -> Option<QaNote> {
    let target = target_word_count?;
    if target == 0 {
        return None;
    }
    let actual = text.split_whitespace().count();
    let ratio = actual as f32 / target as f32;
    if !(0.5..=2.0).contains(&ratio) {
        Some(QaNote {
            severity: Severity::Info,
            message: format!("Generated {actual} words against a rough target of {target} — noticeably off."),
        })
    } else {
        None
    }
}

/// Flags a line that repeats 3+ times consecutively — a common failure
/// mode for small models (degenerate repetition loops), especially at
/// low temperature or early in training.
fn repetition_loop(text: &str) -> Option<QaNote> {
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut run_len = 1usize;
    for i in 1..lines.len() {
        if lines[i] == lines[i - 1] {
            run_len += 1;
            if run_len >= 3 {
                let preview: String = lines[i].chars().take(60).collect();
                return Some(QaNote {
                    severity: Severity::Warning,
                    message: format!(
                        "The line \"{preview}\" repeats {run_len} times in a row — looks like a \
                         degenerate repetition loop. Try a higher temperature or more training."
                    ),
                });
            }
        } else {
            run_len = 1;
        }
    }
    None
}

/// Runs every heuristic check against `text` (freshly generated output),
/// comparing against `known_state` (typically `Corpus::story_state()`,
/// i.e. what's established across the training sources) and an optional
/// rough target word count. Returns notes in a fixed, stable order;
/// empty means nothing stood out.
pub fn check_generated(text: &str, known_state: &StoryState, target_word_count: Option<usize>) -> Vec<QaNote> {
    [
        unbalanced_parens(text),
        repetition_loop(text),
        new_characters(text, known_state),
        length_vs_target(text, target_word_count),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_parens_produce_no_note() {
        assert!(unbalanced_parens("CYPHER (V.O.)\nHello.").is_none());
    }

    #[test]
    fn unbalanced_parens_are_flagged() {
        let note = unbalanced_parens("CYPHER (V.O.\nHello.").unwrap();
        assert_eq!(note.severity, Severity::Warning);
    }

    #[test]
    fn known_characters_are_not_flagged_as_new() {
        let known = StoryState { characters: vec!["JANE".to_string()], ..Default::default() };
        assert!(new_characters("JANE\nHello.", &known).is_none());
    }

    #[test]
    fn unseen_characters_are_flagged_as_info() {
        let known = StoryState { characters: vec!["JANE".to_string()], ..Default::default() };
        let note = new_characters("JOHN\nHi Jane.", &known).unwrap();
        assert_eq!(note.severity, Severity::Info);
        assert!(note.message.contains("JOHN"));
    }

    #[test]
    fn length_close_to_target_is_not_flagged() {
        let text = "one two three four five six seven eight nine ten";
        assert!(length_vs_target(text, Some(10)).is_none());
    }

    #[test]
    fn length_far_from_target_is_flagged() {
        let text = "one two three";
        assert!(length_vs_target(text, Some(100)).is_some());
    }

    #[test]
    fn no_target_means_no_length_check() {
        assert!(length_vs_target("one two three", None).is_none());
    }

    #[test]
    fn repeated_line_three_times_is_flagged() {
        let text = "He walked in.\nHe walked in.\nHe walked in.\nHe sat down.";
        let note = repetition_loop(text).unwrap();
        assert_eq!(note.severity, Severity::Warning);
    }

    #[test]
    fn varied_text_is_not_flagged_as_repetition() {
        let text = "He walked in.\nShe looked up.\nThey spoke quietly.";
        assert!(repetition_loop(text).is_none());
    }

    #[test]
    fn check_generated_returns_empty_for_clean_text() {
        let known = StoryState::default();
        let notes = check_generated("A calm, ordinary scene with no issues.", &known, None);
        assert!(notes.is_empty());
    }

    #[test]
    fn check_generated_collects_multiple_notes() {
        let known = StoryState::default();
        let text = "JOHN (V.O.\nHe walked in.\nHe walked in.\nHe walked in.";
        let notes = check_generated(text, &known, None);
        assert!(notes.len() >= 2, "expected both a parens and a repetition note, got {notes:?}");
    }
}
