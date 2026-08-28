//! Heuristic screenplay structure detection: scene headings and character
//! cues, recognized from plain text with line-shape heuristics rather than
//! a real grammar. This is deliberately *not* machine learning — standard
//! screenplay formatting (`INT. KITCHEN - DAY`, an ALL-CAPS name before a
//! dialogue block) is regular enough that plain string matching works,
//! without training a tagger or asking the user to annotate anything.
//! It will misfire occasionally on unusual formatting or on prose that
//! happens to contain a short ALL-CAPS line — treat its output as a
//! useful hint, not ground truth.
//!
//! Used by `mix.rs` to classify a source as a screenplay for the training
//! plan's corpus-mix advice.

// Dotted forms first: `.find()` below returns the first match, and a bare
// form (e.g. "EXT") is a prefix of its own dotted form ("EXT."), so the
// dotted, more-specific variants have to be checked first or they'd never
// be reached. The bare forms exist because plenty of real shooting
// scripts (especially older or TV/documentary ones) drop the period.
const SLUGLINE_PREFIXES: &[&str] = &[
    "INT./EXT.",
    "EXT./INT.",
    "INT/EXT.",
    "EXT/INT.",
    "INT.",
    "EXT.",
    "EST.",
    "I/E.",
    "INT/EXT",
    "EXT/INT",
    "INT",
    "EXT",
    "EST",
    "I/E",
];

const TRANSITION_WORDS: &[&str] = &[
    "CUT TO",
    "FADE IN",
    "FADE OUT",
    "FADE TO",
    "FADE TO BLACK",
    "DISSOLVE TO",
    "SMASH CUT TO",
    "MATCH CUT TO",
    "JUMP CUT TO",
    "TITLE CARD",
    "THE END",
    "SUPER",
    "V.O.",
    "O.S.",
    "CONT'D",
    "CONTINUED",
];

/// All-caps lines that sit right above dialogue but name a moment, a
/// shot, or an edit rather than a speaker. These clear every shape check
/// (short, all caps, a lowercase line under them), so they have to be
/// named explicitly.
const NON_CHARACTER_CUES: &[&str] = &[
    "BACK TO SCENE",
    "BEAT",
    "CONTINUOUS",
    "DAY",
    "DELETED",
    "END INTERCUT",
    "END MONTAGE",
    "FLASHBACK",
    "LATER",
    "MOMENTS LATER",
    "MONTAGE",
    "MORNING",
    "NIGHT",
    "OMITTED",
    "PAUSE",
    "SAME",
    "SILENCE",
    "SYNOPSIS",
    "VARIOUS",
];

/// Words a shot or camera direction starts with. A line beginning with
/// one describes the frame ("ANGLE ON DELILAH", "CLOSE ON LOGAN", "BACK
/// TO LAURA") and is followed by prose, so the dialogue check alone lets
/// every one of them through as a "character".
const SHOT_PREFIXES: &[&str] = &[
    "ANGLE",
    "ANOTHER ANGLE",
    "BACK TO",
    "CLOSE ",
    "CLOSER ",
    "CU ",
    "END ON",
    "EXTREME ",
    "FULL SHOT",
    "HIGH ANGLE",
    "INSERT",
    "INTERCUT",
    "LOW ANGLE",
    "MACRO ",
    "MEDIUM ",
    "NEW ANGLE",
    "ON ",
    "PANNING",
    "POV",
    "REVEAL",
    "REVERSE",
    "SERIES OF",
    "TIGHT ",
    "TILT",
    "WIDE",
];

fn strip_parenthetical(s: &str) -> &str {
    match s.find('(') {
        Some(idx) => s[..idx].trim_end(),
        None => s,
    }
}

/// Finds the matching slugline prefix, requiring a non-alphanumeric (or
/// end-of-string) character right after it — without this, a bare prefix
/// like "EXT" would also match the start of "EXTRA" or "EXTERIOR".
fn matches_slugline_prefix(upper: &str) -> Option<&'static str> {
    SLUGLINE_PREFIXES
        .iter()
        .find(|p| upper.starts_with(**p) && upper[p.len()..].chars().next().map_or(true, |c| !c.is_ascii_alphanumeric()))
        .copied()
}

/// Whether `line` looks like a scene slugline (`INT.`/`EXT.`/... prefix).
pub fn is_scene_heading(line: &str) -> bool {
    let trimmed = line.trim();
    let upper = trimmed.to_ascii_uppercase();
    matches_slugline_prefix(&upper).is_some()
}

/// Whether a single word looks like a shot/section/date code rather than
/// part of a person's name: either all digits ("2001"), or a short (1-3
/// letter) prefix glued directly to digits with nothing else ("A1", "A12",
/// "B3") - the standard shorthand shooting scripts use for take/section
/// numbers.
fn looks_like_code_word(word: &str) -> bool {
    let mut chars = word.chars().peekable();
    let mut letters = 0;
    while let Some(&c) = chars.peek() {
        if c.is_ascii_uppercase() {
            letters += 1;
            chars.next();
        } else {
            break;
        }
    }
    let rest: String = chars.collect();
    (letters == 0 && !word.is_empty() && word.chars().all(|c| c.is_ascii_digit()))
        || ((1..=3).contains(&letters) && !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Whether `line` looks like a character cue (an ALL-CAPS name, optionally
/// with a `(V.O.)`/`(CONT'D)`-style parenthetical, on its own line).
pub fn is_character_cue(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 40 {
        return false;
    }
    if is_scene_heading(trimmed) || trimmed.ends_with(':') {
        return false;
    }
    let name_part = strip_parenthetical(trimmed).trim();
    if name_part.is_empty() {
        return false;
    }
    // "X - Y" is the slugline location/time separator - a real character
    // name is never a hyphen-flanked-by-spaces pair of phrases, so this
    // catches heading-like section/segment titles that don't have an
    // INT./EXT. prefix to match (e.g. a documentary's "VIEWS OF AFRICAN
    // DRYLANDS - DROUGHT").
    if [" - ", " \u{2013} ", " \u{2014} "].iter().any(|sep| name_part.contains(sep)) {
        return false;
    }
    // Real character names are essentially always 1-3 words, and never
    // contain a bare code-like word ("2001", "A1", "A12") - this catches
    // multi-word shot/segment descriptions and date/section markers that
    // the hyphen check above doesn't (they don't all use " - ").
    let words: Vec<&str> = name_part.split_whitespace().collect();
    if words.is_empty() || words.len() > 3 || words.iter().any(|w| looks_like_code_word(w)) {
        return false;
    }
    let looks_like_shouting_caps = name_part
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, ' ' | '.' | '\'' | '-'));
    if !looks_like_shouting_caps || !name_part.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    if TRANSITION_WORDS.contains(&name_part) || NON_CHARACTER_CUES.contains(&name_part) {
        return false;
    }
    // A cue is a name, so it doesn't end in a sentence's punctuation.
    // Action written in caps does ("ALTITUDE.", "HIS HAND."), and that one
    // check removes most of what a caps-heavy shooting script otherwise
    // contributes to the character list.
    if name_part.ends_with('.') || name_part.ends_with(',') || name_part.ends_with('-') {
        return false;
    }
    !SHOT_PREFIXES.iter().any(|p| name_part.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_scene_headings() {
        assert!(is_scene_heading("INT. KITCHEN - DAY"));
        assert!(is_scene_heading("  ext. mars colony - night  "));
        assert!(is_scene_heading("INT./EXT. CAR - MOVING - NIGHT"));
        assert!(!is_scene_heading("He walked into the kitchen."));
        assert!(!is_scene_heading("CYPHER"));
    }

    #[test]
    fn detects_bare_prefixes_without_a_period() {
        assert!(is_scene_heading("EXT THE STREAM - THE OTHERS"));
        assert!(is_scene_heading("INT KITCHEN"));
        // A word boundary is required: "EXTRA"/"INTERIOR" must not match
        // the bare "EXT"/"INT" prefixes just because they start with them.
        assert!(!is_scene_heading("EXTRA CREDIT SCENE"));
        assert!(!is_scene_heading("INTERIOR DESIGN MAGAZINE"));
    }

    #[test]
    fn detects_character_cues_with_and_without_parenthetical() {
        assert!(is_character_cue("CYPHER"));
        assert!(is_character_cue("        CYPHER (V.O.)"));
        assert!(is_character_cue("MARY-ANNE O'BRIEN"));
    }

    #[test]
    fn rejects_transitions_and_scene_headings_as_cues() {
        assert!(!is_character_cue("CUT TO:"));
        assert!(!is_character_cue("FADE OUT"));
        assert!(!is_character_cue("CONTINUED"));
        assert!(!is_character_cue("INT. KITCHEN - DAY"));
    }

    #[test]
    fn rejects_heading_style_segment_titles_without_a_slugline_prefix() {
        // Real shooting-script false positives reported from a nature
        // documentary: all-caps segment titles that use the same
        // " - " location/time separator sluglines use, but without an
        // INT./EXT. prefix to catch them as a heading in the first place.
        assert!(!is_character_cue("VIEWS OF AFRICAN DRYLANDS - DROUGHT"));
        assert!(!is_character_cue("EXT PARCHED COUNTRYSIDE - THE LION"));
    }

    #[test]
    fn rejects_bare_section_codes_and_multi_word_shot_descriptions() {
        // More false positives from the same documentary: bare take/section
        // codes (no hyphen, so the check above doesn't catch them) and
        // longer shot descriptions that happen to be followed by real
        // narration prose, which the dialogue-lookahead check alone can't
        // tell apart from a genuine character cue.
        assert!(!is_character_cue("A12"));
        assert!(!is_character_cue("A1"));
        assert!(!is_character_cue("YEAR 2001"));
        assert!(!is_character_cue("EARTH FROM 200 MILES UP NARRATOR"));
        // Still accepts ordinary short names.
        assert!(is_character_cue("JANE"));
        assert!(is_character_cue("DR SMITH"));
    }

    #[test]
    fn rejects_lowercase_and_long_lines() {
        assert!(!is_character_cue("He caught the bus."));
        assert!(!is_character_cue("THIS LINE IS DELIBERATELY WAY TOO LONG TO BE A CHARACTER CUE"));
    }

    #[test]
    fn a_cue_with_leading_whitespace_is_still_recognized() {
        assert!(is_character_cue("                    CYPHER (V.O.)"));
    }
}
