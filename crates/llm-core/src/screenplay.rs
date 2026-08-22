//! Heuristic screenplay structure parser: scene headings and character
//! cues, extracted from plain text with line-shape heuristics rather than
//! a real grammar. This is deliberately *not* machine learning — standard
//! screenplay formatting (`INT. KITCHEN - DAY`, an ALL-CAPS name before a
//! dialogue block) is regular enough that plain string matching gets a
//! useful character list, location list, and scene count for free,
//! without training a tagger or asking the user to annotate anything.
//! It will misfire occasionally on unusual formatting or on prose that
//! happens to contain a short ALL-CAPS line — treat its output as a
//! useful hint, not ground truth.

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneHeading {
    pub raw: String,
    pub location: String,
    pub time: Option<String>,
}

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

pub fn parse_scene_heading(line: &str) -> Option<SceneHeading> {
    let trimmed = line.trim();
    let upper = trimmed.to_ascii_uppercase();
    let prefix = matches_slugline_prefix(&upper)?;
    let rest = &trimmed[prefix.len()..];
    let (location, time) = split_location_time(rest.trim_start());
    Some(SceneHeading { raw: trimmed.to_string(), location: location.trim().to_string(), time })
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

fn split_location_time(rest: &str) -> (&str, Option<String>) {
    for sep in [" - ", " \u{2013} ", " \u{2014} "] {
        if let Some(idx) = rest.find(sep) {
            let (loc, tail) = rest.split_at(idx);
            let time = tail[sep.len()..].trim();
            return (loc, if time.is_empty() { None } else { Some(time.to_string()) });
        }
    }
    (rest, None)
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
    // "X - Y" is the slugline location/time separator (split_location_time
    // uses the same three dashes) - a real character name is never a
    // hyphen-flanked-by-spaces pair of phrases, so this catches heading-like
    // section/segment titles that don't have an INT./EXT. prefix to match
    // (e.g. a documentary's "VIEWS OF AFRICAN DRYLANDS - DROUGHT").
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

/// The bare character name (parenthetical stripped) if `line` is a cue.
pub fn character_name(line: &str) -> Option<String> {
    if !is_character_cue(line) {
        return None;
    }
    Some(strip_parenthetical(line.trim()).trim().to_string())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoryState {
    pub characters: Vec<String>,
    pub locations: Vec<String>,
    pub scene_count: usize,
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.iter().any(|existing| existing == &value) {
        list.push(value);
    }
}

/// Whether `line` reads as ordinary dialogue/prose rather than another
/// all-caps cue, heading, or section marker.
fn looks_like_dialogue(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !is_scene_heading(trimmed) && trimmed.chars().any(|c| c.is_ascii_lowercase())
}

/// Scans `text` line by line, collecting scene headings and character
/// cues in first-seen order.
///
/// Beyond looking like a name, a candidate cue has to clear three things,
/// because a caps-heavy shooting script is full of lines that don't:
///
/// * a blank line above it - a cue opens a dialogue block, whereas a
///   wryly ("THOUGHTFULLY") or a caps fragment inside an action paragraph
///   sits directly under another line;
/// * dialogue below it - the next non-empty line has to contain a
///   lowercase letter, so runs of all-caps section/shot codes don't sweep
///   each other up;
/// * a second appearance in the same text - people in a screenplay speak
///   more than once, one-off caps fragments ("CCCP MARKINGS") do not.
///   This does drop characters with a single line, which is the right way
///   round: a short list that is right beats a long list that is noise.
pub fn extract_story_state(text: &str) -> StoryState {
    let mut state = StoryState::default();
    let lines: Vec<&str> = text.lines().collect();
    let mut cues: Vec<String> = Vec::new();
    for (i, &line) in lines.iter().enumerate() {
        if let Some(heading) = parse_scene_heading(line) {
            state.scene_count += 1;
            if !heading.location.is_empty() {
                push_unique(&mut state.locations, heading.location);
            }
            continue;
        }
        if i > 0 && !lines[i - 1].trim().is_empty() {
            continue;
        }
        let Some(name) = character_name(line) else { continue };
        let next_line = lines[i + 1..].iter().find(|l| !l.trim().is_empty());
        if !next_line.is_some_and(|l| looks_like_dialogue(l)) {
            continue;
        }
        cues.push(name);
    }
    for (i, name) in cues.iter().enumerate() {
        if cues[..i].contains(name) {
            push_unique(&mut state.characters, name.clone());
        }
    }
    state
}

impl StoryState {
    pub fn merge(&mut self, other: &StoryState) {
        for c in &other.characters {
            push_unique(&mut self.characters, c.clone());
        }
        for l in &other.locations {
            push_unique(&mut self.locations, l.clone());
        }
        self.scene_count += other.scene_count;
    }

    /// A short text block summarizing what's tracked so far, meant to be
    /// prepended to a generation prompt as a plain, non-neural reminder of
    /// who/where already exists in the story. Empty if nothing was found.
    pub fn as_prompt_preamble(&self) -> String {
        if self.characters.is_empty() && self.locations.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        if !self.characters.is_empty() {
            out.push_str("Characters so far: ");
            out.push_str(&self.characters.join(", "));
            out.push('\n');
        }
        if !self.locations.is_empty() {
            out.push_str("Locations so far: ");
            out.push_str(&self.locations.join(", "));
            out.push('\n');
        }
        out
    }
}

/// Splits text into scene-level chunks, one per detected scene heading
/// (any text before the first heading becomes its own leading chunk).
/// Used by the retrieval index so "find similar scenes" operates at a
/// meaningful granularity instead of arbitrary token windows.
pub fn split_into_scenes(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if is_scene_heading(line) && !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATRIX_EXCERPT: &str = "\
The entire screen fills with racing columns of numbers.
Shimmering like green-electric rivets, they rush at a 10-
digit phone number in the top corner.

                    CYPHER (V.O.)
          He caught the northbound Howard
          line. Got off at Sheridan.
          Stopped at 7-11.  Purchased six-
          pack of beer and a box of Captain
          Crunch.  Returned home.

The area code is identified.  The first three numbers
suddenly fixed, leaving only seven flowing columns.

                    CYPHER (V.O.)
          Nothing else. Not a thing.";

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
    fn parses_location_and_time() {
        let h = parse_scene_heading("INT. KITCHEN - DAY").unwrap();
        assert_eq!(h.location, "KITCHEN");
        assert_eq!(h.time.as_deref(), Some("DAY"));

        let h2 = parse_scene_heading("EXT. MARS COLONY").unwrap();
        assert_eq!(h2.location, "MARS COLONY");
        assert_eq!(h2.time, None);

        let h3 = parse_scene_heading("INT./EXT. CAR - MOVING - NIGHT").unwrap();
        assert_eq!(h3.location, "CAR");
        assert_eq!(h3.time.as_deref(), Some("MOVING - NIGHT"));
    }

    #[test]
    fn detects_character_cues_with_and_without_parenthetical() {
        assert_eq!(character_name("CYPHER"), Some("CYPHER".to_string()));
        assert_eq!(character_name("        CYPHER (V.O.)"), Some("CYPHER".to_string()));
        assert_eq!(character_name("MARY-ANNE O'BRIEN"), Some("MARY-ANNE O'BRIEN".to_string()));
    }

    #[test]
    fn rejects_transitions_and_scene_headings_as_cues() {
        assert_eq!(character_name("CUT TO:"), None);
        assert_eq!(character_name("FADE OUT"), None);
        assert_eq!(character_name("CONTINUED"), None);
        assert_eq!(character_name("INT. KITCHEN - DAY"), None);
    }

    #[test]
    fn rejects_heading_style_segment_titles_without_a_slugline_prefix() {
        // Real shooting-script false positives reported from a nature
        // documentary: all-caps segment titles that use the same
        // " - " location/time separator sluglines use, but without an
        // INT./EXT. prefix to catch them as a heading in the first place.
        assert_eq!(character_name("VIEWS OF AFRICAN DRYLANDS - DROUGHT"), None);
        assert_eq!(character_name("EXT PARCHED COUNTRYSIDE - THE LION"), None);
    }

    #[test]
    fn rejects_bare_section_codes_and_multi_word_shot_descriptions() {
        // More false positives from the same documentary: bare take/section
        // codes (no hyphen, so the check above doesn't catch them) and
        // longer shot descriptions that happen to be followed by real
        // narration prose, which the dialogue-lookahead check alone can't
        // tell apart from a genuine character cue.
        assert_eq!(character_name("A12"), None);
        assert_eq!(character_name("A1"), None);
        assert_eq!(character_name("YEAR 2001"), None);
        assert_eq!(character_name("EARTH FROM 200 MILES UP NARRATOR"), None);
        // Still accepts ordinary short names.
        assert_eq!(character_name("JANE"), Some("JANE".to_string()));
        assert_eq!(character_name("DR SMITH"), Some("DR SMITH".to_string()));
    }

    #[test]
    fn rejects_lowercase_and_long_lines() {
        assert_eq!(character_name("He caught the bus."), None);
        assert_eq!(
            character_name("THIS LINE IS DELIBERATELY WAY TOO LONG TO BE A CHARACTER CUE"),
            None
        );
    }

    #[test]
    fn extracts_story_state_from_matrix_excerpt() {
        let state = extract_story_state(MATRIX_EXCERPT);
        assert_eq!(state.characters, vec!["CYPHER".to_string()]);
        assert_eq!(state.scene_count, 0); // no scene heading in this excerpt
    }

    #[test]
    fn extracts_scene_headings_and_multiple_characters() {
        let script = "\
INT. KITCHEN - DAY

JANE
Where were you?

JOHN (O.S.)
Out.

EXT. GARDEN - NIGHT

JANE
It's cold.

JOHN
Then go inside.";
        let state = extract_story_state(script);
        assert_eq!(state.scene_count, 2);
        assert_eq!(state.characters, vec!["JANE".to_string(), "JOHN".to_string()]);
        assert_eq!(state.locations, vec!["KITCHEN".to_string(), "GARDEN".to_string()]);
    }

    #[test]
    fn nature_documentary_excerpt_yields_no_characters() {
        let script = "\
TITLE PART I

AFRICA

A1

VIEWS OF AFRICAN DRYLANDS - DROUGHT

The sun beats down on a cracked, endless plain.

A2

CONTINUED

EXT THE STREAM - THE OTHERS

A few animals gather at what remains of the water.

A3

EXT AFRICAN PLAIN - HERBIVORES

Herds move slowly across the grassland.";
        let state = extract_story_state(script);
        assert!(state.characters.is_empty(), "expected no characters, got {:?}", state.characters);
        // The bare EXT prefix (no period) is still recognized as a heading.
        assert_eq!(state.scene_count, 2);
    }

    #[test]
    fn rejects_shooting_script_section_codes_not_followed_by_dialogue() {
        let script = "\
TITLE PART I

AFRICA

A1

VIEWS OF AFRICAN DRYLANDS - DROUGHT

A2

CONTINUED";
        let state = extract_story_state(script);
        assert!(state.characters.is_empty(), "expected no characters, got {:?}", state.characters);
    }

    #[test]
    fn accepts_cue_followed_by_dialogue_rejects_cue_followed_by_another_cue() {
        let script = "\
JANE
Hello there.

BOB

ALICE
Hi Bob.

JANE
Still here.

ALICE
So I see.";
        let state = extract_story_state(script);
        // BOB is immediately followed by another all-caps line (ALICE), so
        // it reads as two consecutive section-style markers rather than a
        // character cue followed by dialogue, and is rejected.
        assert_eq!(state.characters, vec!["JANE".to_string(), "ALICE".to_string()]);
    }

    #[test]
    fn rejects_caps_action_wrylies_and_shot_lines() {
        // Every stray line here showed up as a "character" before these
        // rules existed, from a real caps-heavy shooting script.
        let script = "\
FLOYD reaches for the handset, checking the readout.

ALTITUDE.

The needle drops.

HAND.

His fingers close around it.

FLOYD
Bring us in.
THOUGHTFULLY.
It has been a long trip.

ANGLE ON FLOYD

He waits for the docking light.

PAUSE

The light comes on.

FLOYD
There it is.";
        let state = extract_story_state(script);
        assert_eq!(state.characters, vec!["FLOYD".to_string()]);
    }

    #[test]
    fn one_off_caps_fragments_are_dropped() {
        let script = "\
CCCP MARKINGS

The hull is stencilled along its length.

SMYSLOV
Good to see you.

SMYSLOV
Sit, please.";
        let state = extract_story_state(script);
        assert_eq!(state.characters, vec!["SMYSLOV".to_string()]);
    }

    #[test]
    fn merge_deduplicates() {
        let mut a = extract_story_state("INT. KITCHEN - DAY\n\nJANE\nHi.\n\nJANE\nStill here.");
        let b = extract_story_state("EXT. GARDEN - NIGHT\n\nJANE\nBye.\n\nJOHN\nWhat?\n\nJOHN\nWait.");
        a.merge(&b);
        assert_eq!(a.characters, vec!["JANE".to_string(), "JOHN".to_string()]);
        assert_eq!(a.locations, vec!["KITCHEN".to_string(), "GARDEN".to_string()]);
        assert_eq!(a.scene_count, 2);
    }

    #[test]
    fn prompt_preamble_is_empty_when_nothing_found() {
        assert_eq!(StoryState::default().as_prompt_preamble(), "");
    }

    #[test]
    fn prompt_preamble_lists_characters_and_locations() {
        let state = extract_story_state("INT. KITCHEN - DAY\n\nJANE\nHi.\n\nJANE\nBye.");
        let preamble = state.as_prompt_preamble();
        assert!(preamble.contains("Characters so far: JANE"));
        assert!(preamble.contains("Locations so far: KITCHEN"));
    }

    #[test]
    fn split_into_scenes_splits_on_headings() {
        let script = "\
INT. KITCHEN - DAY

JANE
Hi.

EXT. GARDEN - NIGHT

JOHN
Bye.";
        let scenes = split_into_scenes(script);
        assert_eq!(scenes.len(), 2);
        assert!(scenes[0].starts_with("INT. KITCHEN"));
        assert!(scenes[1].starts_with("EXT. GARDEN"));
    }

    #[test]
    fn split_into_scenes_keeps_leading_text_without_heading() {
        let script = "Some cold open text.\n\nINT. KITCHEN - DAY\n\nJANE\nHi.";
        let scenes = split_into_scenes(script);
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0], "Some cold open text.");
        assert!(scenes[1].starts_with("INT. KITCHEN"));
    }
}
