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

const SLUGLINE_PREFIXES: &[&str] =
    &["INT./EXT.", "EXT./INT.", "INT/EXT.", "EXT/INT.", "INT.", "EXT.", "EST.", "I/E."];

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

/// Whether `line` looks like a scene slugline (`INT.`/`EXT.`/... prefix).
pub fn is_scene_heading(line: &str) -> bool {
    let trimmed = line.trim();
    let upper = trimmed.to_ascii_uppercase();
    SLUGLINE_PREFIXES.iter().any(|p| upper.starts_with(p))
}

pub fn parse_scene_heading(line: &str) -> Option<SceneHeading> {
    let trimmed = line.trim();
    let upper = trimmed.to_ascii_uppercase();
    let prefix = SLUGLINE_PREFIXES.iter().find(|p| upper.starts_with(**p))?;
    let rest = &trimmed[prefix.len()..];
    let (location, time) = split_location_time(rest.trim_start());
    Some(SceneHeading { raw: trimmed.to_string(), location: location.trim().to_string(), time })
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
    let looks_like_shouting_caps = name_part
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || matches!(c, ' ' | '.' | '\'' | '-'));
    if !looks_like_shouting_caps || !name_part.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    !TRANSITION_WORDS.contains(&name_part)
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
/// cues in first-seen order. A candidate cue only counts as a character
/// if the next non-empty line looks like dialogue (contains a lowercase
/// letter) rather than another all-caps line — real character cues are
/// always followed by their dialogue, whereas non-screenplay all-caps
/// text (shooting-script section/shot codes like `A1`, `SECTION TIMING`,
/// timing tables) tends to run in consecutive all-caps lines with nothing
/// in between, which would otherwise all get swept up as "characters".
pub fn extract_story_state(text: &str) -> StoryState {
    let mut state = StoryState::default();
    let lines: Vec<&str> = text.lines().collect();
    for (i, &line) in lines.iter().enumerate() {
        if let Some(heading) = parse_scene_heading(line) {
            state.scene_count += 1;
            if !heading.location.is_empty() {
                push_unique(&mut state.locations, heading.location);
            }
            continue;
        }
        let Some(name) = character_name(line) else { continue };
        let next_line = lines[i + 1..].iter().find(|l| !l.trim().is_empty());
        if !next_line.is_some_and(|l| looks_like_dialogue(l)) {
            continue;
        }
        push_unique(&mut state.characters, name);
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
suddenly fixed, leaving only seven flowing columns.";

    #[test]
    fn detects_scene_headings() {
        assert!(is_scene_heading("INT. KITCHEN - DAY"));
        assert!(is_scene_heading("  ext. mars colony - night  "));
        assert!(is_scene_heading("INT./EXT. CAR - MOVING - NIGHT"));
        assert!(!is_scene_heading("He walked into the kitchen."));
        assert!(!is_scene_heading("CYPHER"));
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
        assert_eq!(character_name("INT. KITCHEN - DAY"), None);
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
It's cold.";
        let state = extract_story_state(script);
        assert_eq!(state.scene_count, 2);
        assert_eq!(state.characters, vec!["JANE".to_string(), "JOHN".to_string()]);
        assert_eq!(state.locations, vec!["KITCHEN".to_string(), "GARDEN".to_string()]);
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
Hi Bob.";
        let state = extract_story_state(script);
        // BOB is immediately followed by another all-caps line (ALICE), so
        // it reads as two consecutive section-style markers rather than a
        // character cue followed by dialogue, and is rejected.
        assert_eq!(state.characters, vec!["JANE".to_string(), "ALICE".to_string()]);
    }

    #[test]
    fn merge_deduplicates() {
        let mut a = extract_story_state("INT. KITCHEN - DAY\n\nJANE\nHi.");
        let b = extract_story_state("EXT. GARDEN - NIGHT\n\nJANE\nBye.\n\nJOHN\nWhat?");
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
        let state = extract_story_state("INT. KITCHEN - DAY\n\nJANE\nHi.");
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
