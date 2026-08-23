//! What kind of writing a source is, by the shape of its lines.
//!
//! This exists so the training plan can say something more useful than
//! "add more text". A corpus of thirty film scripts and nothing else
//! teaches a model that every paragraph is one line of dialogue long;
//! what it needs is not more of the same, and the page can only say so
//! if it knows what is already there.
//!
//! Everything here is a heuristic over line shape and word endings — no
//! model, no word list of any size, nothing that needs the network. It
//! is right often enough to base a suggestion on and wrong often enough
//! that the suggestion has to be phrased as one.

use crate::screenplay::{is_character_cue, is_scene_heading};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Scene headings and character cues: a screenplay, a stage play, a
    /// teleplay.
    Screenplay,
    /// Prose with dialogue in it.
    Novel,
    /// Prose without much dialogue and with the vocabulary of argument:
    /// essays, philosophy, criticism, non-fiction.
    Essay,
    /// Short lines, stanza breaks: poetry, song lyrics.
    Verse,
    /// Recognizably none of the above, or too short to tell.
    Other,
}

impl SourceKind {
    /// What to call it in a sentence written for the person who loaded
    /// the file.
    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Screenplay => "film scripts",
            SourceKind::Novel => "novels and prose fiction",
            SourceKind::Essay => "essays and philosophy",
            SourceKind::Verse => "verse and lyrics",
            SourceKind::Other => "unclassified text",
        }
    }

    /// A stable machine-readable name, for the JSON the page reads.
    pub fn key(self) -> &'static str {
        match self {
            SourceKind::Screenplay => "screenplay",
            SourceKind::Novel => "novel",
            SourceKind::Essay => "essay",
            SourceKind::Verse => "verse",
            SourceKind::Other => "other",
        }
    }
}

/// Word endings that mark the vocabulary of argument rather than of
/// narration. A page of philosophy is thick with them; a page of a novel
/// is not. Crude, and deliberately so — the alternative is a word list
/// the size of a dictionary.
const ABSTRACT_SUFFIXES: &[&str] =
    &["tion", "sion", "ment", "ness", "ity", "ism", "ence", "ance", "ophy", "ology"];

/// Any of the quotation marks a book might use, straight or curly, plus
/// the dash that opens dialogue in several European typographies.
fn opens_dialogue(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('"')
        || trimmed.starts_with('\u{201c}')
        || trimmed.starts_with('\u{2018}')
        || trimmed.starts_with('\u{2014}')
        || line.contains('"')
        || line.contains('\u{201c}')
}

pub fn classify(text: &str) -> SourceKind {
    let lines: Vec<&str> = text.lines().map(str::trim_end).collect();
    let content: Vec<&str> = lines.iter().copied().filter(|l| !l.trim().is_empty()).collect();
    // Under a few dozen lines nothing below means anything: a pasted
    // scene and a pasted stanza look identical.
    if content.len() < 12 {
        return SourceKind::Other;
    }

    let headings = content.iter().filter(|l| is_scene_heading(l)).count();
    let cues = content.iter().filter(|l| is_character_cue(l)).count();
    // Five percent of lines, or three scene headings outright. Either is
    // far past what prose produces by accident.
    if headings >= 3 || (headings + cues) * 20 >= content.len() {
        return SourceKind::Screenplay;
    }

    let total_len: usize = content.iter().map(|l| l.trim().chars().count()).sum();
    let mean_len = total_len as f32 / content.len() as f32;
    let short = content.iter().filter(|l| l.trim().chars().count() < 60).count();
    // Verse is short lines nearly all the way down. Prose that has been
    // hard-wrapped at 80 columns is not: its lines cluster just under the
    // wrap width rather than well below it.
    if mean_len < 45.0 && short * 4 >= content.len() * 3 {
        return SourceKind::Verse;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 200 {
        return SourceKind::Other;
    }
    let abstracts = words
        .iter()
        .filter(|w| {
            let w = w.trim_matches(|c: char| !c.is_alphabetic()).to_ascii_lowercase();
            w.len() > 5 && ABSTRACT_SUFFIXES.iter().any(|s| w.ends_with(s))
        })
        .count();
    let dialogue = content.iter().filter(|l| opens_dialogue(l)).count();

    let abstract_rate = abstracts as f32 / words.len() as f32;
    let dialogue_rate = dialogue as f32 / content.len() as f32;
    // Dialogue wins the tie: a novel full of long words is still a
    // novel, but an essay does not carry quoted speech line after line.
    if dialogue_rate > 0.10 {
        SourceKind::Novel
    } else if abstract_rate > 0.02 {
        SourceKind::Essay
    } else {
        SourceKind::Novel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat(block: &str, times: usize) -> String {
        block.repeat(times)
    }

    #[test]
    fn a_screenplay_is_recognized_by_its_headings_and_cues() {
        let text = repeat(
            "INT. KITCHEN - NIGHT\n\nShe stands at the window.\n\nMARIA\nYou came back.\n\n\
             DAVID\nI did.\n\n",
            4,
        );
        assert_eq!(classify(&text), SourceKind::Screenplay);
    }

    #[test]
    fn lyrics_are_recognized_by_short_lines() {
        let text = repeat(
            "I walked the road alone\nThe rain came down\nAnd I was gone\nBefore the morning\n\n",
            8,
        );
        assert_eq!(classify(&text), SourceKind::Verse);
    }

    #[test]
    fn prose_with_quoted_speech_reads_as_a_novel() {
        let text = repeat(
            "\"You came back,\" she said, and the words hung in the kitchen air for longer than \
             either of them wanted them to.\nHe put down the bag he had carried since the station \
             and looked at the window instead of at her face.\n\"I did,\" he said.\n\n",
            8,
        );
        assert_eq!(classify(&text), SourceKind::Novel);
    }

    #[test]
    fn expository_prose_reads_as_an_essay() {
        let text = repeat(
            "The distinction between perception and imagination is not a distinction of intensity \
             but of intention, and any explanation that reduces one to a weaker version of the \
             other has abandoned the phenomenon it set out to describe.\nThe identity of an \
             observation depends on the situation of the observer, and that dependence is not a \
             limitation of knowledge but a condition of it.\n\n",
            8,
        );
        assert_eq!(classify(&text), SourceKind::Essay);
    }

    #[test]
    fn a_pasted_fragment_is_not_classified_at_all() {
        assert_eq!(classify("two lines\nof nothing much"), SourceKind::Other);
    }
}
