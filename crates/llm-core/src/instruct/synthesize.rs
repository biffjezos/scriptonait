//! Building instruction-format training examples out of raw corpus text —
//! an unrelated concern to parsing/generation, but one that reuses the
//! same `Form`/`Request` types since a synthesized example is exactly the
//! request that would have asked for the chunk it's paired with.

use super::{Form, Request};
use crate::screenplay;

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
        "could", "its", "our", "himself", "herself", "just", "like", "only", "over", "some",
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
}
