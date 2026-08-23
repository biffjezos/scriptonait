//! Measuring a model by something other than its loss.
//!
//! Loss is the only number a training loop naturally produces, and it is
//! close to unreadable: 4.8 nats per token means nothing on its own, it
//! is not comparable between two vocabularies, and it says nothing at
//! all about whether the text that comes out is English.
//!
//! Three numbers here answer questions loss cannot:
//!
//!   * **Bits per byte** converts the loss into how many bits it takes
//!     this model to encode a byte of the user's actual text. That is
//!     comparable across vocabularies — a model with a bigger vocabulary
//!     has a lower loss per token for free, and the same bits per byte —
//!     and it is comparable against known references (gzip is around 2.5
//!     bits/byte on English prose; a good small character model is
//!     around 1.2).
//!   * **Known-word rate** is the fraction of the words the model
//!     produces that appear anywhere in the user's own corpus. Not a
//!     dictionary: their corpus is the right target, and it needs no
//!     word list shipped with the page. A model producing readable
//!     English from this text scores above 0.9; one producing plausible
//!     letter soup scores far lower, and that is the difference the loss
//!     curve hides.
//!   * **Repeated four-gram rate** catches the other failure: a model
//!     that has stopped saying anything new and is cycling a phrase.
//!     Human writing repeats a four-word run occasionally; a degenerate
//!     sample repeats most of them.

use std::collections::HashSet;

/// Split text into comparable words: lowercased, stripped of the
/// punctuation around them, empties dropped.
///
/// Both the corpus vocabulary and the generated sample go through this,
/// which is the only thing that makes the comparison mean anything.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Every distinct word in some text, for use as the known-word set.
pub fn vocabulary(texts: impl IntoIterator<Item = impl AsRef<str>>) -> HashSet<String> {
    let mut set = HashSet::new();
    for text in texts {
        set.extend(words(text.as_ref()));
    }
    set
}

/// How many bits this model spends per byte of the original text.
///
/// `loss_nats` is the mean cross-entropy per token, `tokens` and `bytes`
/// describe the same stretch of text — so the ratio between them is how
/// much text a token carries, and multiplying through converts a
/// per-token number into a per-byte one. Returns 0 when there is nothing
/// to divide by.
pub fn bits_per_byte(loss_nats: f32, tokens: usize, bytes: usize) -> f32 {
    if bytes == 0 || tokens == 0 {
        return 0.0;
    }
    bits_per_byte_from_ratio(loss_nats, bytes as f32 / tokens as f32)
}

/// The same conversion when the ratio is already known — a corpus keeps
/// its own bytes-per-token, and recomputing it from two totals just to
/// divide them again is noise.
pub fn bits_per_byte_from_ratio(loss_nats: f32, bytes_per_token: f32) -> f32 {
    if bytes_per_token <= 0.0 || !loss_nats.is_finite() {
        return 0.0;
    }
    loss_nats / bytes_per_token / std::f32::consts::LN_2
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextStats {
    /// How many words the text had at all. Every rate below is over
    /// this, and under a couple of dozen none of them mean much.
    pub words: usize,
    /// Fraction of words that appear in the reference vocabulary.
    pub known_word_rate: f32,
    /// Fraction of four-word runs that had already appeared earlier in
    /// the same text.
    pub repeated_4gram_rate: f32,
    /// Distinct words over total words. A sample stuck in a loop has a
    /// low one whatever its loss says.
    pub distinct_word_rate: f32,
    /// The first few words that were *not* in the corpus, so "0.7 known"
    /// can be looked at rather than taken on faith.
    pub unknown_examples: Vec<String>,
}

impl Default for TextStats {
    fn default() -> Self {
        Self {
            words: 0,
            known_word_rate: 0.0,
            repeated_4gram_rate: 0.0,
            distinct_word_rate: 0.0,
            unknown_examples: Vec::new(),
        }
    }
}

/// Measure one piece of generated text against the words the corpus uses.
pub fn text_stats(text: &str, known: &HashSet<String>) -> TextStats {
    let ws = words(text);
    if ws.is_empty() {
        return TextStats::default();
    }

    let mut unknown_examples = Vec::new();
    let mut known_count = 0usize;
    for w in &ws {
        if known.contains(w) {
            known_count += 1;
        } else if unknown_examples.len() < 8 && !unknown_examples.contains(w) {
            unknown_examples.push(w.clone());
        }
    }

    let distinct: HashSet<&String> = ws.iter().collect();

    // Four-word runs, counting a run as repeated the second time it is
    // seen and every time after.
    let mut seen: HashSet<[&str; 4]> = HashSet::new();
    let mut repeats = 0usize;
    let mut grams = 0usize;
    for window in ws.windows(4) {
        let gram = [
            window[0].as_str(),
            window[1].as_str(),
            window[2].as_str(),
            window[3].as_str(),
        ];
        grams += 1;
        if !seen.insert(gram) {
            repeats += 1;
        }
    }

    TextStats {
        words: ws.len(),
        known_word_rate: known_count as f32 / ws.len() as f32,
        repeated_4gram_rate: if grams == 0 { 0.0 } else { repeats as f32 / grams as f32 },
        distinct_word_rate: distinct.len() as f32 / ws.len() as f32,
        unknown_examples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_vocab() -> HashSet<String> {
        vocabulary(["the quick brown fox jumps over the lazy dog and then walks away"])
    }

    #[test]
    fn words_are_lowercased_and_stripped() {
        assert_eq!(words("The, QUICK -- fox's!"), vec!["the", "quick", "fox's"]);
    }

    #[test]
    fn text_the_corpus_could_have_written_scores_high() {
        let stats = text_stats("The quick fox jumps over the lazy dog", &corpus_vocab());
        assert_eq!(stats.words, 8);
        assert_eq!(stats.known_word_rate, 1.0);
        assert!(stats.unknown_examples.is_empty());
    }

    #[test]
    fn letter_soup_scores_low_and_names_the_soup() {
        let stats = text_stats("the qwx brln fzz jmps ovr the lzy dg", &corpus_vocab());
        assert!(stats.known_word_rate < 0.3, "{stats:?}");
        assert!(stats.unknown_examples.contains(&"qwx".to_string()));
    }

    #[test]
    fn a_loop_shows_up_as_repeated_four_grams() {
        let text = "the quick brown fox the quick brown fox the quick brown fox";
        let stats = text_stats(text, &corpus_vocab());
        assert!(stats.repeated_4gram_rate > 0.5, "{stats:?}");
        assert!(stats.distinct_word_rate < 0.4, "{stats:?}");
    }

    #[test]
    fn text_that_says_something_new_each_time_does_not() {
        let text = "the quick brown fox jumps over the lazy dog and then walks away";
        let stats = text_stats(text, &corpus_vocab());
        assert_eq!(stats.repeated_4gram_rate, 0.0, "{stats:?}");
    }

    #[test]
    fn bits_per_byte_converts_a_per_token_loss() {
        // Four bytes to a token, one nat per token: a nat is 1/ln2 bits,
        // spread over four bytes.
        let bpb = bits_per_byte(1.0, 100, 400);
        assert!((bpb - (0.25 / std::f32::consts::LN_2)).abs() < 1e-6, "{bpb}");
    }

    #[test]
    fn bits_per_byte_is_zero_when_there_is_nothing_to_measure() {
        assert_eq!(bits_per_byte(4.0, 0, 0), 0.0);
        assert_eq!(bits_per_byte(f32::NAN, 10, 10), 0.0);
    }

    #[test]
    fn empty_text_measures_as_nothing_rather_than_dividing_by_zero() {
        assert_eq!(text_stats("   ", &corpus_vocab()), TextStats::default());
    }
}
