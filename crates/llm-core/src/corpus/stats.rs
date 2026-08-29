//! Corpus-wide and per-source analytics: how much text there is, what
//! kind it is, and how much of it training has actually drawn from.

use super::Corpus;

/// One source's share of the corpus, and how much of it training has
/// actually drawn from. See [`Corpus::per_source_stats`].
#[derive(Debug)]
pub struct SourceStats {
    pub id: String,
    pub train_tokens: usize,
    pub held_out_tokens: usize,
    pub sampled: u64,
}

impl Corpus {
    /// Every distinct word the sources use, for measuring whether
    /// generated text is made of words this corpus contains.
    ///
    /// The user's own text is the right reference here, not a
    /// dictionary: a model trained on screenplays should be judged on
    /// whether it writes the words those screenplays use, and no word
    /// list has to ship with the page for that.
    pub fn word_vocabulary(&mut self) -> std::rc::Rc<std::collections::HashSet<String>> {
        if let Some(cached) = &self.word_vocab {
            return std::rc::Rc::clone(cached);
        }
        let built = std::rc::Rc::new(crate::eval::vocabulary(self.cleaned_text.values()));
        self.word_vocab = Some(std::rc::Rc::clone(&built));
        built
    }

    /// Characters of cleaned text across every source.
    ///
    /// The stable measure of "how much text is there". The token count
    /// is not: relearning the vocabulary re-encodes the same text into a
    /// different number of tokens, and a number that moves for reasons
    /// the user did not cause is a number they stop believing.
    pub fn total_chars(&self) -> usize {
        self.cleaned_text.values().map(|t| t.chars().count()).sum()
    }

    /// Bytes of cleaned text per token, over the whole corpus.
    ///
    /// This is what turns a loss per token into bits per byte, which is
    /// the only form of the number that can be compared between two
    /// vocabularies — or against gzip.
    pub fn bytes_per_token(&self) -> f32 {
        let bytes: usize = self.cleaned_text.values().map(|t| t.len()).sum();
        let tokens = self.total_tokens();
        if tokens == 0 { 0.0 } else { bytes as f32 / tokens as f32 }
    }

    /// How many tokens of each kind of writing the corpus holds.
    ///
    /// Counted in tokens rather than in sources, because one novel and
    /// twenty scenes are not a balanced corpus however the file count
    /// reads. Kinds with no text at all are left out.
    pub fn mix(&self) -> Vec<(crate::mix::SourceKind, usize)> {
        let mut totals: Vec<(crate::mix::SourceKind, usize)> = Vec::new();
        for id in &self.order {
            let (Some(text), Some(tokens)) = (self.cleaned_text.get(id), self.sources.get(id))
            else {
                continue;
            };
            let kind = crate::mix::classify(text);
            match totals.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, count)) => *count += tokens.len(),
                None => totals.push((kind, tokens.len())),
            }
        }
        totals.sort_by(|a, b| b.1.cmp(&a.1));
        totals
    }

    /// Per-source token counts and sample draws, for showing which
    /// sources training has actually drawn from.
    ///
    /// `order`, `spans` and `val_spans` are kept index-aligned by
    /// `rebuild_flat_if_needed` (a too-small-to-split source still gets a
    /// zero-length `val_spans` entry), so zipping the three by index is
    /// safe here.
    pub fn per_source_stats(&mut self) -> Vec<SourceStats> {
        self.rebuild_flat_if_needed();
        self.order
            .iter()
            .zip(self.spans.iter())
            .zip(self.val_spans.iter())
            .map(|((id, &(_, train_tokens)), &(_, held_out_tokens))| SourceStats {
                id: id.clone(),
                train_tokens,
                held_out_tokens,
                sampled: self.sample_counts.get(id).copied().unwrap_or(0),
            })
            .collect()
    }

    /// Restore a sample count after a fresh page load re-upserts sources
    /// into a new `Corpus` — the count belongs to the source, not to this
    /// in-memory instance, so it has to be handed back in from wherever it
    /// was persisted.
    pub fn set_sample_count(&mut self, id: &str, count: u64) {
        if self.sources.contains_key(id) {
            self.sample_counts.insert(id.to_string(), count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Corpus;

    #[test]
    fn set_sample_count_restores_a_persisted_value() {
        let mut c = Corpus::new();
        c.upsert("a", "hello world", false);
        c.set_sample_count("a", 4_200);
        assert_eq!(c.per_source_stats()[0].sampled, 4_200);
        // A count for a source that no longer exists is dropped rather
        // than resurrecting a phantom entry.
        c.set_sample_count("nonexistent", 99);
        assert!(c.per_source_stats().iter().all(|s| s.id != "nonexistent"));
    }

    #[test]
    fn per_source_stats_stay_aligned_when_a_source_is_too_small_to_hold_out() {
        // "a" is long enough to split into train/held-out; "b" is a few
        // words, well under the 32-token floor, so it stays entirely in
        // training. Before the val_spans fix this shifted every stat
        // after "b" onto the wrong source.
        let mut c = Corpus::new();
        c.upsert("a", &"the cave and the fire and the shadows on the wall. ".repeat(200), false);
        c.upsert("b", "too short to hold out", false);
        c.upsert("c", &"the cave and the fire and the shadows on the wall. ".repeat(200), false);
        let stats = c.per_source_stats();
        let by_id = |id: &str| stats.iter().find(|s| s.id == id).unwrap();
        assert_eq!(by_id("b").held_out_tokens, 0);
        assert!(by_id("a").held_out_tokens > 0);
        assert!(by_id("c").held_out_tokens > 0);
    }
}
