//! Retrieval over the user's own corpus: find scenes similar to a prompt
//! and hand them to generation as few-shot context. This is classic TF-IDF
//! + cosine similarity, not a learned embedding model — no dependency,
//! and at the corpus sizes this project targets (a browser tab, a handful
//! to dozens of sources) it's both cheap and effective. Chunking reuses
//! `screenplay::split_into_scenes` so retrieval works at scene
//! granularity rather than arbitrary token windows.

use std::collections::HashMap;

use crate::screenplay;

fn tokenize_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn term_freq(words: &[String]) -> HashMap<String, usize> {
    let mut tf = HashMap::new();
    for w in words {
        *tf.entry(w.clone()).or_insert(0) += 1;
    }
    tf
}

struct Chunk {
    source_id: String,
    text: String,
    term_freq: HashMap<String, usize>,
}

pub struct RetrievedChunk {
    pub source_id: String,
    pub text: String,
    pub score: f32,
}

/// TF-IDF index over scene-level chunks from the corpus's sources.
pub struct RetrievalIndex {
    chunks: Vec<Chunk>,
    doc_freq: HashMap<String, usize>,
}

impl Default for RetrievalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl RetrievalIndex {
    pub fn new() -> Self {
        Self { chunks: Vec::new(), doc_freq: HashMap::new() }
    }

    /// (Re-)indexes one source's scene chunks. Safe to call again for the
    /// same `source_id` after an edit — it clears that source's previous
    /// chunks first, so it replaces rather than duplicates.
    pub fn upsert_document(&mut self, source_id: &str, cleaned_text: &str) {
        self.remove_document(source_id);
        for scene_text in screenplay::split_into_scenes(cleaned_text) {
            let words = tokenize_words(&scene_text);
            if words.is_empty() {
                continue;
            }
            let tf = term_freq(&words);
            for term in tf.keys() {
                *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            self.chunks.push(Chunk { source_id: source_id.to_string(), text: scene_text, term_freq: tf });
        }
    }

    pub fn remove_document(&mut self, source_id: &str) {
        let doc_freq = &mut self.doc_freq;
        self.chunks.retain(|chunk| {
            if chunk.source_id != source_id {
                return true;
            }
            for term in chunk.term_freq.keys() {
                if let Some(count) = doc_freq.get_mut(term) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        doc_freq.remove(term);
                    }
                }
            }
            false
        });
    }

    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Smoothed inverse document frequency: always positive, no
    /// div-by-zero/`ln(0)` even for a term that (post-removal bookkeeping
    /// races aside) isn't in `doc_freq` at all.
    fn idf(&self, term: &str) -> f32 {
        let n = self.chunks.len().max(1) as f32;
        let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
        ((n + 1.0) / (df + 1.0)).ln() + 1.0
    }

    fn tfidf_vector(&self, tf: &HashMap<String, usize>) -> HashMap<String, f32> {
        tf.iter()
            .map(|(term, &count)| {
                let tf_weight = 1.0 + (count as f32).ln();
                (term.clone(), tf_weight * self.idf(term))
            })
            .collect()
    }

    fn cosine(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
        let mut dot = 0.0f32;
        for (term, &wa) in a {
            if let Some(&wb) = b.get(term) {
                dot += wa * wb;
            }
        }
        let norm_a = a.values().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b = b.values().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Up to `k` chunks most similar to `query`, highest similarity
    /// first. Chunks with zero overlap with `query` are excluded
    /// entirely (a positive score always means at least one shared term).
    pub fn top_k(&self, query: &str, k: usize) -> Vec<RetrievedChunk> {
        if k == 0 || self.chunks.is_empty() {
            return Vec::new();
        }
        let query_tf = term_freq(&tokenize_words(query));
        if query_tf.is_empty() {
            return Vec::new();
        }
        let query_vec = self.tfidf_vector(&query_tf);

        let mut scored: Vec<(f32, usize)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| (Self::cosine(&query_vec, &self.tfidf_vector(&chunk.term_freq)), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .filter(|(score, _)| *score > 0.0)
            .take(k)
            .map(|(score, i)| {
                let chunk = &self.chunks[i];
                RetrievedChunk { source_id: chunk.source_id.clone(), text: chunk.text.clone(), score }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_returns_no_results() {
        let idx = RetrievalIndex::new();
        assert!(idx.top_k("anything", 3).is_empty());
    }

    #[test]
    fn empty_query_returns_no_results() {
        let mut idx = RetrievalIndex::new();
        idx.upsert_document("a", "INT. KITCHEN - DAY\n\nJANE\nHello there.");
        assert!(idx.top_k("", 3).is_empty());
        assert!(idx.top_k("   ", 3).is_empty());
    }

    #[test]
    fn finds_the_more_similar_document() {
        let mut idx = RetrievalIndex::new();
        idx.upsert_document(
            "spy",
            "INT. SURVEILLANCE VAN - NIGHT\n\nAgents monitor a wiretap, tracing a phone number \
             through encrypted satellite relays.",
        );
        idx.upsert_document(
            "romance",
            "INT. RESTAURANT - EVENING\n\nTwo old friends share a quiet dinner and reminisce \
             about a summer holiday by the sea.",
        );

        let results = idx.top_k("wiretap surveillance phone tracing", 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].source_id, "spy");
    }

    #[test]
    fn upsert_replaces_not_duplicates() {
        let mut idx = RetrievalIndex::new();
        idx.upsert_document("a", "INT. KITCHEN - DAY\n\nSome text about apples.");
        let first_count = idx.num_chunks();
        idx.upsert_document("a", "INT. KITCHEN - DAY\n\nCompletely different text about oranges.");
        assert_eq!(idx.num_chunks(), first_count);
        let results = idx.top_k("apples", 5);
        assert!(results.is_empty(), "stale chunk should have been replaced");
    }

    #[test]
    fn remove_document_clears_its_chunks_and_doc_freq() {
        let mut idx = RetrievalIndex::new();
        idx.upsert_document("a", "INT. KITCHEN - DAY\n\nA scene about spaceships and lasers.");
        assert!(idx.num_chunks() > 0);
        idx.remove_document("a");
        assert_eq!(idx.num_chunks(), 0);
        assert!(idx.doc_freq.is_empty());
        assert!(idx.top_k("spaceships lasers", 3).is_empty());
    }

    #[test]
    fn top_k_respects_k() {
        let mut idx = RetrievalIndex::new();
        for i in 0..5 {
            idx.upsert_document(&format!("s{i}"), &format!("INT. ROOM {i} - DAY\n\nA character talks about dogs."));
        }
        let results = idx.top_k("dogs", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn common_words_alone_dont_dominate_ranking() {
        // "the" appears in both documents; only "wormhole" is distinctive.
        let mut idx = RetrievalIndex::new();
        idx.upsert_document("a", "The captain stared at the the the readout in silence.");
        idx.upsert_document("b", "The ship enters the wormhole and the crew brace for impact.");
        let results = idx.top_k("wormhole", 2);
        assert_eq!(results[0].source_id, "b");
    }
}
