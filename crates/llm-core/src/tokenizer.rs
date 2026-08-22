//! Byte-level BPE tokenizer.
//!
//! The base alphabet is the 256 byte values plus a handful of special
//! tokens, so — like the pure byte-level tokenizer this replaces — it can
//! encode *any* input (any UTF-8 text, any language, or arbitrary bytes
//! pasted by the user) and never emits an unknown token. On top of that
//! base it learns merges from the training corpus, so common sequences
//! ("the ", "INT. ", a newline followed by an indent) become single
//! tokens.
//!
//! Why this matters more than it looks: at byte level, a 700-word story is
//! about 4,000 tokens. With an 8k-merge vocabulary it's about 900. That is
//! four times less compute to generate the same story, four times more
//! story inside the same attention window, and four times more text seen
//! per training step. It is the cheapest large win available to a model
//! this size.
//!
//! A `Tokenizer` with no merges is *exactly* the old byte-level
//! tokenizer, which is why it's the default: nothing has to special-case
//! "no vocabulary trained yet".
//!
//! ## Layout
//!
//! ```text
//!   0..=255   raw byte values
//!   256..=263 special tokens (see below; 261..=263 reserved)
//!   264..     learned merges, in the order they were learned
//! ```

use std::collections::{HashMap, HashSet};

/// Padding (unused by the current trainer, reserved so a batch-padding
/// implementation doesn't have to renumber the vocabulary).
pub const PAD: u32 = 256;
/// Start of a document.
pub const BOS: u32 = 257;
/// End of a document.
pub const EOS: u32 = 258;
/// Start of an instruction ("write a 700 word novel about ...").
pub const TASK: u32 = 259;
/// Start of the text that answers the preceding instruction. Generation
/// from a parsed prompt emits `TASK <instruction> STORY` and samples from
/// there; see `instruct.rs`.
pub const STORY: u32 = 260;

/// First id available to a learned merge. The gap after `STORY` is three
/// reserved ids: adding a special token later must not shift every merge
/// id, because that would invalidate every checkpoint ever saved.
pub const FIRST_MERGE_ID: u32 = 264;

/// Vocabulary size with no merges learned — i.e. plain byte level.
pub const BASE_VOCAB_SIZE: usize = FIRST_MERGE_ID as usize;

/// Merges are never learned across one of these boundaries.
///
/// Without a pre-tokenizer, BPE happily learns a single token for
/// `". He said"`, which wastes vocabulary on phrases that only ever
/// appear in one context. Splitting first — into "a word, optionally with
/// its leading space", "a run of digits", "a run of punctuation", "a run
/// of whitespace" — is what keeps the learned vocabulary made of
/// reusable pieces.
///
/// Whitespace runs are deliberately their own chunk rather than being
/// trimmed: plain-text screenplays use indentation as the only signal
/// separating scene headings, character cues, dialogue, and action, so
/// `"\n\n          "` becoming one token is a *feature* — it's the model
/// learning what a character cue looks like.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut chunks = Vec::new();
    let mut i = 0usize;
    // The ASCII classifiers below can only ever match single-byte
    // characters, so those arms are safe to advance byte at a time. Any
    // byte >= 0x80 falls through to the final arm, which advances whole
    // characters — so a chunk boundary can never land inside one.
    while i < bytes.len() {
        let start = i;
        let b = bytes[i];
        if b == b' ' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_alphabetic() {
            // One leading space joins the word after it: " the" is one
            // token, which is how the same word mid-sentence and
            // start-of-line stay distinguishable without doubling the
            // vocabulary.
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else if b.is_ascii_alphabetic() {
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else if b.is_ascii_digit() {
            // Capped at three so the vocabulary doesn't fill with
            // specific numbers.
            let stop = (i + 3).min(bytes.len());
            while i < stop && bytes[i].is_ascii_digit() {
                i += 1;
            }
        } else if b.is_ascii_whitespace() {
            let stop = (i + 32).min(bytes.len());
            while i < stop && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
        } else {
            // Punctuation, and all non-ASCII text. Capped by character
            // count, and stepped by `len_utf8`, so a multi-byte
            // character is never cut in half.
            let mut chars = 0;
            while i < bytes.len()
                && chars < 8
                && !bytes[i].is_ascii_alphanumeric()
                && !bytes[i].is_ascii_whitespace()
            {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
                chars += 1;
            }
        }
        chunks.push(&text[start..i]);
    }
    chunks
}

#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    /// Learned merges in learned order: `merges[n]` produces id
    /// `FIRST_MERGE_ID + n`.
    merges: Vec<(u32, u32)>,
    /// `(left, right) -> merge index`, i.e. the priority of a merge.
    ranks: HashMap<(u32, u32), u32>,
    /// The literal bytes each token id stands for. Special tokens have
    /// none, which is what makes `decode` drop them.
    pieces: Vec<Vec<u8>>,
}

impl Tokenizer {
    /// A tokenizer with no merges: plain byte level.
    pub fn byte_level() -> Self {
        let mut t = Self { merges: Vec::new(), ranks: HashMap::new(), pieces: Vec::new() };
        t.rebuild_pieces();
        t
    }

    pub fn vocab_size(&self) -> usize {
        BASE_VOCAB_SIZE + self.merges.len()
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    fn rebuild_pieces(&mut self) {
        let mut pieces: Vec<Vec<u8>> = Vec::with_capacity(self.vocab_size());
        for b in 0u32..256 {
            pieces.push(vec![b as u8]);
        }
        for _ in 256..FIRST_MERGE_ID {
            pieces.push(Vec::new()); // specials render as nothing
        }
        for &(a, b) in &self.merges {
            let mut piece = pieces[a as usize].clone();
            piece.extend_from_slice(&pieces[b as usize]);
            pieces.push(piece);
        }
        self.pieces = pieces;
        self.ranks =
            self.merges.iter().enumerate().map(|(i, &pair)| (pair, i as u32)).collect();
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 8);
        // Chunks repeat constantly in natural text (" the" thousands of
        // times in a novel), and merging one is the expensive part, so
        // one cache per call turns most of the work into a hash lookup.
        // It's per call rather than a field so a `Tokenizer` stays
        // immutable and shareable across threads.
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        for chunk in pretokenize(text) {
            if let Some(cached) = cache.get(chunk) {
                out.extend_from_slice(cached);
                continue;
            }
            let encoded = self.encode_chunk(chunk);
            out.extend_from_slice(&encoded);
            cache.insert(chunk, encoded);
        }
        out
    }

    /// BPE over one pre-tokenized chunk: repeatedly merge the adjacent
    /// pair with the lowest rank (i.e. the merge that was learned
    /// earliest, so the most frequent) until no adjacent pair is a known
    /// merge.
    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> = chunk.bytes().map(u32::from).collect();
        if self.merges.is_empty() || symbols.len() < 2 {
            return symbols;
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.ranks.get(&(symbols[i], symbols[i + 1])) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((rank, _)) = best else { break };
            let pair = self.merges[rank as usize];
            let new_id = FIRST_MERGE_ID + rank;
            // Merge every non-overlapping occurrence of this pair in one
            // sweep, not just the one found above: same result, far fewer
            // rescans.
            let mut merged = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && (symbols[i], symbols[i + 1]) == pair {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(symbols[i]);
                    i += 1;
                }
            }
            symbols = merged;
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }

    /// Decode token ids back to text. Special tokens contribute nothing;
    /// any invalid UTF-8 (from a window that cut a multi-byte character
    /// in half) is replaced per the standard lossy rules.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            if let Some(piece) = self.pieces.get(id as usize) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The bytes a single token stands for (empty for specials, and for
    /// ids past the end of the vocabulary).
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Learn merges from `texts` until the vocabulary reaches
    /// `target_vocab_size` (or no pair occurs more than once).
    ///
    /// Standard BPE, with the usual bookkeeping that keeps it from being
    /// quadratic: pair counts are maintained incrementally, and an index
    /// from pair to the words containing it means each merge only touches
    /// the words it actually affects.
    pub fn train(texts: &[&str], target_vocab_size: usize) -> Self {
        let mut word_counts: HashMap<&str, u64> = HashMap::new();
        for text in texts {
            for chunk in pretokenize(text) {
                *word_counts.entry(chunk).or_insert(0) += 1;
            }
        }

        let mut words: Vec<Vec<u32>> = Vec::with_capacity(word_counts.len());
        let mut counts: Vec<u64> = Vec::with_capacity(word_counts.len());
        for (word, count) in word_counts {
            if word.len() < 2 {
                continue; // nothing to merge inside a single byte
            }
            words.push(word.bytes().map(u32::from).collect());
            counts.push(count);
        }

        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut containing: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for (w, symbols) in words.iter().enumerate() {
            for pair in symbols.windows(2) {
                let key = (pair[0], pair[1]);
                *pair_counts.entry(key).or_insert(0) += counts[w] as i64;
                containing.entry(key).or_default().insert(w);
            }
        }

        let mut merges: Vec<(u32, u32)> = Vec::new();
        while BASE_VOCAB_SIZE + merges.len() < target_vocab_size {
            // Ties broken by the pair itself so training is deterministic
            // despite HashMap's iteration order.
            let best = pair_counts
                .iter()
                .filter(|(_, &c)| c > 1)
                .max_by_key(|(&pair, &count)| (count, std::cmp::Reverse(pair)))
                .map(|(&pair, _)| pair);
            let Some(pair) = best else { break };
            let new_id = FIRST_MERGE_ID + merges.len() as u32;
            merges.push(pair);

            let affected: Vec<usize> =
                containing.get(&pair).map(|s| s.iter().copied().collect()).unwrap_or_default();
            for w in affected {
                let count = counts[w] as i64;
                let old = &words[w];
                let mut new_symbols = Vec::with_capacity(old.len());
                let mut i = 0;
                while i < old.len() {
                    if i + 1 < old.len() && (old[i], old[i + 1]) == pair {
                        new_symbols.push(new_id);
                        i += 2;
                    } else {
                        new_symbols.push(old[i]);
                        i += 1;
                    }
                }
                if new_symbols.len() == old.len() {
                    continue; // pair wasn't actually here any more
                }
                for p in old.windows(2) {
                    let key = (p[0], p[1]);
                    if let Some(c) = pair_counts.get_mut(&key) {
                        *c -= count;
                    }
                    if let Some(set) = containing.get_mut(&key) {
                        set.remove(&w);
                    }
                }
                for p in new_symbols.windows(2) {
                    let key = (p[0], p[1]);
                    *pair_counts.entry(key).or_insert(0) += count;
                    containing.entry(key).or_default().insert(w);
                }
                words[w] = new_symbols;
            }
            pair_counts.remove(&pair);
            containing.remove(&pair);
            pair_counts.retain(|_, c| *c > 0);
        }

        let mut t = Self { merges, ranks: HashMap::new(), pieces: Vec::new() };
        t.rebuild_pieces();
        t
    }

    /// Serialize to the on-disk `.tok` format: a magic number, a format
    /// version, then the merge list. The pieces table and the rank map
    /// are both derived from it, so they aren't stored.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.merges.len() * 8);
        out.extend_from_slice(TOKENIZER_MAGIC);
        out.extend_from_slice(&TOKENIZER_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(a, b) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != TOKENIZER_MAGIC {
            return Err("not a scriptonait tokenizer file".to_string());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != TOKENIZER_VERSION {
            return Err(format!("tokenizer format version {version}, expected {TOKENIZER_VERSION}"));
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let expected = 12 + n * 8;
        if bytes.len() != expected {
            return Err(format!("expected {expected} bytes for {n} merges, got {}", bytes.len()));
        }
        let mut merges = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 8;
            let a = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let b = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            // A merge may only refer to ids that already exist at the
            // point it's learned; anything else would make `pieces`
            // recursion ill-defined.
            let limit = FIRST_MERGE_ID + i as u32;
            if a >= limit || b >= limit {
                return Err(format!("merge {i} refers to token id >= {limit}"));
            }
            merges.push((a, b));
        }
        let mut t = Self { merges, ranks: HashMap::new(), pieces: Vec::new() };
        t.rebuild_pieces();
        Ok(t)
    }
}

const TOKENIZER_MAGIC: &[u8; 4] = b"SCTK";
const TOKENIZER_VERSION: u32 = 1;

/// Wrap a token sequence with BOS/EOS boundary markers, used when
/// concatenating multiple sources into one training corpus so the model can
/// learn where one document ends and another begins.
pub fn wrap_with_boundaries(tokens: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(tokens.len() + 2);
    out.push(BOS);
    out.extend_from_slice(tokens);
    out.push(EOS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_round_trip_ascii() {
        let t = Tokenizer::byte_level();
        let s = "Hello, world! This is a script.";
        assert_eq!(t.decode(&t.encode(s)), s);
        assert_eq!(t.encode(s).len(), s.len());
    }

    #[test]
    fn byte_level_round_trip_utf8() {
        let t = Tokenizer::byte_level();
        let s = "caf\u{e9} \u{1f980} \u{65e5}\u{672c}\u{8a9e}";
        assert_eq!(t.decode(&t.encode(s)), s);
    }

    #[test]
    fn special_tokens_outside_byte_range() {
        for special in [PAD, BOS, EOS, TASK, STORY] {
            assert!(special >= 256 && special < FIRST_MERGE_ID);
        }
        assert_eq!(Tokenizer::byte_level().vocab_size(), BASE_VOCAB_SIZE);
    }

    #[test]
    fn decode_ignores_special_tokens() {
        let t = Tokenizer::byte_level();
        let ids = vec![BOS, b'h' as u32, b'i' as u32, EOS, PAD, TASK, STORY];
        assert_eq!(t.decode(&ids), "hi");
    }

    #[test]
    fn empty_input() {
        let t = Tokenizer::byte_level();
        assert_eq!(t.encode(""), Vec::<u32>::new());
        assert_eq!(t.decode(&[]), "");
    }

    #[test]
    fn pretokenizer_keeps_indentation_as_its_own_chunk() {
        // The screenplay case: the indent before a character cue has to
        // survive as something mergeable, not be trimmed away.
        let chunks = pretokenize("ACTION\n\n     SOCRATES\n");
        assert!(chunks.contains(&"\n\n     "), "{chunks:?}");
        assert!(chunks.contains(&"SOCRATES"), "{chunks:?}");
    }

    #[test]
    fn pretokenizer_attaches_one_leading_space_to_a_word() {
        assert_eq!(pretokenize("the cave"), vec!["the", " cave"]);
    }

    #[test]
    fn pretokenizer_covers_the_input_exactly() {
        let s = "INT. CAVE - DAY\n\n  He said: \"42 shadows\", then left.\u{1f980}\n";
        assert_eq!(pretokenize(s).concat(), s);
    }

    fn training_text() -> String {
        "the shadows on the cave wall are the only world the prisoners know. \
         the fire behind them throws the shadows. "
            .repeat(40)
    }

    #[test]
    fn training_learns_merges_and_shortens_the_encoding() {
        let text = training_text();
        let t = Tokenizer::train(&[&text], 400);
        assert!(t.num_merges() > 50, "learned only {} merges", t.num_merges());
        assert_eq!(t.vocab_size(), BASE_VOCAB_SIZE + t.num_merges());

        let sample = "the shadows on the cave wall";
        let bpe = t.encode(sample).len();
        let bytes = sample.len();
        assert!(bpe * 2 < bytes, "expected a real compression: {bpe} tokens for {bytes} bytes");
    }

    #[test]
    fn trained_tokenizer_round_trips_including_unseen_text() {
        let text = training_text();
        let t = Tokenizer::train(&[&text], 400);
        for sample in [
            "the shadows on the cave wall",
            "a totally unseen sentence with \u{e9}\u{1f980} and 12345 digits!",
            "",
            "\n\n     SOCRATES\n",
        ] {
            assert_eq!(t.decode(&t.encode(sample)), sample, "round trip failed for {sample:?}");
        }
    }

    #[test]
    fn training_is_deterministic() {
        let text = training_text();
        let a = Tokenizer::train(&[&text], 350);
        let b = Tokenizer::train(&[&text], 350);
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn serialization_round_trip() {
        let text = training_text();
        let t = Tokenizer::train(&[&text], 400);
        let restored = Tokenizer::from_bytes(&t.to_bytes()).unwrap();
        assert_eq!(restored.vocab_size(), t.vocab_size());
        let sample = "the fire behind them throws the shadows.";
        assert_eq!(restored.encode(sample), t.encode(sample));
    }

    #[test]
    fn rejects_corrupt_tokenizer_files() {
        assert!(Tokenizer::from_bytes(b"nope").is_err());
        let mut bad = Tokenizer::byte_level().to_bytes();
        bad.extend_from_slice(&[0u8; 8]); // claims 0 merges, carries one
        assert!(Tokenizer::from_bytes(&bad).is_err());
        // A merge referring to a token that doesn't exist yet.
        let mut forward_ref = Vec::new();
        forward_ref.extend_from_slice(TOKENIZER_MAGIC);
        forward_ref.extend_from_slice(&TOKENIZER_VERSION.to_le_bytes());
        forward_ref.extend_from_slice(&1u32.to_le_bytes());
        forward_ref.extend_from_slice(&FIRST_MERGE_ID.to_le_bytes());
        forward_ref.extend_from_slice(&97u32.to_le_bytes());
        assert!(Tokenizer::from_bytes(&forward_ref).is_err());
    }

    #[test]
    fn every_token_id_is_decodable() {
        let text = training_text();
        let t = Tokenizer::train(&[&text], 400);
        for id in 0..t.vocab_size() as u32 {
            let piece = t.piece(id);
            if (256..FIRST_MERGE_ID).contains(&id) {
                assert!(piece.is_empty(), "special token {id} should render as nothing");
            } else {
                assert!(!piece.is_empty(), "token {id} decodes to nothing");
            }
        }
    }
}
