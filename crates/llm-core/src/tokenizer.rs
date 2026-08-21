//! Byte-level tokenizer.
//!
//! There is no BPE training step: every possible input (any UTF-8 text, or
//! even arbitrary bytes pasted/uploaded by the user) maps to a sequence of
//! its raw bytes, each treated as one token. This keeps the vocabulary tiny
//! (256 byte values + 3 special tokens = 259), which keeps every embedding
//! and per-layer-embedding table small regardless of corpus size or
//! language, and it never produces an "unknown token".

pub const PAD: u32 = 256;
pub const BOS: u32 = 257;
pub const EOS: u32 = 258;
pub const VOCAB_SIZE: usize = 259;

/// Encode text as raw UTF-8 bytes, one token per byte.
pub fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// Decode a token stream back to a string. Special tokens (PAD/BOS/EOS) are
/// dropped; any resulting invalid UTF-8 (e.g. from a window that cuts a
/// multi-byte character in half) is replaced per the standard lossy rules.
pub fn decode(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .copied()
        .filter(|&id| id < 256)
        .map(|id| id as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

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
    fn round_trip_ascii() {
        let s = "Hello, world! This is a script.";
        assert_eq!(decode(&encode(s)), s);
    }

    #[test]
    fn round_trip_utf8() {
        let s = "caf\u{e9} \u{1f980} \u{65e5}\u{672c}\u{8a9e}";
        assert_eq!(decode(&encode(s)), s);
    }

    #[test]
    fn special_tokens_outside_byte_range() {
        assert!(BOS >= 256);
        assert!(EOS >= 256);
        assert!(PAD >= 256);
        assert_eq!(VOCAB_SIZE, 259);
    }

    #[test]
    fn decode_ignores_special_tokens() {
        let ids = vec![BOS, b'h' as u32, b'i' as u32, EOS, PAD];
        assert_eq!(decode(&ids), "hi");
    }

    #[test]
    fn empty_input() {
        assert_eq!(encode(""), Vec::<u32>::new());
        assert_eq!(decode(&[]), "");
    }
}
