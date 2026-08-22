//! A pre-tokenized training set on disk.
//!
//! Tokenizing tens of MB of text takes a while, and a pretraining run
//! that spans several CI jobs would otherwise redo it on every restart.
//! `llm-data` writes this file once; `llm-train` reads it and gets
//! straight to work.
//!
//! Documents are kept separate rather than concatenated into one stream
//! because the boundaries matter: `Corpus::sample_batch` deliberately
//! starts a share of its windows at a document's first token, and for an
//! instruction example that is the whole point — the model only learns
//! that the `TASK` line governs what follows if it reliably sees the
//! `TASK` line at the start of a window.
//!
//! ```text
//!   magic  "SCDS"
//!   u32    format version
//!   u32    document count
//!   u32    per document: its length in tokens
//!   u32    every document's tokens, in order
//! ```

const MAGIC: &[u8; 4] = b"SCDS";
const VERSION: u32 = 1;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenDataset {
    pub documents: Vec<Vec<u32>>,
}

impl TokenDataset {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, tokens: Vec<u32>) {
        if !tokens.is_empty() {
            self.documents.push(tokens);
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.documents.iter().map(Vec::len).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.documents.len() * 4 + self.total_tokens() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.documents.len() as u32).to_le_bytes());
        for doc in &self.documents {
            out.extend_from_slice(&(doc.len() as u32).to_le_bytes());
        }
        for doc in &self.documents {
            for &token in doc {
                out.extend_from_slice(&token.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != MAGIC {
            return Err("not a scriptonait dataset file".to_string());
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("dataset format version {version}, expected {VERSION}"));
        }
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let lengths_end = 12 + count * 4;
        if bytes.len() < lengths_end {
            return Err("dataset truncated in its length table".to_string());
        }
        let lengths: Vec<usize> = (0..count)
            .map(|i| {
                let off = 12 + i * 4;
                u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize
            })
            .collect();
        let total: usize = lengths.iter().sum();
        if bytes.len() != lengths_end + total * 4 {
            return Err(format!(
                "dataset claims {total} tokens across {count} documents, which needs {} bytes, but the file is {}",
                lengths_end + total * 4,
                bytes.len()
            ));
        }
        let mut at = lengths_end;
        let mut documents = Vec::with_capacity(count);
        for len in lengths {
            let mut doc = Vec::with_capacity(len);
            for _ in 0..len {
                doc.push(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
                at += 4;
            }
            documents.push(doc);
        }
        Ok(Self { documents })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut dataset = TokenDataset::new();
        dataset.push(vec![257, 1, 2, 3, 258]);
        dataset.push(vec![257, 259, 40, 41, 260, 7, 8, 258]);
        let restored = TokenDataset::from_bytes(&dataset.to_bytes()).unwrap();
        assert_eq!(restored, dataset);
        assert_eq!(restored.total_tokens(), 13);
    }

    #[test]
    fn empty_documents_are_dropped() {
        let mut dataset = TokenDataset::new();
        dataset.push(vec![]);
        assert_eq!(dataset.documents.len(), 0);
    }

    #[test]
    fn rejects_corrupt_files() {
        assert!(TokenDataset::from_bytes(b"nope").is_err());
        let mut dataset = TokenDataset::new();
        dataset.push(vec![1, 2, 3]);
        let bytes = dataset.to_bytes();
        assert!(TokenDataset::from_bytes(&bytes[..bytes.len() - 4]).is_err());
    }
}
