//! Source text and vocabulary management: everything that edits or
//! reports on the corpus a model trains against.

use wasm_bindgen::prelude::*;

use llm_core::model::ModelWeights;

use crate::dto::{json_string, SourceStats};
use crate::WasmLLM;

#[wasm_bindgen]
impl WasmLLM {
    /// Cleans and tokenizes `raw_text`, storing (or replacing, if `id`
    /// already exists) it as a source. `is_html` should be true for text
    /// fetched from a URL, false for pasted or uploaded plain text.
    pub fn upsert_source(&self, id: String, raw_text: String, is_html: bool) -> SourceStats {
        let stats = self.0.borrow_mut().corpus.upsert(&id, &raw_text, is_html);
        SourceStats {
            char_count: stats.char_count as u32,
            byte_count: stats.byte_count as u32,
            token_count: stats.token_count as u32,
        }
    }

    pub fn remove_source(&self, id: String) -> bool {
        self.0.borrow_mut().corpus.remove(&id)
    }

    pub fn num_sources(&self) -> u32 {
        self.0.borrow().corpus.num_sources() as u32
    }

    pub fn total_tokens(&self) -> f64 {
        self.0.borrow().corpus.total_tokens() as f64
    }

    /// Per-source token counts and how many training windows have been
    /// drawn from each, as JSON — for showing which sources training has
    /// actually used, not just which sources exist. Read-only and never
    /// touches the GPU, so it needs no `busy` guard.
    pub fn corpus_source_stats(&self) -> String {
        let stats = self.0.borrow_mut().corpus.per_source_stats();
        let rows = stats
            .iter()
            .map(|s| {
                format!(
                    "{{\"id\":{},\"trainTokens\":{},\"heldOutTokens\":{},\"sampled\":{}}}",
                    json_string(&s.id), s.train_tokens, s.held_out_tokens, s.sampled,
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{rows}]")
    }

    /// Restore a source's persisted sample count after a fresh page load
    /// re-upserts it into a new corpus — see `Corpus::set_sample_count`.
    pub fn set_source_sample_count(&self, id: String, count: f64) {
        self.0.borrow_mut().corpus.set_sample_count(&id, count as u64);
    }

    /// How often a sampled training window starts exactly at a source's
    /// beginning rather than at the next window due in its rotation —
    /// see `Corpus::boundary_sample_rate`. A training setting, not a
    /// fixed constant, because how much of a source's opening is front
    /// matter (a title page, a table of contents) rather than prose
    /// varies by corpus.
    pub fn boundary_sample_rate(&self) -> f32 {
        self.0.borrow().corpus.boundary_sample_rate()
    }

    pub fn set_boundary_sample_rate(&self, rate: f32) {
        self.0.borrow_mut().corpus.set_boundary_sample_rate(rate);
    }

    /// One source's progress through its own shuffled pass over its
    /// training windows, as `{"epoch":n,"cursor":n}`, or `null` if
    /// nothing has been drawn from it yet — for persisting so a reload
    /// resumes that pass instead of restarting it (see
    /// `Corpus::window_progress`).
    pub fn window_progress(&self, id: String) -> String {
        match self.0.borrow().corpus.window_progress(&id) {
            Some((epoch, cursor)) => format!("{{\"epoch\":{epoch},\"cursor\":{cursor}}}"),
            None => "null".to_string(),
        }
    }

    /// Every source's window-pass progress that exists yet, as JSON —
    /// for writing it all back to storage in one pass rather than one
    /// round trip per source. See `Corpus::all_window_progress`.
    pub fn corpus_window_progress(&self) -> String {
        let entries = self.0.borrow().corpus.all_window_progress();
        let rows = entries
            .iter()
            .map(|(id, epoch, cursor)| format!("{{\"id\":{id:?},\"epoch\":{epoch},\"cursor\":{cursor}}}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{rows}]")
    }

    /// Restore a source's window-pass progress after a fresh page load
    /// re-upserts it into a new corpus — see `Corpus::set_window_progress`.
    pub fn set_window_progress(&self, id: String, epoch: u32, cursor: u32) {
        self.0.borrow_mut().corpus.set_window_progress(&id, epoch, cursor);
    }

    /// Learn a BPE vocabulary from the loaded sources and re-encode them
    /// with it. Returns the new vocabulary size, or 0 when there is no
    /// text yet.
    ///
    /// `max_vocab_size` is a ceiling. What is actually learned scales with
    /// how much text this visitor loaded — their vocabulary, their model,
    /// their machine; nothing here is shared with anyone else's session.
    ///
    /// Must happen before a model is created: the vocabulary size fixes
    /// the embedding table. Without it every token is one byte, which
    /// costs about four times the tokens - and therefore four times the
    /// training time - for the same text.
    pub fn learn_vocabulary(&self, max_vocab_size: u32) -> Result<u32, JsValue> {
        self.acquire()?;
        let result = self.learn_vocabulary_inner(max_vocab_size);
        self.release();
        Ok(result)
    }

    /// Titles of sources whose text duplicates an earlier one. Training
    /// on the same script twice weights it double.
    pub fn duplicate_sources(&self) -> Vec<String> {
        self.0.borrow().corpus.duplicate_sources()
    }

    pub fn vocab_size(&self) -> u32 {
        self.0.borrow().corpus.tokenizer().vocab_size() as u32
    }

    /// Whether a training step can run: enough source text to fill a
    /// context window, and a GPU to run it on.
    pub fn can_train(&self) -> bool {
        let inner = &mut *self.0.borrow_mut();
        if inner.gpu.is_none() {
            return false;
        }
        let context_len = inner.config.context_len;
        inner.corpus.can_sample(context_len)
    }

    /// Whether this browser gave us a device to train on at all. The
    /// page uses this to explain itself when `can_train` is false.
    pub fn has_gpu(&self) -> bool {
        self.0.borrow().gpu.is_some()
    }

    /// Runs `llm_core::qa::check_generated` against `text`, returning
    /// each note as `"[INFO] ..."`/`"[WARNING] ..."`.
    /// `target_word_count = 0` means "no target" (skips the length
    /// check).
    pub fn qa_check(&self, text: String, target_word_count: u32) -> Vec<String> {
        let target = if target_word_count == 0 { None } else { Some(target_word_count as usize) };
        llm_core::qa::check_generated(&text, target)
            .into_iter()
            .map(|note| {
                let prefix = match note.severity {
                    llm_core::qa::Severity::Info => "INFO",
                    llm_core::qa::Severity::Warning => "WARNING",
                };
                format!("[{prefix}] {}", note.message)
            })
            .collect()
    }
}

impl WasmLLM {
    fn learn_vocabulary_inner(&self, max_vocab_size: u32) -> u32 {
        let inner = &mut *self.0.borrow_mut();
        let current = inner.corpus.tokenizer().vocab_size() as u32;
        // A trained model's weights are indexed by the vocabulary that
        // trained them: changing it would make every token id mean
        // something else.
        if inner.step > 0 || inner.pretrained {
            return current;
        }
        let Some(size) = inner.corpus.learn_vocabulary(max_vocab_size as usize) else {
            return current;
        };
        // The embedding table is one row per token, so a new vocabulary
        // is a new model. It has not been trained yet, so nothing is lost.
        inner.config.vocab_size = size;
        inner.weights = ModelWeights::init(&inner.config, inner.seed);
        // Both the uploaded generation weights and the resident training
        // state belong to the old shape.
        inner.gpu = None;
        size as u32
    }
}
