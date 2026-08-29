//! Finding sources whose text is identical to an earlier one.

use std::collections::HashMap;

use super::sampling::fnv1a;
use super::Corpus;

impl Corpus {
    /// Source ids whose cleaned text is identical to an earlier source's.
    ///
    /// The same script added twice - a re-upload, the same file under two
    /// names - is trained on twice, which weights it double and inflates
    /// how well the model appears to do on it. Reported rather than
    /// removed: which copy to keep is the user's call.
    pub fn duplicate_sources(&self) -> Vec<String> {
        let mut seen: HashMap<u64, &str> = HashMap::new();
        let mut duplicates = Vec::new();
        for id in &self.order {
            let Some(text) = self.cleaned_text.get(id) else { continue };
            // A collision here would only mean one false report.
            let hash = fnv1a(text.as_bytes());
            match seen.get(&hash) {
                Some(_) => duplicates.push(id.clone()),
                None => {
                    seen.insert(hash, id.as_str());
                }
            }
        }
        duplicates
    }
}

#[cfg(test)]
mod tests {
    use super::super::Corpus;

    #[test]
    fn duplicate_sources_are_reported_not_removed() {
        let mut c = Corpus::new();
        c.upsert("a", "INT. KITCHEN - DAY\n\nJANE\nHi.", false);
        c.upsert("copy", "INT. KITCHEN - DAY\n\nJANE\nHi.", false);
        c.upsert("other", "EXT. GARDEN - NIGHT\n\nJOHN\nBye.", false);
        assert_eq!(c.duplicate_sources(), vec!["copy".to_string()]);
        assert_eq!(c.num_sources(), 3, "reporting a duplicate must not remove it");
    }
}
