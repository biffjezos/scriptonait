//! Text preparation: turns raw source text (pasted, uploaded, or fetched
//! from a URL) into clean training text.

use crate::tokenizer;

/// Very small, dependency-free HTML-to-text pass: drops `<script>`/`<style>`
/// bodies entirely, strips all other tags, and decodes the handful of named
/// entities that show up in ordinary web pages. It is intentionally not a
/// full HTML parser — good enough to keep markup out of the training
/// corpus, not a general-purpose scraper.
pub fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();

    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip <script ...>...</script> and <style ...>...</style> bodies.
            let lower_tail = input[i..].to_ascii_lowercase();
            if let Some(skip_to) = skip_raw_element(&lower_tail, "script")
                .or_else(|| skip_raw_element(&lower_tail, "style"))
            {
                i += skip_to;
                continue;
            }
            // Skip a normal tag.
            if let Some(close) = input[i..].find('>') {
                i += close + 1;
                continue;
            } else {
                // Unterminated '<': treat rest as text.
                out.push_str(&input[i..]);
                break;
            }
        }
        // Copy one char.
        let ch_len = input[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&input[i..i + ch_len]);
        i += ch_len;
    }

    decode_entities(&out)
}

fn skip_raw_element(lower_tail: &str, tag: &str) -> Option<usize> {
    let open = format!("<{tag}");
    if !lower_tail.starts_with(&open) {
        return None;
    }
    let close_tag = format!("</{tag}>");
    lower_tail
        .find(&close_tag)
        .map(|pos| pos + close_tag.len())
}

fn decode_entities(input: &str) -> String {
    input
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

/// Collapse runs of whitespace within each line, collapse blank-line runs
/// down to at most one, and trim trailing whitespace — but keep each
/// line's *leading* indentation. Plain-text script exports (Fountain,
/// Final Draft "print to text", etc.) commonly use indentation as the only
/// signal separating scene headings, character cues, dialogue, and action
/// lines, so stripping it would destroy structure the model could
/// otherwise learn.
pub fn normalize_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut blank_run = 0u32;
    for line in input.lines() {
        let cleaned = clean_line(line);
        if cleaned.is_empty() {
            blank_run += 1;
            if blank_run <= 1 && !out.is_empty() {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&cleaned);
        }
    }
    out.trim_end().to_string()
}

/// Clean one line: expand tabs and cap indentation width, collapse runs of
/// internal whitespace in the line body to a single space, trim trailing
/// whitespace. Returns an empty string for blank/whitespace-only lines.
fn clean_line(line: &str) -> String {
    let indent_end = line
        .char_indices()
        .find(|(_, c)| *c != ' ' && *c != '\t')
        .map(|(i, _)| i)
        .unwrap_or(line.len());
    let indent_width: usize = line[..indent_end]
        .chars()
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum::<usize>()
        .min(40);
    let body = collapse_spaces(line[indent_end..].trim_end());
    if body.is_empty() {
        String::new()
    } else {
        format!("{}{}", " ".repeat(indent_width), body)
    }
}

fn collapse_spaces(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        let is_space = c.is_whitespace();
        if is_space {
            if !prev_space {
                result.push(' ');
            }
        } else {
            result.push(c);
        }
        prev_space = is_space;
    }
    result
}

/// Statistics returned after preparing a source, used by the UI to show the
/// user what happened without re-deriving it in JS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedStats {
    pub char_count: usize,
    pub byte_count: usize,
    pub token_count: usize,
}

/// Clean raw source text and tokenize it in one step.
pub fn prepare(raw: &str, is_html: bool) -> (String, Vec<u32>, PreparedStats) {
    let cleaned = if is_html {
        normalize_whitespace(&strip_html(raw))
    } else {
        normalize_whitespace(raw)
    };
    let tokens = tokenizer::encode(&cleaned);
    let stats = PreparedStats {
        char_count: cleaned.chars().count(),
        byte_count: cleaned.len(),
        token_count: tokens.len(),
    };
    (cleaned, tokens, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_basic_tags() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b>!</p></body></html>";
        assert_eq!(strip_html(html), "TitleHello world!");
    }

    #[test]
    fn drops_script_and_style_bodies() {
        let html = "<p>keep</p><script>var x = 1 < 2;</script><style>.a{color:red}</style><p>me</p>";
        assert_eq!(strip_html(html), "keepme");
    }

    #[test]
    fn decodes_common_entities() {
        assert_eq!(strip_html("Fish &amp; Chips &mdash;? &lt;tag&gt;"), "Fish & Chips &mdash;? <tag>");
    }

    #[test]
    fn normalizes_internal_whitespace_and_blank_lines() {
        let input = "  Hello   world  \n\n\n\nSecond   paragraph.\n\n";
        // Leading indentation is preserved (2 spaces); internal runs of
        // whitespace collapse and excess blank lines collapse to one.
        assert_eq!(normalize_whitespace(input), "  Hello world\n\nSecond paragraph.");
    }

    #[test]
    fn preserves_screenplay_indentation() {
        // A typical plain-text screenplay export: scene heading flush
        // left, character cue and dialogue indented differently.
        let input = "INT. KITCHEN - DAY\n\n          JANE\n     Where were you?\n";
        let out = normalize_whitespace(input);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "INT. KITCHEN - DAY");
        assert_eq!(lines[2], "          JANE");
        assert_eq!(lines[3], "     Where were you?");
    }

    #[test]
    fn tabs_are_expanded_and_indentation_capped() {
        assert_eq!(clean_line("\tfoo"), "    foo");
        let huge_indent = " ".repeat(500) + "x";
        assert_eq!(clean_line(&huge_indent), " ".repeat(40) + "x");
    }

    #[test]
    fn prepare_end_to_end() {
        let (text, tokens, stats) = prepare("<p>Hi   there</p>", true);
        assert_eq!(text, "Hi there");
        assert_eq!(tokens.len(), stats.token_count);
        assert_eq!(stats.byte_count, "Hi there".len());
        assert_eq!(stats.char_count, 8);
    }

    #[test]
    fn prepare_plain_text_is_not_html_stripped() {
        let (text, _, _) = prepare("a < b and b > c", false);
        assert_eq!(text, "a < b and b > c");
    }
}
