# Corpus (Corpus tab)

## Adding text

- **Add Files** — file picker, multiple selection. Accepted: `.txt`, `.md`,
  `.markdown`, `.fountain`, `.text`. Plain text only; no PDF/DOCX parsing.
- **Paste** — a textarea plus a required name field. Same processing path
  as a file.
- Sources are stored in IndexedDB and reloaded on the next visit. Nothing
  is uploaded anywhere.
- Re-adding a file with the same title updates that source in place
  (preserving its sampling counters) rather than creating a duplicate.

## Source list

- One row per source: title, kind (`file`/`paste`), character count.
- **Filter by name** — text search, shown once the list exceeds a size
  threshold.
- **Remove Copies** — appears only when duplicate sources are detected;
  removes the extras.
- **Remove All** — clears every source, with a confirmation.
- Per-source **Remove**.

## Tokenizer

- Byte-level BPE (Byte-Pair Encoding), trained from scratch on the
  corpus at hand, sized to it — the vocabulary is the phrases the loaded
  text actually uses, not a fixed pretrained vocabulary.
  Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words
  with Subword Units*, 2016 (arXiv:1508.07909).
- Base alphabet is all 256 byte values plus special tokens (`BOS`,
  `EOS`, `TASK`, `STORY`), so any input encodes; there is no unknown
  token.
- Retrained whenever the corpus changes enough to warrant it; a model's
  checkpoint carries its own tokenizer, so a checkpoint is only valid
  against the vocabulary it was trained with.

## Held-out split

- 5% of every source's tokens are held out of training, used only to
  measure the model against text it has never trained on.
- The held-out set is a fixed, deterministic set of windows, evenly
  spaced through each source's held-out slice and identical on every
  measurement — two loss numbers differ because the weights differ, not
  because two random draws landed on different text.

## Corpus stats

- Character and token counts, per source and total, shown in the Corpus
  tab and summarized on Overview.
- Sources are classified by line shape into film scripts, novels and
  prose fiction, essays and philosophy, or verse and lyrics — a
  heuristic over line shapes and word endings, not a model — so the
  training plan (see [training.md](training.md)) can say a corpus is
  skewed toward one form.
