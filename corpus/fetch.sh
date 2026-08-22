#!/usr/bin/env bash
#
# Download the public-domain corpus listed in manifest.tsv.
#
#   corpus/fetch.sh [manifest] [output-dir]
#
# Already-downloaded works are left alone, so this is cheap to re-run and
# safe to point at a cache. A work whose id fetches the wrong title is
# skipped rather than used — see the note in manifest.tsv.
set -uo pipefail

manifest=${1:-corpus/manifest.tsv}
out=${2:-corpus/raw}
# Below this, something is wrong with the network or with Gutenberg, and
# training on the handful that did arrive would waste hours of runner
# time producing a worse model than the last one.
min_works=${MIN_WORKS:-12}

mkdir -p "$out"
got=0
missed=0

while IFS=$'\t' read -r form id slug expect || [ -n "${form:-}" ]; do
  case "$form" in ''|'#'*) continue ;; esac
  dest="$out/${form}__${slug}.txt"
  if [ -s "$dest" ]; then
    got=$((got + 1))
    continue
  fi

  tmp=$(mktemp)
  fetched=""
  # Gutenberg serves the plain-text copy under a couple of paths
  # depending on how old the work's files are.
  for url in \
    "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt" \
    "https://www.gutenberg.org/files/${id}/${id}-0.txt" \
    "https://www.gutenberg.org/files/${id}/${id}.txt"
  do
    if curl -fsSL --max-time 120 --retry 3 --retry-delay 3 -o "$tmp" "$url"; then
      fetched=$url
      break
    fi
  done

  if [ -z "$fetched" ]; then
    echo "MISS  ${slug} (id ${id}): no plain-text copy fetched"
    missed=$((missed + 1))
    rm -f "$tmp"
    continue
  fi

  # The check that makes a wrong id cheap instead of poisonous.
  if ! head -c 4000 "$tmp" | grep -qiF -- "$expect"; then
    echo "SKIP  ${slug} (id ${id}): expected a title containing '${expect}'"
    missed=$((missed + 1))
    rm -f "$tmp"
    continue
  fi

  mv "$tmp" "$dest"
  echo "ok    ${form}__${slug} ($(wc -c < "$dest") bytes)"
  got=$((got + 1))
done < "$manifest"

echo
echo "$got works available, $missed unavailable"
if [ "$got" -lt "$min_works" ]; then
  echo "only $got works, need at least $min_works — refusing to build a corpus this thin" >&2
  exit 1
fi
