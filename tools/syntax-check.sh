#!/usr/bin/env bash
# Parse every Rust file in the crates that cannot be compiled here.
#
# `cargo test --workspace` builds llm-core and nothing else: llm-gpu and
# wasm-app need crates.io, which the development sandbox has no route
# to. So a syntax error in either of them compiles for the first time in
# CI, ten minutes after a push, and a deploy fails on something that a
# parser would have caught instantly.
#
# `rustc --emit=metadata` on a file with no dependencies available still
# parses it first and only then fails on name resolution, so filtering
# resolution errors out leaves exactly the syntax errors. It is not a
# type check and does not pretend to be; it is the gate that catches an
# escaped quote in a format string.
set -uo pipefail
status=0
for file in $(find crates/wasm-app/src crates/llm-gpu/src -name '*.rs'); do
  output=$(rustc --edition 2021 --crate-type lib --emit=metadata -o /dev/null "$file" 2>&1 \
    | grep -E '^error(\[E0(4[0-9]{2}|5[0-9]{2})\])?:' \
    | grep -vE 'E0432|E0433|E0463|unresolved|unlinked|can.t find crate' || true)
  if [ -n "$output" ]; then
    echo "$file:"
    echo "$output"
    status=1
  fi
done
if [ "$status" -eq 0 ]; then echo "wasm-app and llm-gpu parse"; fi
exit "$status"
