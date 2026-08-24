#!/usr/bin/env python3
"""Catch a Rust string literal that ended early inside a JSON format string.

Every JSON-producing `format!` in this workspace writes its keys as
`\\"key\\":`. When one of those backslashes goes missing the literal
closes at the key, the rest of the line becomes stray tokens, and the
file stops compiling — with an error naming a column rather than a cause.

That break cannot be caught here by building. llm-gpu and wasm-app need
crates.io, which this sandbox has no route to, so they compile for the
first time in CI: a deploy fails ten minutes after a push, over a missing
backslash. This is the check that catches it in a second.

The rule is exact rather than clever. `"key":{` appearing in source
outside a comment is JSON key syntax that escaped its literal — no valid
Rust writes a bare string followed by a colon and a brace. Adjacent
string literals in an array (`"and", "for"`) do not match it, which is
what a looser earlier version of this script got wrong.
"""
import re
import sys
from pathlib import Path

# A quoted identifier, then a colon, then an opening brace: a JSON key
# that is no longer inside the string it belongs to.
LEAKED_KEY = re.compile(r'"[A-Za-z_][A-Za-z0-9_]*"\s*:\s*\{')

status = 0
for path in sorted(Path("crates").rglob("*.rs")):
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        match = LEAKED_KEY.search(line)
        if match:
            print(f"{path}:{number}: a JSON key is outside its string literal — "
                  f"a backslash is missing before the quotes: {match.group(0)}")
            status = 1

print("format strings are well-formed" if status == 0 else "", end="")
sys.exit(status)
