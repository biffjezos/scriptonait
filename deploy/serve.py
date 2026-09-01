#!/usr/bin/env python3
"""Static file server for the built frontend/, for a container runtime
(Railway or similar) — not part of the GitHub Pages pipeline, which
serves the same directory a different way (see .github/workflows/
deploy.yml). Reads PORT from the environment, the way Railway's own
routing expects; falls back to 8080 for a local `docker run -p 8080:8080`
test.

.wasm gets an explicit MIME type: without it, some Python builds guess
the wrong one, and wasm-pack's --target web output uses fetch() +
WebAssembly.instantiateStreaming, which needs `application/wasm` to
actually stream-compile instead of silently falling back to a slower
path.
"""
import http.server
import mimetypes
import os

mimetypes.add_type('application/wasm', '.wasm')

PORT = int(os.environ.get('PORT', 8080))
Handler = http.server.SimpleHTTPRequestHandler

with http.server.ThreadingHTTPServer(('0.0.0.0', PORT), Handler) as httpd:
    print(f'Serving {os.getcwd()} on 0.0.0.0:{PORT}', flush=True)
    httpd.serve_forever()
