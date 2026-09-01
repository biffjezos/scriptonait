# Builds and serves the same frontend/ GitHub Pages gets (see
# .github/workflows/deploy.yml, which this mirrors) as a standalone
# container, for a host that runs a Dockerfile (Railway, say) instead of
# GitHub Pages. frontend/pkg/ (the compiled wasm module) and
# frontend/vendor/ (Bootstrap) are both gitignored and built fresh here,
# the same reason deploy.yml builds them fresh on every run rather than
# committing them — neither exists in this repo checkout on its own.
#
# Deliberately lighter than deploy.yml's own CI job: skips cargo test,
# the shader-check, and the JS-parses-as-ES-modules check (all already
# gated on the GitHub side before any commit reaches here) and keeps
# only the one cheap, high-value check — a malformed JSON format string
# in wasm-app would otherwise waste the entire wasm-pack build below
# finding out.
#
# llm-server (the GPU training backend) is a separate native binary,
# not part of this image at all — see crates/llm-server and
# .github/workflows/llm-server-build.yml. This container only ever
# serves the page; training still happens wherever llm-server runs.

FROM rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates unzip python3 \
    && rm -rf /var/lib/apt/lists/*

RUN rustup target add wasm32-unknown-unknown
# The installer script's own PATH detection doesn't always land on
# /usr/local/cargo/bin (this image's actual $CARGO_HOME) — it can put
# the binary under /root/.cargo/bin instead, which isn't on PATH here.
# Covering both rather than trusting one.
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
ENV PATH="/root/.cargo/bin:${PATH}"
RUN wasm-pack --version

WORKDIR /repo
COPY . .

RUN python3 tools/check-format-strings.py

# Vendored the same way deploy.yml does it: fetched here, not committed,
# never loaded from a CDN at runtime. Pinned to one version's official
# prebuilt dist archive so a Bootstrap release can't silently change
# this page.
ENV BOOTSTRAP_VERSION=5.3.3
RUN curl -sSLf -o /tmp/bootstrap.zip \
      "https://github.com/twbs/bootstrap/releases/download/v${BOOTSTRAP_VERSION}/bootstrap-${BOOTSTRAP_VERSION}-dist.zip" \
    && unzip -q /tmp/bootstrap.zip -d /tmp/bootstrap-dist \
    && mkdir -p frontend/vendor/bootstrap \
    && cp -r "/tmp/bootstrap-dist/bootstrap-${BOOTSTRAP_VERSION}-dist/css" frontend/vendor/bootstrap/css \
    && cp -r "/tmp/bootstrap-dist/bootstrap-${BOOTSTRAP_VERSION}-dist/js" frontend/vendor/bootstrap/js

# +simd128 is what lets LLVM turn llm-core's dot/axpy inner loops into
# real wasm SIMD instead of scalar float ops — same reason deploy.yml
# sets it. WebAssembly SIMD has been baseline in every major browser
# since 2023.
ENV RUSTFLAGS="-C target-feature=+simd128"
RUN wasm-pack build crates/wasm-app --release --target web --out-dir ../../frontend/pkg

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /site
COPY --from=builder /repo/frontend ./
COPY deploy/serve.py /serve.py

# Railway assigns PORT at runtime; serve.py reads it (falls back to
# 8080 for a local `docker run -p 8080:8080` test of this image).
CMD ["python3", "/serve.py"]
