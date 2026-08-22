//! See Cargo.toml: this crate exists only to carry the shader-validation
//! test in `tests/shaders.rs`, which compiles every WGSL shader in
//! `crates/llm-gpu/src/shaders` with naga so a broken shader fails CI
//! instead of failing in a user's browser.
