//! Parses and validates every WGSL shader in `crates/llm-gpu/src/shaders`.
//!
//! naga is the same front end wgpu uses to compile these, so anything
//! rejected here would have been rejected at `create_shader_module` time —
//! which, on the web, is when the page first tries to use the GPU.
//! Catching it in CI instead is the entire point of this test.
//!
//! Validation covers more than syntax: type checking, binding
//! declarations, and the uniformity analysis that rejects a
//! `workgroupBarrier()` reachable from non-uniform control flow (a real
//! hazard in the tiled matmul kernels, which barrier inside their tile
//! loop).

use std::path::{Path, PathBuf};

fn shader_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../llm-gpu/src/shaders")
}

fn shader_paths() -> Vec<PathBuf> {
    let dir = shader_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("readable dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn every_shader_parses_and_validates() {
    let paths = shader_paths();
    assert!(!paths.is_empty(), "found no .wgsl files in {}", shader_dir().display());

    let mut failures = Vec::new();
    for path in &paths {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let source = std::fs::read_to_string(path).expect("shader is readable");

        let module = match naga::front::wgsl::parse_str(&source) {
            Ok(module) => module,
            Err(err) => {
                failures.push(format!("{name}: parse error\n{}", err.emit_to_string(&source)));
                continue;
            }
        };

        // Default capabilities, not `all()`: these shaders run in a
        // browser against baseline WebGPU, so validating against a
        // superset of that would accept things Chrome would reject.
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        if let Err(err) = validator.validate(&module) {
            failures.push(format!("{name}: validation error\n{}", err.emit_to_string(&source)));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} shader(s) failed to compile:\n\n{}",
        failures.len(),
        paths.len(),
        failures.join("\n\n")
    );
}

#[test]
fn every_shader_referenced_by_the_backend_exists() {
    // Guards against a shader being renamed on disk but not in
    // context.rs's include_str! calls (which would fail the llm-gpu build,
    // but with a message about a missing file rather than a missing kernel).
    let context_rs = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../llm-gpu/src/context.rs"),
    )
    .expect("llm-gpu/src/context.rs is readable");

    for path in shader_paths() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            context_rs.contains(&format!("shaders/{name}")),
            "{name} exists but no pipeline in context.rs includes it"
        );
    }
}
