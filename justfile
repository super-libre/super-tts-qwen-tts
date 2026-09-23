# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Qwen TTS backend. Mirrors the recipe names
# used by the main super-tts repo and by the other out-of-tree backends, so
# `just check` means the same thing everywhere.
#
# One thing is specific to this backend: Burn comes from a fork, pinned by
# revision in Cargo.toml, because the Qwen3-TTS port lives there and not in
# upstream Burn. The first build fetches and compiles it, which is slow;
# nothing else about the build is unusual and there is no submodule and no C
# toolchain to install.
#
# Every recipe that builds passes `--no-default-features` with one accelerator
# named, matching the release workflow: features are additive, so a build that
# kept the default `flex` alongside `cuda` would carry two backends.

tokenizer_url := "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
tokenizer_dir := justfile_directory() / "target/test-backend"

# Default: build release
default: build-release

# Compiles with debug profile. Usage: just build-debug [args]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile — the pure-Rust CPU backend.
# Usage: just build-release [args]
build-release *args:
    cargo build --release --locked {{ args }}

# Build with CUDA. Needs the CUDA toolkit headers on the host; no GPU and no
# compute capability are needed, since CubeCL compiles the kernels at runtime.
build-cuda *args:
    cargo build --release --locked --no-default-features --features cuda {{ args }}

# Build with ROCm. Needs the ROCm headers `cubecl-hip-sys` binds against.
build-rocm *args:
    cargo build --release --locked --no-default-features --features rocm {{ args }}

# Build with Vulkan — the vendor-neutral GPU path. Needs no SDK to build; the
# loader is found at runtime.
build-vulkan *args:
    cargo build --release --locked --no-default-features --features vulkan {{ args }}

# Build with Metal, on macOS. Needs nothing beyond the SDK Xcode's command line
# tools carry; the kernels are compiled to MSL at runtime.
build-metal *args:
    cargo build --release --locked --no-default-features --features metal {{ args }}

# Cargo already names the artifact as `backend.toml`'s entrypoint, and the
# release workflow tarballs it under the same name, so a local install and a
# published one stage the same bytes.
#
# Build and stage the binary for Import-from-dir. Usage: just stage [args]
stage *args: (build-release args)
    cp target/release/super-tts-backend-qwen-tts super-tts-backend-qwen-tts
    @echo "staged super-tts-backend-qwen-tts — this directory is now installable with Import from dir"

# Remove build output.
clean:
    cargo clean

# `cargo clean` only touches target/. These four are the rest of what a build
# leaves behind, and the only build artifacts .gitignore lists beside it: the
# binary `stage` copies to the repo root, the lcov report `coverage-lcov`
# writes, and the profile data coverage runs can drop in the working directory.
#
# Remove build output and every generated artifact in the tree.
clean-all: clean
    rm -f super-tts-backend-qwen-tts lcov.info *.profraw *.profdata

# Runs a clippy check — mirrors super-tts's lint.
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# The suite needs no weights: everything that would need them is behind the
# model load, and what is tested here is the wire format, the language and voice
# joins, and the prompt template. One test does want the tokenizer; it skips
# without it, see `test-tokenizer`.
#
# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# The daemon fetches this file in production; it is 11 MB and deliberately not
# in the repository.
#
# Fetch `tokenizer.json` so the reference tokenization test runs instead of
# skipping.
fetch-tokenizer:
    #!/usr/bin/env bash
    set -euo pipefail
    dest="{{ tokenizer_dir }}/tokenizer.json"
    if [ -f "$dest" ]; then echo "already present: $dest"; exit 0; fi
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 3 -o "$dest" "{{ tokenizer_url }}"
    echo "fetched $dest"

# Run the whole suite including the reference tokenization fixture.
test-tokenizer *args: fetch-tokenizer
    cargo test --locked {{ args }}

# cross-rs builds inside a container, so no local CUDA or C toolchain is needed.
#
# Cross-compile for a target. Usage: just cross-build <target>
cross-build target="x86_64-unknown-linux-gnu":
    cross build --release --locked --target {{ target }}

# --remap-path-prefix keeps report paths relative (src/...), and tests/ is
# excluded so only product code is counted.
#
# Measure coverage, requires cargo-llvm-cov. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# No doctests: this is a binary-only crate, so `cargo test --doc` has no lib
# target.
#
# Full local CI gate: format, lint, build, test.
ci: fmt-check check build-release test
