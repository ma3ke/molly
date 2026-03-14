# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Molly is a pure-Rust library for reading and writing Gromacs XTC (molecular dynamics trajectory) files. It includes an optional CLI tool, Python bindings via PyO3, and benchmarks.

## Commands

```bash
# Build
cargo build --release
cargo build --release -F cli          # with CLI tool

# Test (release mode is recommended — some tests are slow in debug)
cargo test --release
cargo test --release <test_name>      # single test by name
cargo test --release -F cli           # with CLI feature

# Benchmarks
cargo bench

# Lint
cargo clippy
cargo semver-checks                   # check for SemVer breakage

# Python bindings
cd bindings/python && pip install . && python tests/verification.py
```

## Architecture

The library exposes `XTCReader<R: Read + Seek>` and `XTCWriter<W: Write>` as the primary public API (defined in `src/lib.rs`).

**Core data types**
- `Frame` — one trajectory snapshot: `step`, `time`, `boxvec: [f32; 9]`, `precision: f32`, `positions: Vec<f32>` (flat 3N array)
- `Header` — per-frame metadata (magic, natoms, step, time, boxvec)
- `Magic` — XTC format version: `1995` (32-bit frame sizes) or `2023` (64-bit frame sizes)
- `AtomSelection` / `FrameSelection` — filtering enums (`All`, `Mask`, `Until`, `Range`, `FrameList`)

**Module responsibilities**
| File | Responsibility |
|---|---|
| `src/lib.rs` | Public API, `XTCReader`, `XTCWriter`, `Frame`, `Header` |
| `src/reader.rs` | XTC decompression; `MAGICINTS` lookup table; `read_compressed_positions` |
| `src/writer.rs` | XTC compression; `write_compressed_positions`, `write_frame_parts` |
| `src/selection.rs` | `AtomSelection` / `FrameSelection` — controls which atoms/frames are read |
| `src/buffer.rs` | `UnBuffered` vs `Buffer` (128 KB blocks) I/O strategy |
| `src/main.rs` | CLI tool (feature-gated with `cli`) |

**I/O buffering strategy**: `UnBuffered` streams directly (fastest for full reads); `Buffer` loads 128 KB blocks (preferred when reading a small atom subset — reduces disk I/O at ~20% speed cost).

**Reading flow**: `read_frame` → parse header → for ≤9 atoms read raw f32s, otherwise decompress via `MAGICINTS` bit-unpacking → apply `AtomSelection` to produce positions slice.

**Writing flow**: `write_frame` / `write_frame_parts` → write header → for ≤9 atoms write raw f32s, otherwise float→int via precision, find per-axis min/max, bit-encode coordinates, write padding to XDR 4-byte boundary.

## Tests

Integration tests live in `tests/`:
- `write.rs` — roundtrip (read → write → read) correctness
- `selections.rs` — atom/frame selection behavior
- `compare.rs` — regression against the xdrfile C library
- `home.rs` — seek/position correctness
- `open.rs` — file parsing edge cases
- `steps.rs` — frame stepping

Test trajectories (defined in `tests/common/mod.rs`) are not committed; they must be present locally. Tests that require them will fail if the files are absent.
