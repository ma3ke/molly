#!/bin/bash
set -x
set -e

trajectory=$1
decimals=$2

# Build all examples to prevent running stale executables.
cargo build --release --examples

# Verify that the output is equal between xdrfile and ours.
time target/release/examples/compare $trajectory $decimals

hyperfine --shell=none --warmup 2 \
	"target/release/examples/reader    $trajectory" \
	"target/release/examples/reader    $trajectory :: buffered" \
	"target/release/examples/xdrreader $trajectory" \
	"target/release/examples/cfreader  $trajectory"
