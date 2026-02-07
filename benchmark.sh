#!/bin/bash
set -x
set -e

trajectory=$1

# Build all examples to prevent running stale executables.
cargo build --release --examples

# Verify that the output is equal between xdrfile and ours.
time target/release/examples/compare   $trajectory

time target/release/examples/cfreader  $trajectory
time target/release/examples/xdrreader $trajectory
time target/release/examples/reader    $trajectory
