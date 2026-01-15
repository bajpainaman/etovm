#!/bin/bash
set -e

echo "=== Building SVM-Subnet ==="

# Check dependencies
command -v go >/dev/null 2>&1 || { echo "Go is required but not installed."; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "Rust/Cargo is required but not installed."; exit 1; }

# Build
make build

echo "=== Build complete ==="
echo "Artifacts in ./build/"
