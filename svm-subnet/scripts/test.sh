#!/bin/bash
set -e

echo "=== Running SVM-Subnet Tests ==="

# Run Rust tests
echo "Running Rust tests..."
cd runtime && cargo test
cd ..

# Run Go tests
echo "Running Go tests..."
make test

echo "=== All tests passed ==="
