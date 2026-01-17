.PHONY: all build build-runtime build-vm build-tools test clean install genesis fmt lint help

GOPATH ?= $(shell go env GOPATH)
GOBIN ?= $(GOPATH)/bin
BUILD_DIR = build

# VM ID for Fuji testnet (derived from subnet creation)
VM_ID = srLZEW2t7KxW8xzjgh6Nq3FX6aPVrJav283g5T5YU9NSp1xC1

# AvalancheGo paths
AVALANCHEGO_PATH ?= $(HOME)/.avalanchego
PLUGINS_PATH = $(AVALANCHEGO_PATH)/plugins

all: build

# Build everything
build: build-runtime build-vm build-tools

# Build Rust runtime as static/shared library
build-runtime:
	@echo "Building Rust runtime..."
	cd runtime && cargo build --release
	@mkdir -p $(BUILD_DIR)
	@echo "Rust runtime built successfully"

# Build Go VM plugin
build-vm: build-runtime
	@echo "Building Go VM plugin..."
	CGO_ENABLED=1 go build -o $(BUILD_DIR)/svm ./cmd/svm
	@echo "VM plugin built: $(BUILD_DIR)/svm"

# Build CLI tools
build-tools:
	@echo "Building CLI tools..."
	go build -o $(BUILD_DIR)/genesis-gen ./cmd/genesis
	@echo "Tools built successfully"

# Generate genesis.json
genesis: build-tools
	@echo "Generating genesis.json..."
	$(BUILD_DIR)/genesis-gen -output $(BUILD_DIR)/genesis.json
	@echo "Genesis generated: $(BUILD_DIR)/genesis.json"

# Install VM to AvalancheGo plugins directory
install: build
	@echo "Installing VM plugin..."
	mkdir -p $(PLUGINS_PATH)
	cp $(BUILD_DIR)/svm $(PLUGINS_PATH)/$(VM_ID)
	@echo "VM installed to $(PLUGINS_PATH)/$(VM_ID)"
	@echo ""
	@echo "Copy genesis.json to your subnet configuration."

# Run tests
test: test-rust test-go

test-rust:
	@echo "Running Rust tests..."
	cd runtime && cargo test

test-go: build-runtime
	@echo "Running Go tests..."
	CGO_ENABLED=1 go test -v ./...

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	cd runtime && cargo clean
	go clean

# Format code
fmt:
	go fmt ./...
	cd runtime && cargo fmt

# Lint
lint:
	golangci-lint run || true
	cd runtime && cargo clippy

# Development: clean, build, and install
dev: clean build install genesis
	@echo "Development build complete"

# Run benchmarks
bench: build-runtime
	@echo "Running Turbo benchmark..."
	cd runtime && RUSTFLAGS="-C target-cpu=native" cargo build --release --bin turbo_bench
	./runtime/target/release/turbo_bench

bench-qmdb: build-runtime
	@echo "Running QMDB benchmark..."
	cd runtime && RUSTFLAGS="-C target-cpu=native" cargo build --release --bin qmdb_bench
	./runtime/target/release/qmdb_bench

# Print VM information
info:
	@echo "SVM-Subnet VM"
	@echo "============="
	@echo "VM ID:          $(VM_ID)"
	@echo "Plugins Path:   $(PLUGINS_PATH)"
	@echo "Build Dir:      $(BUILD_DIR)"
	@echo ""
	@echo "Endpoints:"
	@echo "  /            - Unified RPC (auto-detect Solana/Ethereum)"
	@echo "  /solana      - Solana JSON-RPC"
	@echo "  /evm         - Ethereum JSON-RPC"

help:
	@echo "SVM-Subnet Build System"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build everything (default)"
	@echo "  build        - Build Rust runtime and Go binaries"
	@echo "  build-runtime- Build Rust runtime only"
	@echo "  build-vm     - Build Go VM plugin only"
	@echo "  build-tools  - Build CLI tools"
	@echo "  genesis      - Generate default genesis.json"
	@echo "  install      - Install VM plugin to AvalancheGo"
	@echo "  test         - Run all tests"
	@echo "  fmt          - Format all code"
	@echo "  lint         - Lint all code"
	@echo "  clean        - Clean build artifacts"
	@echo "  dev          - Clean, build, install, generate genesis"
	@echo "  info         - Print VM information"
	@echo ""
	@echo "Environment Variables:"
	@echo "  AVALANCHEGO_PATH - Path to AvalancheGo (default: ~/.avalanchego)"
