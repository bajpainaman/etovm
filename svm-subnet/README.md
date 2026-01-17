# SVM Subnet

High-performance Solana Virtual Machine running as an Avalanche L1 subnet.

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Peak TPS** | 11-14M | Turbo executor, delta mode |
| **Sustained TPS** | 1.14M | With merkle root computation |
| **Threads** | 96 | Auto-scales to available CPUs |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Avalanche Consensus                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                         Go VM Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   vm.go     │  │  mempool    │  │     rpc_handler.go      │  │
│  │ (ChainVM)   │  │             │  │  (Solana-compatible)    │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          │    ┌───────────┴──────────────────────┘
          │    │
┌─────────▼────▼──────────────────────────────────────────────────┐
│                    FFI Bindings (vm/ffi/)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  turbo   │  │   qmdb   │  │ sealevel │  │      hybrid      │ │
│  │  11M TPS │  │ parallel │  │ standard │  │    (SVM+EVM)     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │             │             │                 │
┌───────▼─────────────▼─────────────▼─────────────────▼───────────┐
│                     Rust Runtime (libsvm_runtime)               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   hiperf/ (High Performance)                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ││
│  │  │turbo_exec   │  │batch_verify │  │  parallel_merkle    │  ││
│  │  │6-stage pipe │  │Ed25519 8x   │  │  SIMD SHA256        │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   programs/ (Native)                        ││
│  │  system │ token │ associated_token │ stake │ vote │ bpf    ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   evm/ (Interop)                            ││
│  │  executor │ precompiles │ bridge │ state                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Build

```bash
# Build everything
make build

# Or manually:
cd runtime && RUSTFLAGS="-C target-cpu=native" cargo build --release
cd .. && go build -o build/svm ./cmd/svm
```

### Run Benchmarks

```bash
# Turbo executor benchmark (11M+ TPS)
./runtime/target/release/turbo_bench

# Full QMDB benchmark with merkle roots
./runtime/target/release/qmdb_bench
```

### Deploy to Avalanche

```bash
# Generate genesis
./build/genesis-gen --output genesis.json

# Copy VM binary to Avalanche plugins
cp build/svm ~/.avalanchego/plugins/<vmID>

# Configure and start node
avalanchego --track-subnets=<subnetID>
```

## Project Structure

```
svm-subnet/
├── cmd/                    # Go entrypoints
│   ├── svm/               # Main VM binary
│   └── genesis/           # Genesis generator
├── vm/                    # Avalanche VM implementation
│   ├── vm.go              # ChainVM interface
│   ├── block.go           # Block structure
│   ├── mempool.go         # Transaction pool
│   ├── rpc_handler.go     # Solana RPC compatibility
│   └── ffi/               # Rust FFI bindings
│       ├── turbo.go       # 11M TPS executor
│       ├── qmdb.go        # Parallel executor with merkle
│       ├── sealevel.go    # Standard executor
│       └── hybrid.go      # SVM + EVM
├── runtime/               # Rust SVM runtime
│   ├── src/
│   │   ├── hiperf/        # High-performance execution
│   │   ├── programs/      # Native programs
│   │   ├── evm/           # EVM interop
│   │   ├── sealevel/      # Parallel scheduling
│   │   └── types/         # Core types
│   └── docs/              # Detailed documentation
├── genesis/               # Genesis configuration
├── rpc/                   # RPC server (standalone)
├── tools/                 # Benchmarks and utilities
└── test/                  # Integration tests
```

## Executors

| Executor | TPS | Merkle | Use Case |
|----------|-----|--------|----------|
| **Turbo** | 11-14M | Delta mode | Maximum throughput |
| **QMDB** | 500k-1M | Full | Production with state proofs |
| **Sealevel** | 200k | Optional | Solana compatibility |
| **Hybrid** | 100k | Full | SVM + EVM interop |

## RPC Compatibility

Solana-compatible JSON-RPC endpoints:

- `sendTransaction` - Submit transactions
- `getSlot` - Current block height
- `getBalance` - Account balance
- `getAccountInfo` - Full account data
- `getBlockHeight` - Alias for getSlot
- `getRecentBlockhash` - For transaction signing

## Configuration

### VM Config (`config.json`)

```json
{
  "chainId": 43114,
  "mempoolSize": 100000,
  "verifySignatures": true,
  "useTurboMode": true
}
```

### Genesis (`genesis.json`)

```json
{
  "timestamp": 1768621163,
  "accounts": [
    {"pubkey": "11111111111111111111111111111111", "lamports": 1, "executable": true},
    ...
  ]
}
```

## Documentation

- [Runtime Architecture](runtime/docs/ARCHITECTURE.md) - Execution pipeline design
- [Benchmarks](runtime/docs/BENCHMARKS.md) - Performance measurements
- [Optimization](runtime/docs/OPTIMIZATION.md) - Roadmap to higher TPS

## Development

### Prerequisites

- Go 1.21+
- Rust 1.75+ (nightly for some features)
- Avalanche node (for deployment)

### Testing

```bash
# Rust tests
cd runtime && cargo test --release

# Go tests
go test ./...

# Integration tests
go test ./test/...
```

### Building for Production

```bash
# Optimize for target CPU
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release

# Build VM binary
CGO_ENABLED=1 go build -ldflags="-s -w" -o build/svm ./cmd/svm
```

## License

MIT
