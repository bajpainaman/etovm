# SVM Runtime

High-performance Solana Virtual Machine runtime for Avalanche subnet.

## Performance

- **Target:** 200k+ TPS sustained
- **Current:** 870k TPS (100k accounts), 450k TPS (5M accounts)
- **Architecture:** 6-stage parallel execution pipeline

## Quick Start

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run benchmarks
./target/release/turbo_bench

# Run tests
cargo test --release
```

## Architecture

```
Transaction → Verify → Schedule → Execute → Commit → Merkle
     ↓          ↓         ↓          ↓         ↓        ↓
   Input    Ed25519    DashSet    Parallel  Atomic   SIMD
           batch(8x)  lock-free   workers   commit  SHA256
```

Key optimizations:
- **Ed25519 batch verification** - 8x speedup
- **SIMD SHA256** - sha2 asm (SHA-NI/ARMv8)
- **Lock-free scheduling** - DashSet parallel conflict detection
- **Parallel merkle** - Work-stealing with rayon
- **Arena memory pools** - Zero allocation hot path

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [Benchmarks](docs/BENCHMARKS.md) - Performance measurements
- [Optimization](docs/OPTIMIZATION.md) - Roadmap to 1M+ TPS

## Project Structure

```
src/
├── hiperf/              # High-performance execution engine
│   ├── turbo_executor   # 6-stage pipeline
│   ├── batch_verifier   # Ed25519 batch verification
│   ├── fast_scheduler   # Lock-free transaction batching
│   ├── simd_hash        # SIMD SHA256 (sha2 asm)
│   ├── parallel_merkle  # Parallel tree computation
│   └── memory_pool      # Arena allocators
├── qmdb_state.rs        # QMDB state storage integration
├── executor.rs          # Standard executor
└── types.rs             # Core data types
```

## Dependencies

- **rayon** - Work-stealing parallelism
- **dashmap** - Lock-free concurrent maps
- **ed25519-dalek** - Batch signature verification
- **sha2** - SIMD SHA256 (asm feature)
- **qmdb** - Quick Merkle Database

## Hardware Acceleration

Automatically detects and uses:
- SHA-NI (Intel Skylake+, AMD Zen+)
- AVX2 (Intel Haswell+)
- ARMv8 crypto (Apple Silicon)

## License

MIT
