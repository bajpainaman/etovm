# SVM Runtime Architecture

## Overview

High-performance Solana Virtual Machine runtime targeting **200k+ TPS sustained** on Avalanche subnet. Uses a 6-stage parallel execution pipeline with lock-free data structures.

## Architecture Diagram

```
                   ┌─────────────────────────────────────────────┐
                   │            Transaction Ingestion            │
                   └─────────────────┬───────────────────────────┘
                                     │
                   ┌─────────────────▼───────────────────────────┐
                   │     Stage 1: Parallel Batch Verification    │
                   │     - Ed25519 batch verify (8x speedup)     │
                   │     - SHA256 SIMD hashing (sha2 asm)        │
                   └─────────────────┬───────────────────────────┘
                                     │ (lock-free channel)
                   ┌─────────────────▼───────────────────────────┐
                   │     Stage 2: Conflict Analysis & Schedule   │
                   │     - Parallel access set extraction        │
                   │     - Lock-free batch formation (DashSet)   │
                   └─────────────────┬───────────────────────────┘
                                     │ (lock-free channel)
   ┌──────────────────┬──────────────┼──────────────┬──────────────────┐
   │                  │              │              │                  │
   ▼                  ▼              ▼              ▼                  ▼
┌──────┐          ┌──────┐      ┌──────┐      ┌──────┐          ┌──────┐
│Worker│          │Worker│      │Worker│      │Worker│          │Worker│
│  1   │          │  2   │      │  3   │      │  4   │          │  N   │
└──┬───┘          └──┬───┘      └──┬───┘      └──┬───┘          └──┬───┘
   │                 │             │             │                 │
   └────────────────────────────┬──┴─────────────┴─────────────────┘
                                │ (aggregation)
                   ┌────────────▼────────────────────────────────┐
                   │     Stage 3: Parallel State Commit          │
                   │     - Lock-free changeset merge             │
                   │     - Parallel merkle tree computation      │
                   └─────────────────────────────────────────────┘
```

## Module Structure

```
src/hiperf/
├── mod.rs              # Module exports and coordination
├── turbo_executor.rs   # Main execution engine (6-stage pipeline)
├── batch_verifier.rs   # Ed25519 batch signature verification
├── fast_scheduler.rs   # Lock-free transaction batching
├── simd_hash.rs        # SIMD SHA256 (sha2 asm feature)
├── parallel_merkle.rs  # Parallel merkle tree computation
├── incremental_merkle.rs # O(k log n) incremental updates
├── memory_pool.rs      # Arena allocators for zero-alloc hot path
└── pipeline.rs         # Async pipelined execution
```

## Stage Details

### Stage 1: Signature Verification

**File:** `batch_verifier.rs`

- Ed25519 batch verification provides **8x speedup** over individual verification
- Falls back to individual verification on batch failure (identifies bad signatures)
- Uses BATCH_SIZE=64 for L1 cache efficiency
- Message hashing via SIMD SHA256

```rust
// Batch verify all signatures in parallel
let batch_results = ed25519_dalek::verify_batch(&messages, &signatures, &public_keys);
```

### Stage 2: Conflict Analysis & Scheduling

**Files:** `fast_scheduler.rs`, `access_set.rs`

- **Fast path:** Checks if all transactions are independent (no conflicts)
- **Slow path:** Greedy scheduling with conflict detection
- Uses DashSet for lock-free parallel conflict detection

```rust
// Fast path: Check for all-independent batch
if self.all_independent(&access_sets) {
    return vec![all_indices];  // Single batch, max parallelism
}
```

### Stage 3: Parallel Execution

**File:** `turbo_executor.rs`

- Batch account pre-loading (single lock acquisition vs N per transaction)
- SmallVec for stack-allocated account arrays
- Native execution bypass for system programs

### Stage 4: State Commit

**File:** `turbo_executor.rs`

- Changeset aggregation from parallel execution
- Sequential application (current bottleneck - see OPTIMIZATION.md)

### Stage 5: Merkle Root Computation

**Files:** `parallel_merkle.rs`, `incremental_merkle.rs`

- SIMD-accelerated SHA256 hashing
- Parallel tree construction with work-stealing (rayon)
- Incremental updates for O(k log n) complexity when <50% dirty

## Key Optimizations Implemented

| Optimization | Location | Speedup |
|--------------|----------|---------|
| Ed25519 batch verify | batch_verifier.rs | 8x |
| SIMD SHA256 (sha2 asm) | simd_hash.rs | 2-4x |
| Parallel merkle | parallel_merkle.rs | Linear scaling |
| Lock-free scheduling | fast_scheduler.rs | No contention |
| Arena memory pools | memory_pool.rs | Zero allocation |
| Batch account loading | turbo_executor.rs | 1 lock vs N |

## Data Flow

```
Transaction Batch
    ↓
[Stage 1] Parallel Batch Verification
    - Ed25519 batch verify (8x speedup)
    - Fall back to individual if batch fails
    - Output: Vec<bool> validity bitmap
    ↓
[Stage 2] Parallel Access Set Extraction
    - Extract reads/writes from message header
    - Output: Vec<AccessSet>
    ↓
[Stage 3] Fast Scheduling
    - Check all-independent fast path
    - If conflicts, greedy batching
    - Output: Vec<TransactionBatch>
    ↓
[Stage 4] Parallel Execution
    - Batch account pre-loading
    - Parallel execution in cached account state
    - Output: Vec<StateChangeSet>
    ↓
[Stage 5] State Commit
    - Sequential changeset application
    - Output: Updated state
    ↓
[Stage 6] Merkle Root Computation
    - Incremental updates for dirty paths
    - Falls back to parallel rebuild if >50% dirty
    - Output: [u8; 32] merkle root
```

## Performance Characteristics

| Stage | % Time | Scaling | Notes |
|-------|--------|---------|-------|
| Verification | 5-10% | O(n) parallel | Ed25519 curve math |
| Access Analysis | 2-5% | O(n) parallel | HashSet operations |
| Scheduling | 5-10% | O(n*b) greedy | Linear batch search |
| Execution | 50-60% | O(n) parallel | Lock contention |
| Commit | 10-15% | O(n) sequential | **BOTTLENECK** |
| Merkle | 10-20% | O(k log n) | Incremental updates |

## Dependencies

```toml
# Parallelism
rayon = "1.10"
dashmap = "6.0"
crossbeam-channel = "0.5"

# Crypto
ed25519-dalek = { version = "2.1", features = ["batch"] }
sha2 = { version = "0.10", features = ["asm"] }

# State Storage
qmdb = { git = "https://github.com/LayerZero-Labs/qmdb.git" }
```

## Build for Maximum Performance

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables:
- SHA-NI instructions (Intel/AMD SHA extensions)
- AVX2 vectorization
- ARMv8 crypto extensions (Apple Silicon)
