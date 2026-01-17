# SVM Subnet - Project Guide

## What This Is

A high-performance Solana Virtual Machine (SVM) running as an Avalanche L1 subnet. It achieves 11M+ TPS through a custom 6-stage parallel execution pipeline.

## Key Concepts

### Avalanche Integration
- **VM Interface**: Go code in `vm/` implements Avalanche's `ChainVM` interface
- **Consensus**: Avalanche handles consensus; we handle execution
- **Blocks**: Built when `WaitForEvent` signals pending transactions

### Execution Pipeline (Turbo Mode)
```
Verify → Analyze → Schedule → Execute → Commit → Merkle
  │         │          │         │         │        │
Ed25519   Extract    Batch    Parallel   Delta   SIMD
 batch    accounts   conflict  workers   mode   SHA256
```

### FFI Architecture
Go VM calls Rust runtime via CGO:
```
Go (vm.go) → FFI (vm/ffi/turbo.go) → Rust (runtime/src/lib.rs)
```

## Code Locations

### When Working On...

| Task | Primary Files |
|------|---------------|
| Block building | `vm/vm.go` (`BuildBlockWithContext`) |
| Transaction handling | `vm/tx.go`, `vm/mempool.go` |
| RPC endpoints | `vm/rpc_handler.go` |
| Execution engine | `runtime/src/hiperf/turbo_executor.rs` |
| Signature verification | `runtime/src/hiperf/batch_verifier.rs` |
| Merkle computation | `runtime/src/hiperf/parallel_merkle.rs` |
| Native programs | `runtime/src/programs/` |
| EVM interop | `runtime/src/evm/` |
| FFI bindings | `vm/ffi/*.go`, `runtime/src/lib.rs` |

### Critical Paths

**Transaction Flow:**
1. `rpc_handler.go:handleSendTransaction` - Receives RPC
2. `vm.go:SubmitTransaction` - Adds to mempool, signals pending
3. `vm.go:BuildBlockWithContext` - Builds block with transactions
4. `turbo.go:ExecuteBlockDelta` - FFI to Rust executor
5. `turbo_executor.rs:execute_block_delta` - Parallel execution

**Block Verification:**
1. `vm.go:ParseBlock` - Deserialize block
2. `block.go:Verify` - Verify block integrity

## Common Tasks

### Add New RPC Endpoint

```go
// vm/rpc_handler.go
case "newMethod":
    return h.handleNewMethod(params)

func (h *RPCHandler) handleNewMethod(params []interface{}) (interface{}, error) {
    // Implementation
}
```

### Add New Native Program

```rust
// runtime/src/programs/mod.rs
pub mod new_program;

// runtime/src/programs/new_program.rs
pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    data: &[u8],
) -> Result<(), ProgramError> {
    // Implementation
}
```

### Add FFI Function

```rust
// runtime/src/lib.rs
#[no_mangle]
pub extern "C" fn new_ffi_function(...) -> i32 {
    // Implementation
}
```

```go
// vm/ffi/turbo.go (or appropriate file)
/*
extern int new_ffi_function(...);
*/
import "C"

func NewFunction(...) error {
    result := C.new_ffi_function(...)
    // Handle result
}
```

## Build Commands

```bash
# Full build
make build

# Rust only (for benchmarks)
cd runtime && RUSTFLAGS="-C target-cpu=native" cargo build --release

# Go only (for VM)
go build -o build/svm ./cmd/svm

# Run turbo benchmark
./runtime/target/release/turbo_bench

# Run tests
cargo test --release    # Rust
go test ./...           # Go
```

## Architecture Decisions

### Why Multiple Executors?

| Executor | Trade-off |
|----------|-----------|
| Turbo | Max TPS, delta merkle (less security) |
| QMDB | Full merkle proofs, lower TPS |
| Sealevel | Solana-compatible scheduling |
| Hybrid | SVM+EVM, lowest TPS |

### Why Delta Mode?

Full merkle recomputation is expensive. Delta mode:
- Only updates changed accounts
- Computes incremental merkle
- 10x+ faster for typical workloads

### Why FFI vs Pure Go?

Rust provides:
- SIMD intrinsics (SHA256, signature verification)
- Zero-cost abstractions
- Memory safety with performance
- Existing crypto libraries

## Performance Tuning

### CPU Optimization
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Thread Tuning
```rust
// runtime/src/hiperf/turbo_executor.rs
let num_threads = if num_threads == 0 {
    num_cpus::get()  // Auto-detect
} else {
    num_threads
};
```

### Memory
- Arena allocators in `memory_pool.rs`
- Pre-allocated buffers for hot paths
- SmallVec for stack allocation

## Debugging

### Enable Logging
```rust
// Add to Cargo.toml
log = "0.4"
env_logger = "0.10"

// Set environment
RUST_LOG=debug ./target/release/turbo_bench
```

### FFI Debugging
```go
// vm/ffi/turbo.go - add prints before/after C calls
fmt.Printf("Calling turbo_execute_block with %d txs\n", len(txs))
result := C.turbo_execute_block_delta(...)
fmt.Printf("Result: %d successful\n", result.successful)
```

## Known Issues

1. **Avalanche Bootstrap**: Primary network (P/X/C chains) must sync before subnet works
2. **WaitForEvent**: Uses notification channel - ensure `pendingTxs` channel is signaled
3. **Memory**: Large transaction batches may require tuning arena sizes

## Testing Checklist

Before committing:
- [ ] `cargo test --release` passes
- [ ] `go test ./...` passes
- [ ] `turbo_bench` runs without errors
- [ ] Build succeeds: `make build`
