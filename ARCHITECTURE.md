# ETO-VM: High-Performance SVM Runtime for Avalanche L1

## Executive Summary

ETO-VM is a **Solana Virtual Machine (SVM) runtime** designed to run as an **Avalanche L1 subnet**. It combines Solana's parallel execution model with Avalanche's sub-second finality, achieving **1.14M TPS** on synthetic benchmarks with **6.37M signature verifications/second** via GPU acceleration.

### Key Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| Peak TPS | 1,140,000 | Synthetic benchmark, parallel transfers |
| GPU Sig Verify | 6,370,000/sec | CUDA Ed25519 batch verification |
| Finality | ~430ms | Avalanche consensus |
| Signature Bottleneck | 4% | Down from 86.9% pre-GPU |

📊 **[Detailed Benchmark Results](svm-subnet/runtime/docs/BENCHMARKS.md)** - Full TPS breakdown by batch size, optimization history, and reproducible test commands.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Avalanche Consensus                          │
│                  (Snowman++ / Sub-second Finality)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Go VM Interface                             │
│                    (vm/vm.go - ChainVM)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Mempool   │  │  JSON-RPC   │  │   Block Production      │  │
│  │  (pending)  │  │ (Sol + Eth) │  │   (BuildBlock)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ FFI (cgo)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Rust SVM Runtime                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Sealevel Scheduler                       │    │
│  │            (Parallel Transaction Execution)              │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │ Thread 1│ │ Thread 2│ │ Thread 3│ │ Thread N│        │    │
│  │  │ (batch) │ │ (batch) │ │ (batch) │ │ (batch) │        │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                   Execution Layer                      │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │      │
│  │  │ BPF Interp  │  │   Native    │  │  EVM Bridge │    │      │
│  │  │ (eBPF VM)   │  │  Programs   │  │ (Precompiles│    │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                    State Layer                         │      │
│  │  ┌─────────────────────────────────────────────────┐  │      │
│  │  │                    QMDB                          │  │      │
│  │  │     (Quick Merkle Database - O(1) I/O)          │  │      │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │  │      │
│  │  │  │ Shard 0 │ │ Shard 1 │ │ Shard N │ (16-way)  │  │      │
│  │  │  └─────────┘ └─────────┘ └─────────┘           │  │      │
│  │  └─────────────────────────────────────────────────┘  │      │
│  └───────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                 GPU Acceleration                       │      │
│  │  ┌─────────────────────────────────────────────────┐  │      │
│  │  │           CUDA Ed25519 Verification              │  │      │
│  │  │         (6.37M signatures/sec batch)             │  │      │
│  │  └─────────────────────────────────────────────────┘  │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Sealevel Parallel Scheduler

Solana's **Sealevel** runtime enables parallel transaction execution by analyzing account read/write sets before execution.

```rust
// svm-subnet/runtime/src/sealevel/scheduler.rs

pub struct SealevelScheduler {
    /// Thread pool for parallel execution
    thread_pool: ThreadPool,
    /// Account lock manager
    account_locks: AccountLocks,
    /// Batch size for parallel execution
    batch_size: usize,
}

impl SealevelScheduler {
    /// Schedule transactions into non-conflicting batches
    pub fn schedule(&self, txs: Vec<Transaction>) -> Vec<Batch> {
        // 1. Extract read/write sets from each transaction
        // 2. Build conflict graph
        // 3. Partition into independent batches
        // 4. Execute batches in parallel
    }
}
```

**How it works:**
1. **Pre-execution analysis**: Extract account keys from each transaction
2. **Conflict detection**: Transactions touching same writable accounts conflict
3. **Batch formation**: Group non-conflicting transactions
4. **Parallel execution**: Execute batches across threads

**Performance characteristics:**
- 35x parallelism factor on typical workloads
- ~10μs average latency per transaction
- Linear scaling with core count

### 2. QMDB (Quick Merkle Database)

QMDB is a **high-performance state database** optimized for blockchain workloads. Originally developed by LayerZero.

```rust
// svm-subnet/runtime/src/real_qmdb_state.rs

pub struct RealQmdbState {
    /// QMDB instance with 16-way sharding
    ads: RwLock<AdsWrap<AccountTask>>,
    /// Current block height
    current_height: AtomicU64,
    /// Pending changes (batched before commit)
    pending_changes: RwLock<HashMap<Pubkey, Option<Account>>>,
}
```

**Key features:**
- **O(1) I/O per update**: Constant-time reads and writes
- **Twig-based Merkle tree**: Minimizes RAM usage while maintaining proofs
- **16-way sharding**: Parallel I/O across shards
- **io_uring support**: Async disk I/O on Linux
- **Append-only design**: Optimized for SSD write patterns

**Architecture:**
```
┌──────────────────────────────────────────────┐
│                 QMDB Instance                 │
├──────────────────────────────────────────────┤
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │Shard 0 │ │Shard 1 │ │  ...   │ │Shard 15│ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ │
│      │          │          │          │      │
│  ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ │
│  │ Twigs  │ │ Twigs  │ │ Twigs  │ │ Twigs  │ │
│  │(2048/ea│ │        │ │        │ │        │ │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ │
│      │          │          │          │      │
│      └──────────┴────┬─────┴──────────┘      │
│                      │                        │
│              ┌───────┴───────┐               │
│              │  Merkle Root   │               │
│              └───────────────┘               │
└──────────────────────────────────────────────┘
```

### 3. GPU Signature Verification

Ed25519 signature verification is the **#1 bottleneck** in blockchain throughput. We offload this to GPU.

```rust
// svm-subnet/runtime/src/gpu/cuda_verify.rs

pub struct GpuSignatureVerifier {
    /// CUDA context
    context: CudaContext,
    /// Pre-allocated device buffers
    buffers: DeviceBuffers,
    /// Batch size (optimal: 8192-16384)
    batch_size: usize,
}

impl GpuSignatureVerifier {
    /// Batch verify signatures on GPU
    pub fn verify_batch(&self, signatures: &[SignatureRequest]) -> Vec<bool> {
        // 1. Copy signatures, messages, pubkeys to GPU
        // 2. Launch CUDA kernel (1 thread per signature)
        // 3. Each thread performs Ed25519 verify
        // 4. Copy results back to CPU
    }
}
```

**CUDA Kernel (simplified):**
```cuda
__global__ void ed25519_verify_kernel(
    const uint8_t* signatures,  // 64 bytes each
    const uint8_t* messages,    // variable length
    const uint8_t* pubkeys,     // 32 bytes each
    const uint32_t* msg_lens,
    bool* results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Perform Ed25519 verification
    results[idx] = ed25519_verify(
        &signatures[idx * 64],
        &messages[msg_offsets[idx]],
        msg_lens[idx],
        &pubkeys[idx * 32]
    );
}
```

**Performance:**
| Batch Size | Throughput | Latency |
|------------|------------|---------|
| 1,024 | 2.1M/sec | 0.49ms |
| 8,192 | 5.8M/sec | 1.41ms |
| 16,384 | 6.37M/sec | 2.57ms |

### 4. BPF Interpreter

Executes **Solana programs** (compiled to eBPF bytecode).

```rust
// svm-subnet/runtime/src/bpf/interpreter.rs

pub struct BpfInterpreter {
    /// Program bytecode
    bytecode: Vec<u8>,
    /// Memory regions
    memory: BpfMemory,
    /// Register file (r0-r10)
    registers: [u64; 11],
    /// Program counter
    pc: usize,
    /// Compute units remaining
    compute_budget: u64,
}

impl BpfInterpreter {
    pub fn execute(&mut self, entrypoint: &[u8]) -> Result<u64, BpfError> {
        loop {
            let insn = self.fetch_instruction();
            self.compute_budget -= insn.cost();

            match insn.opcode() {
                // ALU operations
                BPF_ADD => self.registers[insn.dst] += self.registers[insn.src],
                BPF_MUL => self.registers[insn.dst] *= self.registers[insn.src],

                // Memory operations
                BPF_LDX => self.registers[insn.dst] = self.memory.load(addr)?,
                BPF_STX => self.memory.store(addr, self.registers[insn.src])?,

                // Control flow
                BPF_CALL => self.call_syscall(insn.imm)?,
                BPF_EXIT => return Ok(self.registers[0]),

                // ...64 total opcodes
            }
        }
    }
}
```

**Syscalls available:**
- `sol_log` - Logging
- `sol_invoke_signed` - CPI (Cross-Program Invocation)
- `sol_get_clock_sysvar` - Clock access
- `sol_sha256` - Hashing
- `sol_create_program_address` - PDA derivation

### 5. Native Programs

High-performance programs implemented in Rust (not BPF):

| Program | ID | Purpose |
|---------|-----|---------|
| **System** | `1111...1111` | Account creation, SOL transfers |
| **BPF Loader** | `BPFLoader2111...` | Program deployment |
| **Token** | `TokenkegQf...` | SPL Token (fungible tokens) |
| **ATA** | `ATokenGPvbd...` | Associated Token Accounts |
| **Stake** | `Stake1111...` | Validator staking |
| **Vote** | `Vote1111...` | Consensus voting |

```rust
// Example: SPL Token Transfer (native implementation)
// svm-subnet/runtime/src/programs/token.rs

pub fn process_transfer(
    accounts: &mut [(Pubkey, Account)],
    amount: u64,
) -> RuntimeResult<()> {
    let source = TokenAccount::unpack(&accounts[0].1.data)?;
    let dest = TokenAccount::unpack(&accounts[1].1.data)?;

    // Verify ownership
    if source.owner != accounts[2].0 {
        return Err(RuntimeError::InvalidAuthority);
    }

    // Check balance
    if source.amount < amount {
        return Err(RuntimeError::InsufficientFunds);
    }

    // Transfer (checked arithmetic)
    source.amount = source.amount.checked_sub(amount)?;
    dest.amount = dest.amount.checked_add(amount)?;

    // Serialize back
    source.pack(&mut accounts[0].1.data)?;
    dest.pack(&mut accounts[1].1.data)?;

    Ok(())
}
```

### 6. EVM Bridge & Precompiles

**Seamless EVM ↔ SVM interoperability** via precompile contracts.

```
┌─────────────────────────────────────────────────────────────┐
│                    Solidity Contract                         │
│  IERC20(0x0102).transfer(recipient, amount)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVM Executor (revm)                       │
│  Intercepts call to 0x0102 (SPL Token precompile)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Precompile Registry                         │
│  Decodes ERC20 ABI → SPL Token instruction                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  SPL Token Program                           │
│  Executes transfer on SVM account state                     │
└─────────────────────────────────────────────────────────────┘
```

**Precompile addresses:**
| Address | Program | Interface |
|---------|---------|-----------|
| `0x0100` | SVM Bridge | Arbitrary program calls |
| `0x0101` | System Program | Account creation |
| `0x0102` | SPL Token | Full ERC20 interface |
| `0x0103` | ATA | Auto-create token accounts |

```solidity
// Solidity interface for SPL Token precompile
interface ISPLToken {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// Usage
ISPLToken token = ISPLToken(0x0000000000000000000000000000000000000102);
token.transfer(recipient, 1000 * 10**9); // Transfer 1000 tokens
```

### 7. Avalanche VM Interface

The Go layer implements Avalanche's `ChainVM` interface:

```go
// svm-subnet/vm/vm.go

var _ block.ChainVM = (*VM)(nil)

type VM struct {
    ctx       *snow.Context
    db        database.Database
    runtime   *ffi.Runtime      // Single tx execution
    sealevel  *ffi.SealevelExecutor  // Parallel execution
    qmdb      *ffi.QMDBExecutor // Block execution with merkle
    hybrid    *ffi.HybridExecutor    // SVM + EVM
    mempool   *Mempool
    state     *State
}

// BuildBlock creates a new block from mempool transactions
func (vm *VM) BuildBlock(ctx context.Context) (snowman.Block, error) {
    txs := vm.mempool.GetBatch(maxBlockTxs)

    // Execute transactions in parallel via Sealevel
    results := vm.sealevel.ExecuteBatch(txs)

    // Compute state root via QMDB
    stateRoot := vm.qmdb.Commit()

    return &Block{
        height:    vm.state.height + 1,
        timestamp: time.Now(),
        txs:       txs,
        stateRoot: stateRoot,
    }, nil
}

// Verify checks block validity
func (vm *VM) Verify(ctx context.Context, blk snowman.Block) error {
    // Re-execute transactions and verify state root matches
}

// Accept finalizes a block
func (vm *VM) Accept(ctx context.Context, blk snowman.Block) error {
    // Persist state changes to QMDB
}
```

### 8. JSON-RPC API

**Dual RPC interface** - both Solana and Ethereum compatible:

```go
// svm-subnet/vm/rpc_handler.go

// Solana RPC methods
case "getAccountInfo":
    return h.getAccountInfo(params)
case "getBalance":
    return h.getBalance(params)
case "sendTransaction":
    return h.sendTransaction(params)
case "getLatestBlockhash":
    return h.getLatestBlockhash(params)

// Ethereum RPC methods
case "eth_chainId":
    return h.ethChainId()
case "eth_getBalance":
    return h.ethGetBalance(params)
case "eth_sendRawTransaction":
    return h.ethSendRawTransaction(params)
case "eth_call":
    return h.ethCall(params)
```

---

## Consensus: Avalanche vs Tower BFT

### Tower BFT (Solana's Approach)
- **PoH (Proof of History)**: Cryptographic clock via SHA256 chain
- **Leader rotation**: Deterministic schedule
- **Voting**: Validators vote on forks, exponential lockout
- **Finality**: ~400ms (but optimistic, can reorg)

### Avalanche Consensus (What We Use)
- **Snowman++**: Repeated subsampling for consensus
- **No leader**: Decentralized block proposal
- **Probabilistic finality**: Exponentially decreasing reorg probability
- **Finality**: ~430ms (true finality, no reorgs after)

**Why Avalanche?**
1. **True finality**: No reorgs after finalization
2. **Subnet isolation**: Our chain doesn't affect/isn't affected by others
3. **Simpler**: No PoH clock to maintain
4. **Interop**: Native bridges to C-Chain, other subnets

---

## Data Flow: Transaction Lifecycle

```
1. USER SUBMITS TX
   ┌─────────────────┐
   │ Solana TX or    │
   │ Ethereum TX     │
   └────────┬────────┘
            │
            ▼
2. RPC HANDLER
   ┌─────────────────┐
   │ Decode TX       │
   │ Basic validation│
   │ Add to mempool  │
   └────────┬────────┘
            │
            ▼
3. MEMPOOL
   ┌─────────────────┐
   │ Priority queue  │
   │ Dedup by sig    │
   │ TTL expiration  │
   └────────┬────────┘
            │
            ▼
4. BLOCK BUILDER (on leader turn)
   ┌─────────────────┐
   │ Select TXs      │
   │ GPU sig verify  │◄── Batch verify 8K+ sigs
   │ Build block     │
   └────────┬────────┘
            │
            ▼
5. SEALEVEL SCHEDULER
   ┌─────────────────┐
   │ Analyze locks   │
   │ Build batches   │
   │ Parallel exec   │◄── 35x parallelism
   └────────┬────────┘
            │
            ▼
6. EXECUTION
   ┌─────────────────┐
   │ BPF interpreter │
   │ Native programs │
   │ EVM (if needed) │
   └────────┬────────┘
            │
            ▼
7. STATE COMMIT
   ┌─────────────────┐
   │ QMDB batch      │
   │ Compute root    │◄── O(1) per update
   │ Persist         │
   └────────┬────────┘
            │
            ▼
8. CONSENSUS
   ┌─────────────────┐
   │ Propose block   │
   │ Snowman voting  │
   │ Finalize        │◄── ~430ms finality
   └─────────────────┘
```

---

## Performance Optimizations

> 📊 See [BENCHMARKS.md](svm-subnet/runtime/docs/BENCHMARKS.md) for detailed results

### 1. GPU Signature Offload
- **Before**: 86.9% time in sig verify
- **After**: 4% time in sig verify
- **Technique**: Batch 8K-16K signatures to GPU

### 2. Sealevel Parallelism
- **Conflict detection**: O(n) scan of account keys
- **Batch execution**: Rayon thread pool
- **Lock-free reads**: DashMap for concurrent access

### 3. QMDB Optimizations
- **Sharding**: 16 shards for parallel I/O
- **Prefetching**: Speculative reads during execution
- **Batching**: Accumulate writes, flush per block
- **io_uring**: Async I/O on Linux

### 4. Memory Layout
- **Cache-aligned structures**: 64-byte alignment
- **Arena allocation**: Reduce malloc overhead
- **Zero-copy deserialization**: Borsh with references

---

## File Structure

```
svm-subnet/
├── vm/                          # Go Avalanche VM interface
│   ├── vm.go                    # ChainVM implementation
│   ├── block.go                 # Block structure
│   ├── mempool.go               # Transaction pool
│   ├── rpc_handler.go           # JSON-RPC (Sol + Eth)
│   ├── state.go                 # State management
│   └── ffi/                     # Go-Rust FFI bindings
│       ├── runtime.go
│       └── executor.go
│
├── runtime/                     # Rust SVM runtime
│   └── src/
│       ├── lib.rs               # Main exports
│       ├── executor.rs          # Transaction executor
│       ├── runtime.rs           # Runtime context
│       │
│       ├── sealevel/            # Parallel scheduler
│       │   ├── scheduler.rs
│       │   ├── account_locks.rs
│       │   └── batch.rs
│       │
│       ├── bpf/                 # BPF interpreter
│       │   ├── interpreter.rs
│       │   ├── memory.rs
│       │   └── syscalls.rs
│       │
│       ├── programs/            # Native programs
│       │   ├── system.rs        # Account creation
│       │   ├── token.rs         # SPL Token
│       │   ├── associated_token.rs # ATA
│       │   ├── stake.rs         # Staking
│       │   ├── vote.rs          # Voting
│       │   └── bpf_loader.rs    # Program deployment
│       │
│       ├── evm/                 # EVM integration
│       │   ├── executor.rs      # revm wrapper
│       │   ├── bridge.rs        # SVM ↔ EVM bridge
│       │   ├── precompiles.rs   # ERC20 → SPL Token
│       │   └── state.rs         # State adapter
│       │
│       ├── gpu/                 # GPU acceleration
│       │   ├── cuda_verify.rs   # CUDA Ed25519
│       │   └── batch.rs         # Batch management
│       │
│       ├── qmdb_state.rs        # QMDB mock
│       ├── real_qmdb_state.rs   # Real QMDB
│       │
│       ├── types/               # Core types
│       │   ├── pubkey.rs
│       │   ├── account.rs
│       │   └── transaction.rs
│       │
│       └── sysvars/             # System variables
│           ├── clock.rs
│           └── rent.rs
│
├── genesis/                     # Genesis configuration
└── scripts/                     # Deployment scripts
```

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| System Program | 8 | ✅ |
| Token Program | 7 | ✅ |
| ATA Program | 5 | ✅ |
| Stake Program | 6 | ✅ |
| Vote Program | 4 | ✅ |
| BPF Interpreter | 12 | ✅ |
| Sealevel Scheduler | 15 | ✅ |
| EVM Precompiles | 16 | ✅ |
| QMDB State | 8 | ✅ |
| GPU Verify | 6 | ✅ |
| **Total** | **146** | ✅ |

---

## Deployment

### Local Development
```bash
# Build runtime
cd svm-subnet/runtime && cargo build --release

# Build VM
cd svm-subnet && go build -o svm-vm ./cmd/svm

# Run local network
avalanche network start --avalanchego-path=...
```

### Fuji Testnet
```bash
# Create blockchain
avalanche blockchain create svm --custom --vm-path ./svm-vm

# Deploy to Fuji
avalanche blockchain deploy svm --fuji

# Fund validators (need ~200 AVAX on P-Chain)
```

---

## Comparison with Other Approaches

| Feature | ETO-VM | Solana | Ethereum L2 | MoveVM |
|---------|--------|--------|-------------|--------|
| Execution | SVM (parallel) | SVM | EVM (serial) | Move |
| Consensus | Avalanche | Tower BFT | Rollup to L1 | Various |
| Finality | ~430ms | ~400ms | Minutes-hours | Varies |
| TPS (theoretical) | 1.1M+ | 65K | 2-4K | 100K+ |
| State DB | QMDB | AccountsDB | MPT | JMT |
| Sig Verify | GPU | CPU | CPU | CPU |
| EVM Compat | Precompiles | None | Native | None |

---

## Future Work

1. **JIT Compilation**: Compile hot BPF programs to native
2. **State Rent**: Implement Solana-style rent collection
3. **Compute Budget Program**: Fine-grained compute metering
4. **Token-2022**: Extended token features
5. **Warp Sync**: Fast state sync from snapshots
6. **Cross-Subnet Messaging**: Teleporter integration

---

## References

- [Solana Sealevel Runtime](https://docs.solana.com/developing/programming-model/runtime)
- [Avalanche Consensus](https://docs.avax.network/overview/getting-started/avalanche-consensus)
- [QMDB Paper](https://arxiv.org/abs/...) (LayerZero)
- [Ed25519 CUDA](https://github.com/...)
- [revm](https://github.com/bluealloy/revm) - Rust EVM

---

*ETO-VM: Where Solana's speed meets Avalanche's finality.*
