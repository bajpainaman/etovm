# Commonware Integration Plan for ETO-VM

## Executive Summary

Commonware provides battle-tested blockchain primitives that can enhance eto-vm's performance and maintainability. This plan identifies **high-value integration opportunities** prioritized by impact.

---

## Priority 1: Immediate Value (1-2 weeks)

### 1.1 Replace Custom Parallel Code with `commonware-parallel`

**Current State:** eto-vm uses raw rayon with custom work-stealing patterns in:
- `parallel_merkle.rs` - Merkle tree computation
- `fast_scheduler.rs` - Transaction scheduling
- `turbo_executor.rs` - Pipeline stages

**Commonware Offers:** `Strategy` trait that abstracts Sequential vs Rayon execution with:
- `fold` / `fold_init` - Parallelized reduce operations
- `map_collect_vec` - Parallel map with result collection
- `join` - Fork-join parallelism

**Benefits:**
- Testable: Swap `Rayon` → `Sequential` for deterministic tests
- Cleaner: No more raw rayon boilerplate
- Flexible: Per-partition initialization for scratch buffers

**Integration Points:**
```rust
// Before (turbo_executor.rs)
rayon::scope(|s| {
    for batch in batches {
        s.spawn(|_| execute_batch(batch));
    }
});

// After
let strategy = Rayon::new(num_threads)?;
strategy.fold(
    batches,
    || ExecutionResult::default(),
    |acc, batch| acc.merge(execute_batch(batch)),
    |a, b| a.merge(b),
);
```

**Effort:** ~3 days

---

### 1.2 Evaluate Commonware's Ed25519 vs Current Implementation

**Current State:** `batch_verifier.rs` uses `ed25519-dalek` with batch verification (~8x speedup)

**Commonware Uses:** `ed25519-consensus` crate (stricter validation rules for consensus stability)

**Key Difference:**
- `ed25519-dalek`: Faster, less strict
- `ed25519-consensus`: Slower, consensus-safe (prevents certain signature malleability attacks)

**Action Items:**
1. Benchmark both implementations
2. Decide if consensus-safety matters for SVM (Solana uses dalek)
3. If switching: use `commonware_cryptography::ed25519::Batch` for batch verification

**Effort:** ~2 days benchmarking

---

## Priority 2: Medium-Term Value (2-4 weeks)

### 2.1 Adopt Commonware's MMR for Incremental Merkle

**Current State:** `incremental_merkle.rs` has custom O(k log n) merkle updates

**Commonware Offers:** Full MMR (Merkle Mountain Range) implementation with:
- Append-only design (perfect for transaction logs)
- Efficient range proofs
- Journaled storage backend
- Pruning support

**Why MMR > Binary Merkle:**
- O(log n) appends vs O(log n) but simpler
- Native range proofs (prove tx 100-200 in one proof)
- Better for light clients

**Integration:**
```rust
use commonware_storage::mmr::{journaled::Mmr, Proof};

// Replace parallel_merkle.rs with:
let mmr: Mmr<Sha256> = Mmr::init(storage, metadata).await?;
mmr.append(tx_hash).await?;
let proof = mmr.prove(location).await?;
```

**Effort:** ~1 week

---

### 2.2 Evaluate Commonware's QMDB vs LayerZero's QMDB

**Current State:** eto-vm uses LayerZero's QMDB for state storage

**Commonware's QMDB:**
- Append-only log of operations
- 4-state machine: Clean → Mutable → Merkleized → Durable
- Built-in pruning via "inactivity floor"
- MMR-based merkleization

**Key Architectural Difference:**
| Feature | LayerZero QMDB | Commonware QMDB |
|---------|----------------|-----------------|
| Design | Twig-based tree | Append-only log + MMR |
| I/O | io_uring | Async journal |
| Sharding | 16-way | Not built-in |
| Prefetch | Automatic | Manual |

**Recommendation:** Keep LayerZero QMDB for now (production-proven), but:
- Use Commonware's `mmr::verify` for proof verification
- Use Commonware's `storage::journal` for auxiliary logs

**Effort:** ~3 days evaluation

---

### 2.3 BLS12-381 Aggregated Signatures for Validator Sets

**Current State:** Ed25519 per-validator signatures

**Commonware Offers:** Full BLS12-381 with:
- `bls12381::primitives::ops::aggregate` - Aggregate N signatures into 1
- `bls12381::certificate::threshold` - t-of-n threshold signatures
- `bls12381::certificate::multisig` - Multi-signature certificates

**Use Case:** Avalanche subnet validator attestations
- Currently: N validators sign block → N signatures to verify
- With BLS: N validators sign → 1 aggregated signature

**Benefits:**
- ~96 bytes per block (vs N * 64 bytes with Ed25519)
- O(1) verification (vs O(n))
- Enables succinct light client proofs

**Integration Path:**
1. Add `commonware-cryptography` with `bls12381` feature
2. Create `AggregatedBlockCertificate` type
3. Validators produce BLS signatures
4. Aggregator combines into single certificate

**Effort:** ~2 weeks

---

## Priority 3: Future Consideration (1-2 months)

### 3.1 Simplex Consensus Engine

**Commonware's Simplex:** Byzantine-fault-tolerant consensus with:
- Pluggable signature schemes (Ed25519, BLS12-381, secp256r1)
- Built-in view change
- Elector abstraction for leader selection

**Current State:** eto-vm delegates to Avalanche Snowman consensus

**Consideration:** If ever building a standalone chain (not Avalanche subnet), Simplex could replace external consensus.

**Not recommended now:** Avalanche consensus is battle-tested and provides subnet benefits.

---

### 3.2 P2P Networking Layer

**Commonware's P2P:**
- Noise protocol handshakes
- Authenticated encrypted channels
- Peer identity verification

**Current State:** Uses Avalanche's gRPC networking

**Consideration:** Only useful if decoupling from Avalanche.

---

### 3.3 Broadcast Primitives

**Commonware's Broadcast:** Reliable broadcast over WAN

**Use Case:** Transaction propagation within subnet

**Current State:** Avalanche handles gossip

---

## Integration Checklist

### Phase 1 (Week 1-2)
- [ ] Add `commonware-parallel` to Cargo.toml
- [ ] Refactor `parallel_merkle.rs` to use `Strategy` trait
- [ ] Refactor `turbo_executor.rs` batch execution
- [ ] Benchmark Ed25519 implementations
- [ ] Write tests with `Sequential` strategy

### Phase 2 (Week 3-4)
- [ ] Evaluate MMR for transaction log merkleization
- [ ] Prototype `AggregatedBlockCertificate` with BLS
- [ ] Benchmark BLS aggregation overhead

### Phase 3 (Month 2)
- [ ] Full MMR integration if benchmarks favorable
- [ ] BLS validator signatures in production
- [ ] Light client proof generation

---

## Dependencies to Add

```toml
[dependencies]
# Core parallelism abstraction
commonware-parallel = "0.0.65"

# Cryptography (Ed25519 + BLS)
commonware-cryptography = { version = "0.0.65", features = ["bls12381"] }

# Storage primitives (MMR, journal)
commonware-storage = { version = "0.0.65", features = ["std"] }

# Runtime (if adopting their async runtime)
# commonware-runtime = "0.0.65"
```

---

## Risk Assessment

| Integration | Risk | Mitigation |
|-------------|------|------------|
| `commonware-parallel` | Low | Drop-in replacement, easy rollback |
| Ed25519 switch | Medium | Benchmark thoroughly, keep dalek as fallback |
| MMR adoption | Medium | Run parallel with current merkle for 1 week |
| BLS signatures | Medium | Feature-flag, gradual rollout |
| Full QMDB swap | High | Not recommended, keep LayerZero's |

---

## Next Steps

1. **Start with `commonware-parallel`** - lowest risk, highest code quality improvement
2. **Benchmark Ed25519** - 2 hours to know if it's worth switching
3. **Prototype BLS** - highest long-term value for validator efficiency
