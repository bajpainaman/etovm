# Optimization Roadmap

## Current State (January 2026)

**Peak Performance:** 870k TPS at 100k accounts, 450k TPS at 5M accounts

## Bottleneck Analysis

```
Stage Breakdown at Scale:

Execution:  50-60%  ← Scales well (parallel)
Merkle:     45-50%  ← Scales well (parallel)
Commit:     10-15%  ← SEQUENTIAL (the ceiling)
Verify:      5-10%  ← Scales well (batch)
Schedule:    2-5%   ← Scales well (lock-free)
```

**The commit stage is the killer.** Sequential changeset application creates a ceiling that no amount of parallel execution can break through.

## Path to 1M+ TPS

### Priority #1: Parallel Changeset Merge (Est. +15-25%)

**Current (sequential):**
```rust
// turbo_executor.rs line 250-253
let all_changesets = changesets.into_inner();
for changeset in all_changesets {
    self.state.add_tx_changes(changeset)?;  // sequential!
}
```

**Optimized (parallel merge + atomic commit):**
```rust
// Merge all changesets in parallel first
let merged = all_changesets
    .into_par_iter()
    .reduce(
        || StateChangeSet::new(),
        |mut a, b| { a.merge(&b); a }
    );

// Single atomic commit
self.state.apply_merged(merged)?;
```

**Why this works:**
- Changesets are independent (different accounts modified by different txs)
- Parallel merge is O(n/p) where p = cores
- Single commit eliminates lock contention
- No sequential bottleneck

**Estimated gain:** 15-25% - removes the sequential ceiling entirely

### Priority #2: Bitset AccessSets (Est. +5-10%)

**Current (HashSet):**
```rust
// access_set.rs
pub struct AccessSet {
    pub reads: HashSet<Pubkey>,
    pub writes: HashSet<Pubkey>,
}
```

**Optimized (Bitset for small sets):**
```rust
pub struct AccessSet {
    // Most txs touch <10 accounts, 256 bits = 4 u64s
    pub reads: SmallBitSet,
    pub writes: SmallBitSet,
}

// Conflict detection becomes bitwise AND
fn conflicts_with(&self, other: &AccessSet) -> bool {
    (self.writes.0 & other.writes.0) != 0 ||
    (self.writes.0 & other.reads.0) != 0 ||
    (self.reads.0 & other.writes.0) != 0
}
```

**Why this works:**
- 95% of transactions touch <10 accounts
- Bitwise operations are single CPU cycles
- No hash computation, no memory allocation
- Cache-friendly (32 bytes vs unbounded HashSet)

**Estimated gain:** 5-10% on scheduling phase, 2-3x faster conflict detection

### Priority #3: Pipelined Execution + Commit (Est. +10-15%)

**Current (sequential stages):**
```
Execute batch N → Wait → Commit batch N → Execute batch N+1
```

**Optimized (overlapped):**
```
Time →
Thread Pool 1: [Execute N  ][Execute N+1][Execute N+2]
Thread Pool 2:              [Commit N   ][Commit N+1 ]
```

**Implementation:**
```rust
// Split thread pool: 70% execute, 30% commit
let (exec_pool, commit_pool) = rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .build_scoped(|pool| {
        // 70/30 split
    });

// Pipeline with channels
let (tx, rx) = crossbeam_channel::bounded(2);

// Execute thread produces changesets
exec_pool.spawn(|| {
    let changeset = execute_batch(batch_n);
    tx.send(changeset).unwrap();
});

// Commit thread consumes in parallel
commit_pool.spawn(|| {
    while let Ok(changeset) = rx.recv() {
        state.apply_merged(changeset);
    }
});
```

**Why this works:**
- Execution and commit are independent
- Commit uses different state locks than read-heavy execution
- Double buffering hides latency
- Never waiting on sequential operations

**Estimated gain:** 10-15% by hiding commit latency

## Secondary Optimizations

### #4: Thread-Local Conflict Detection

Replace DashSet with thread-local HashSets + parallel merge:
```rust
// Current: DashSet with lock contention
writes.par_iter().for_each(|w| {
    if !global_writes.insert(*w) { /* conflict */ }
});

// Better: Thread-local + merge
let local_results: Vec<_> = writes
    .par_chunks(chunk_size)
    .map(|chunk| {
        let mut local = HashSet::new();
        for w in chunk { local.insert(*w); }
        local
    })
    .collect();

// Single merge at end
```

**Estimated gain:** ~1.2x on massive batches (10k+ txs)

### #5: Account Cache Pre-allocation

```rust
// Current: New HashMap per batch
let account_cache: HashMap<Pubkey, Account> = ...;

// Better: Thread-local reusable cache
thread_local! {
    static CACHE: RefCell<HashMap<Pubkey, Account>> =
        RefCell::new(HashMap::with_capacity(1000));
}
```

**Estimated gain:** 5-10% allocation overhead reduction

### #6: Batch Message Hashing

```rust
// Current: Hash one message at a time
for tx in transactions {
    let hash = sha256(&tx.message);
}

// Better: Batch hash all messages
let hashes = sha256_batch_parallel(&messages);
```

**Estimated gain:** 1.5-2x on verification phase for large batches

## Projected Performance

| Optimization | Cumulative TPS (1M accounts) |
|--------------|------------------------------|
| Current | ~620k |
| +Parallel Merge | ~775k (+25%) |
| +Bitset Access | ~850k (+10%) |
| +Pipeline Overlap | ~950k (+12%) |
| +Thread-local | ~1.0M (+5%) |

## Implementation Order

1. **Parallel Changeset Merge** - Biggest single win, low risk
2. **Bitset AccessSets** - Clean abstraction, easy to test
3. **Pipeline Overlap** - More complex, requires careful synchronization
4. **Thread-local optimizations** - Marginal gains, polish phase

## Testing Strategy

Each optimization should:
1. Maintain identical merkle roots (backwards compatible)
2. Pass all existing tests
3. Show measurable improvement in benchmarks
4. Not regress any batch size

```bash
# Regression test
for size in 100000 500000 1000000 2000000 5000000; do
    ./target/release/turbo_bench $size
done

# Verify merkle roots match baseline
diff <(./target/release/turbo_bench 2>&1 | grep "Merkle Root") baseline_roots.txt
```

## Risk Assessment

| Optimization | Risk | Mitigation |
|--------------|------|------------|
| Parallel Merge | Low | Changesets are independent by design |
| Bitset Access | Low | Fallback to HashSet for >256 accounts |
| Pipeline Overlap | Medium | Careful lock ordering, extensive testing |
| Thread-local | Low | Standard pattern, well-understood |

## Success Criteria

- **Target:** 1M+ TPS at 1M accounts
- **Constraint:** Identical merkle roots (consensus compatible)
- **Constraint:** No test regressions
- **Constraint:** Stable performance (variance <10%)
