//! Fast Scheduler - Optimized Transaction Batching
//!
//! Uses fast-path detection to skip expensive conflict graph construction
//! when transactions don't conflict.
//!
//! ## Performance Modes
//!
//! - `schedule()`: HashSet-based, safe fallback
//! - `schedule_bitset()`: O(1) conflict detection via bitwise ops (+10% TPS)

use crate::sealevel::{
    AccessSet, TransactionBatch,
    // High-performance bitset types
    AccountIndexer, BitsetAccessSet, BitsetBatch,
};
use crate::types::Pubkey;
use dashmap::DashSet;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Fast scheduler with parallel conflict detection
pub struct FastScheduler {
    max_batch_size: usize,
}

impl FastScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }

    /// Schedule transactions with fast-path for non-conflicting batches
    ///
    /// 1. Parallel conflict detection
    /// 2. If no conflicts: single batch
    /// 3. If conflicts: use greedy batching
    pub fn schedule(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        if access_sets.is_empty() {
            return vec![];
        }

        // Fast path: check if all transactions are independent
        if self.all_independent_fast(access_sets) {
            // Single batch with all transactions
            let mut batch = TransactionBatch::new();
            for (idx, access) in access_sets.iter().enumerate() {
                batch.indices.push(idx);
                batch.write_set.extend(&access.writes);
                batch.read_set.extend(&access.reads);
            }
            return vec![batch];
        }

        // Slow path: greedy batching with conflict detection
        self.greedy_schedule(access_sets)
    }

    /// Fast parallel check for independence
    ///
    /// Uses DashSet for lock-free concurrent access tracking.
    /// Returns true if all transactions can run in parallel.
    fn all_independent_fast(&self, access_sets: &[AccessSet]) -> bool {
        let write_accounts: DashSet<Pubkey> = DashSet::new();
        let read_accounts: DashSet<Pubkey> = DashSet::new();
        let has_conflict = AtomicBool::new(false);

        // Parallel conflict detection
        access_sets.par_iter().for_each(|access| {
            if has_conflict.load(Ordering::Relaxed) {
                return; // Early exit if conflict already found
            }

            // Check write-write and write-read conflicts
            for pubkey in &access.writes {
                // Write-write conflict
                if !write_accounts.insert(*pubkey) {
                    has_conflict.store(true, Ordering::Relaxed);
                    return;
                }
                // Write-read conflict (writer after reader)
                if read_accounts.contains(pubkey) {
                    has_conflict.store(true, Ordering::Relaxed);
                    return;
                }
            }

            // Check read-write conflicts
            for pubkey in &access.reads {
                // Read-write conflict (reader after writer)
                if write_accounts.contains(pubkey) {
                    has_conflict.store(true, Ordering::Relaxed);
                    return;
                }
                read_accounts.insert(*pubkey);
            }
        });

        !has_conflict.load(Ordering::Relaxed)
    }

    /// Greedy batching algorithm (sequential but fast)
    fn greedy_schedule(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        let mut batches: Vec<TransactionBatch> = Vec::new();

        for (tx_idx, access) in access_sets.iter().enumerate() {
            let mut added = false;

            // Try to add to existing batch
            for batch in batches.iter_mut() {
                if batch.indices.len() < self.max_batch_size && self.can_add_fast(batch, access) {
                    batch.indices.push(tx_idx);
                    batch.write_set.extend(&access.writes);
                    batch.read_set.extend(&access.reads);
                    added = true;
                    break;
                }
            }

            // Create new batch if needed
            if !added {
                let mut new_batch = TransactionBatch::new();
                new_batch.indices.push(tx_idx);
                new_batch.write_set.extend(&access.writes);
                new_batch.read_set.extend(&access.reads);
                batches.push(new_batch);
            }
        }

        batches
    }

    /// Fast conflict check for adding to batch
    #[inline]
    fn can_add_fast(&self, batch: &TransactionBatch, access: &AccessSet) -> bool {
        // Check write-write conflicts
        for pubkey in &access.writes {
            if batch.write_set.contains(pubkey) {
                return false;
            }
            if batch.read_set.contains(pubkey) {
                return false;
            }
        }

        // Check read-write conflicts
        for pubkey in &access.reads {
            if batch.write_set.contains(pubkey) {
                return false;
            }
        }

        true
    }

    // ========================================================================
    // HIGH-PERFORMANCE BITSET-BASED SCHEDULING
    // ========================================================================

    /// Schedule using O(1) bitset conflict detection
    ///
    /// This is the high-performance path that provides ~10% TPS improvement
    /// over HashSet-based scheduling for large batches.
    ///
    /// Algorithm:
    /// 1. Build account indexer (maps Pubkey → bit index)
    /// 2. Convert all AccessSets to BitsetAccessSets
    /// 3. Greedy schedule with bitwise AND conflict checks
    pub fn schedule_bitset(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        if access_sets.is_empty() {
            return vec![];
        }

        // Build indexer from all access sets
        let indexer = AccountIndexer::from_access_sets(access_sets);

        // Convert to bitset access sets
        let bitset_access: Vec<BitsetAccessSet> = access_sets
            .iter()
            .map(|a| BitsetAccessSet::from_access_set(a, &indexer))
            .collect();

        // Fast path: check if all independent using bitsets
        if self.all_independent_bitset(&bitset_access) {
            let mut batch = TransactionBatch::new();
            for (idx, access) in access_sets.iter().enumerate() {
                batch.indices.push(idx);
                batch.write_set.extend(&access.writes);
                batch.read_set.extend(&access.reads);
            }
            return vec![batch];
        }

        // Greedy batching with O(1) conflict checks
        self.greedy_schedule_bitset(access_sets, &bitset_access)
    }

    /// Fast O(N) independence check using bitsets
    fn all_independent_bitset(&self, bitset_access: &[BitsetAccessSet]) -> bool {
        let mut combined_writes = crate::sealevel::Bitset256::new();
        let mut combined_reads = crate::sealevel::Bitset256::new();

        for access in bitset_access {
            // Check for conflicts with previously seen accounts
            // Write-write conflict
            if combined_writes.intersects(&access.writes) {
                return false;
            }
            // Write-read conflict (new writer vs old reader)
            if combined_reads.intersects(&access.writes) {
                return false;
            }
            // Read-write conflict (new reader vs old writer)
            if combined_writes.intersects(&access.reads) {
                return false;
            }

            // Add to combined sets
            combined_writes.union_inplace(&access.writes);
            combined_reads.union_inplace(&access.reads);
        }

        true
    }

    /// Greedy batching with O(1) bitset conflict detection
    fn greedy_schedule_bitset(
        &self,
        access_sets: &[AccessSet],
        bitset_access: &[BitsetAccessSet],
    ) -> Vec<TransactionBatch> {
        let mut batches: Vec<TransactionBatch> = Vec::new();
        let mut bitset_batches: Vec<BitsetBatch> = Vec::new();

        for (tx_idx, (access, bitset)) in access_sets.iter().zip(bitset_access.iter()).enumerate() {
            let mut added = false;

            // Try to add to existing batch using O(1) bitset check
            for (batch, bb) in batches.iter_mut().zip(bitset_batches.iter_mut()) {
                if batch.indices.len() < self.max_batch_size && bb.can_add(bitset) {
                    batch.indices.push(tx_idx);
                    batch.write_set.extend(&access.writes);
                    batch.read_set.extend(&access.reads);
                    bb.add(tx_idx, bitset);
                    added = true;
                    break;
                }
            }

            // Create new batch if needed
            if !added {
                let mut new_batch = TransactionBatch::new();
                new_batch.indices.push(tx_idx);
                new_batch.write_set.extend(&access.writes);
                new_batch.read_set.extend(&access.reads);
                batches.push(new_batch);

                let mut new_bb = BitsetBatch::new();
                new_bb.add(tx_idx, bitset);
                bitset_batches.push(new_bb);
            }
        }

        batches
    }
}

// ============================================================================
// CHUNKED PARALLEL SCHEDULING - O(N) instead of O(N²)
// ============================================================================

impl FastScheduler {
    /// Chunked parallel scheduling for massive transaction counts
    ///
    /// Instead of O(N × B) greedy scheduling where B grows with N,
    /// this splits into fixed-size chunks and schedules in parallel:
    /// - O(N/C) chunks, each with O(C × B_chunk) scheduling
    /// - Total: O(N × B_chunk) where B_chunk is bounded
    ///
    /// This is the key to scaling from 35K TPS to 170K+ TPS
    pub fn schedule_chunked(&self, access_sets: &[AccessSet], chunk_size: usize) -> Vec<TransactionBatch> {
        if access_sets.is_empty() {
            return vec![];
        }

        // For small inputs, use regular scheduling
        if access_sets.len() <= chunk_size {
            return self.schedule_bitset(access_sets);
        }

        // Split into chunks and schedule each in parallel
        let chunks: Vec<_> = access_sets.chunks(chunk_size).collect();

        let chunk_batches: Vec<Vec<TransactionBatch>> = chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_offset = chunk_idx * chunk_size;
                let local_batches = self.schedule_bitset(chunk);

                // Adjust indices to global positions
                local_batches
                    .into_iter()
                    .map(|mut batch| {
                        for idx in batch.indices.iter_mut() {
                            *idx += base_offset;
                        }
                        batch
                    })
                    .collect()
            })
            .collect();

        // Flatten all chunk batches
        chunk_batches.into_iter().flatten().collect()
    }

    /// Hybrid scheduling: fast-path for independent, chunked for conflicting
    ///
    /// This is the "holy optimization" - combines:
    /// - O(N) independence check
    /// - Direct batching if independent (170K+ TPS)
    /// - Chunked scheduling if conflicts exist (still ~100K+ TPS)
    pub fn schedule_adaptive(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        if access_sets.is_empty() {
            return vec![];
        }

        // Build indexer once for fast independence check
        let indexer = AccountIndexer::from_access_sets(access_sets);

        // Convert to bitsets
        let bitset_access: Vec<BitsetAccessSet> = access_sets
            .iter()
            .map(|a| BitsetAccessSet::from_access_set(a, &indexer))
            .collect();

        // O(N) independence check
        if self.all_independent_bitset(&bitset_access) {
            // FAST PATH: All independent - direct batching, no conflict checks
            let batcher = DirectBatcher::new(self.max_batch_size);
            return batcher.batch_direct(access_sets.len());
        }

        // CHUNKED PATH: Has conflicts but still fast via chunking
        // Use 10K chunk size - sweet spot for cache locality
        const CHUNK_SIZE: usize = 10_000;
        self.schedule_chunked(access_sets, CHUNK_SIZE)
    }
}

/// Ultra-fast scheduler for known-independent transactions
///
/// Use this when you know all transactions are independent
/// (e.g., from external pre-validation).
pub struct DirectBatcher {
    max_batch_size: usize,
}

impl DirectBatcher {
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }

    /// Create batches without conflict checking
    ///
    /// WARNING: Only use when you've pre-verified no conflicts exist!
    pub fn batch_direct(&self, count: usize) -> Vec<TransactionBatch> {
        if count == 0 {
            return vec![];
        }

        // Single batch if under max size
        if count <= self.max_batch_size {
            let mut batch = TransactionBatch::new();
            batch.indices = (0..count).collect();
            return vec![batch];
        }

        // Multiple batches if over max size
        let num_batches = (count + self.max_batch_size - 1) / self.max_batch_size;
        let mut batches = Vec::with_capacity(num_batches);

        let mut start = 0;
        while start < count {
            let end = (start + self.max_batch_size).min(count);
            let mut batch = TransactionBatch::new();
            batch.indices = (start..end).collect();
            batches.push(batch);
            start = end;
        }

        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_fast_scheduler_no_conflicts() {
        let scheduler = FastScheduler::new(10000);

        // 100 transactions, all different accounts
        let access_sets: Vec<AccessSet> = (0..100u8)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let batches = scheduler.schedule(&access_sets);

        // Should be single batch
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].indices.len(), 100);
    }

    #[test]
    fn test_fast_scheduler_with_conflicts() {
        let scheduler = FastScheduler::new(10000);

        // 4 transactions, 2 write to same account
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..4)
            .map(|i| {
                let mut a = AccessSet::new();
                if i < 2 {
                    a.add_write(pk);
                } else {
                    a.add_write(make_pubkey(i + 10));
                }
                a
            })
            .collect();

        let batches = scheduler.schedule(&access_sets);

        // Should have multiple batches due to conflict
        assert!(batches.len() >= 2);
    }

    #[test]
    fn test_direct_batcher() {
        let batcher = DirectBatcher::new(1000);

        let batches = batcher.batch_direct(2500);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].indices.len(), 1000);
        assert_eq!(batches[1].indices.len(), 1000);
        assert_eq!(batches[2].indices.len(), 500);
    }

    // ========================================
    // Bitset Scheduler Tests
    // ========================================

    #[test]
    fn test_bitset_scheduler_no_conflicts() {
        let scheduler = FastScheduler::new(10000);

        // 100 transactions, all different accounts
        let access_sets: Vec<AccessSet> = (0..100u8)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let batches = scheduler.schedule_bitset(&access_sets);

        // Should be single batch (same as HashSet version)
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].indices.len(), 100);
    }

    #[test]
    fn test_bitset_scheduler_with_conflicts() {
        let scheduler = FastScheduler::new(10000);

        // 4 transactions, 2 write to same account
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..4)
            .map(|i| {
                let mut a = AccessSet::new();
                if i < 2 {
                    a.add_write(pk);
                } else {
                    a.add_write(make_pubkey(i + 10));
                }
                a
            })
            .collect();

        let batches = scheduler.schedule_bitset(&access_sets);

        // Should have multiple batches due to conflict
        assert!(batches.len() >= 2);
    }

    #[test]
    fn test_bitset_matches_hashset_results() {
        let scheduler = FastScheduler::new(100);

        // Generate random-ish access patterns
        for seed in 0..50u8 {
            let access_sets: Vec<AccessSet> = (0..20)
                .map(|i| {
                    let mut a = AccessSet::new();
                    // Some writes, some reads, some overlap
                    a.add_write(make_pubkey(((seed as usize + i) % 30) as u8));
                    a.add_read(make_pubkey(((seed as usize + i + 5) % 30) as u8));
                    a
                })
                .collect();

            let hashset_batches = scheduler.schedule(&access_sets);
            let bitset_batches = scheduler.schedule_bitset(&access_sets);

            // Same number of batches
            assert_eq!(
                hashset_batches.len(),
                bitset_batches.len(),
                "Batch count mismatch at seed={}", seed
            );

            // Same transactions in each batch
            for (hb, bb) in hashset_batches.iter().zip(bitset_batches.iter()) {
                assert_eq!(
                    hb.indices, bb.indices,
                    "Batch indices mismatch at seed={}", seed
                );
            }
        }
    }

    #[test]
    fn test_bitset_scheduler_empty() {
        let scheduler = FastScheduler::new(100);
        let access_sets: Vec<AccessSet> = vec![];

        let batches = scheduler.schedule_bitset(&access_sets);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_bitset_read_read_no_conflict() {
        let scheduler = FastScheduler::new(10000);

        // Multiple transactions reading same account = no conflict
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..10)
            .map(|_| {
                let mut a = AccessSet::new();
                a.add_read(pk);
                a
            })
            .collect();

        let batches = scheduler.schedule_bitset(&access_sets);

        // Should be single batch (read-read is OK)
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].indices.len(), 10);
    }
}
