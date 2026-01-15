//! Fast Scheduler - Optimized Transaction Batching
//!
//! Uses fast-path detection to skip expensive conflict graph construction
//! when transactions don't conflict.

use crate::sealevel::{AccessSet, TransactionBatch};
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
}
