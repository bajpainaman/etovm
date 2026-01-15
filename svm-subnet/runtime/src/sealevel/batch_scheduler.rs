//! Batch Scheduler - Groups Transactions for Parallel Execution
//!
//! Takes a set of transactions and their access sets, and produces
//! batches of non-conflicting transactions that can execute in parallel.

use super::{AccessSet, ConflictGraph};
use crate::types::{Pubkey, Transaction};
use std::collections::HashSet;

/// A batch of transactions that can execute in parallel
#[derive(Debug)]
pub struct TransactionBatch {
    /// Indices of transactions in this batch
    pub indices: Vec<usize>,
    /// Combined write set (accounts being modified)
    pub write_set: HashSet<Pubkey>,
    /// Combined read set (accounts being read)
    pub read_set: HashSet<Pubkey>,
}

impl TransactionBatch {
    /// Create an empty batch
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            write_set: HashSet::new(),
            read_set: HashSet::new(),
        }
    }

    /// Check if a transaction can be added to this batch without conflict
    pub fn can_add(&self, access: &AccessSet) -> bool {
        // Check for write-write conflicts
        for pubkey in &access.writes {
            if self.write_set.contains(pubkey) {
                return false;
            }
        }

        // Check for write-read conflicts
        for pubkey in &access.writes {
            if self.read_set.contains(pubkey) {
                return false;
            }
        }

        // Check for read-write conflicts
        for pubkey in &access.reads {
            if self.write_set.contains(pubkey) {
                return false;
            }
        }

        true
    }

    /// Add a transaction to this batch
    pub fn add(&mut self, tx_idx: usize, access: &AccessSet) {
        self.indices.push(tx_idx);
        self.write_set.extend(&access.writes);
        self.read_set.extend(&access.reads);
    }

    /// Get the number of transactions in this batch
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl Default for TransactionBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch scheduler using greedy algorithm
pub struct BatchScheduler {
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum number of batches per scheduling round
    max_batches: usize,
}

impl BatchScheduler {
    /// Create a new batch scheduler
    pub fn new(max_batch_size: usize, max_batches: usize) -> Self {
        Self {
            max_batch_size,
            max_batches,
        }
    }

    /// Schedule transactions into parallel batches
    ///
    /// Uses a greedy algorithm that processes transactions in order,
    /// adding each to the first batch it doesn't conflict with.
    pub fn schedule(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        let mut batches: Vec<TransactionBatch> = Vec::new();

        for (tx_idx, access) in access_sets.iter().enumerate() {
            let mut added = false;

            // Try to add to existing batch
            for batch in batches.iter_mut() {
                if batch.len() < self.max_batch_size && batch.can_add(access) {
                    batch.add(tx_idx, access);
                    added = true;
                    break;
                }
            }

            // Create new batch if needed
            if !added {
                let mut new_batch = TransactionBatch::new();
                new_batch.add(tx_idx, access);
                batches.push(new_batch);
            }
        }

        batches
    }

    /// Schedule using dependency graph (respects execution order)
    ///
    /// More sophisticated scheduling that uses topological sort
    /// to ensure dependencies are respected while maximizing parallelism.
    pub fn schedule_with_dependencies(&self, access_sets: &[AccessSet]) -> Vec<TransactionBatch> {
        let graph = ConflictGraph::from_access_sets(access_sets);

        let topo_order = match graph.topological_order() {
            Some(order) => order,
            None => {
                // Fallback to sequential if cycle detected (shouldn't happen)
                return access_sets
                    .iter()
                    .enumerate()
                    .map(|(idx, access)| {
                        let mut batch = TransactionBatch::new();
                        batch.add(idx, access);
                        batch
                    })
                    .collect();
            }
        };

        // Level-based scheduling: transactions at same level can run in parallel
        let mut levels = vec![0usize; access_sets.len()];
        for &tx_idx in &topo_order {
            let my_level = graph
                .get_dependencies(tx_idx)
                .iter()
                .map(|&dep| levels[dep] + 1)
                .max()
                .unwrap_or(0);
            levels[tx_idx] = my_level;
        }

        // Group by level
        let max_level = levels.iter().max().copied().unwrap_or(0);
        let mut batches: Vec<TransactionBatch> = Vec::with_capacity(max_level + 1);

        for level in 0..=max_level {
            let mut current_batches: Vec<TransactionBatch> = Vec::new();

            for (tx_idx, &tx_level) in levels.iter().enumerate() {
                if tx_level != level {
                    continue;
                }

                let access = &access_sets[tx_idx];
                let mut added = false;

                // Try to add to existing batch at this level
                for batch in current_batches.iter_mut() {
                    if batch.len() < self.max_batch_size && batch.can_add(access) {
                        batch.add(tx_idx, access);
                        added = true;
                        break;
                    }
                }

                if !added {
                    let mut new_batch = TransactionBatch::new();
                    new_batch.add(tx_idx, access);
                    current_batches.push(new_batch);
                }
            }

            batches.extend(current_batches);
        }

        batches
    }

    /// Get statistics about scheduling
    pub fn get_stats(&self, batches: &[TransactionBatch]) -> ScheduleStats {
        let total_txs: usize = batches.iter().map(|b| b.len()).sum();
        let max_batch_size = batches.iter().map(|b| b.len()).max().unwrap_or(0);
        let avg_batch_size = if batches.is_empty() {
            0.0
        } else {
            total_txs as f64 / batches.len() as f64
        };

        // Parallelism factor: how many txs can run in parallel vs sequential
        let parallelism = if batches.is_empty() {
            1.0
        } else {
            total_txs as f64 / batches.len() as f64
        };

        ScheduleStats {
            total_transactions: total_txs,
            num_batches: batches.len(),
            max_batch_size,
            avg_batch_size,
            parallelism_factor: parallelism,
        }
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(30000, 1000) // Increased for high TPS
    }
}

/// Statistics about a scheduling result
#[derive(Debug, Clone)]
pub struct ScheduleStats {
    pub total_transactions: usize,
    pub num_batches: usize,
    pub max_batch_size: usize,
    pub avg_batch_size: f64,
    pub parallelism_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_all_parallel() {
        // 4 transactions, all different accounts
        let access_sets: Vec<AccessSet> = (0..4)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let scheduler = BatchScheduler::new(1000, 100);
        let batches = scheduler.schedule(&access_sets);

        // Should be 1 batch with all 4
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn test_all_sequential() {
        // 4 transactions, all same account
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..4)
            .map(|_| {
                let mut a = AccessSet::new();
                a.add_write(pk);
                a
            })
            .collect();

        let scheduler = BatchScheduler::new(1000, 100);
        let batches = scheduler.schedule(&access_sets);

        // Should be 4 batches with 1 each
        assert_eq!(batches.len(), 4);
        for batch in &batches {
            assert_eq!(batch.len(), 1);
        }
    }

    #[test]
    fn test_mixed_parallelism() {
        // 6 transactions: 0,1,2 write A; 3,4,5 write B
        let pk_a = make_pubkey(1);
        let pk_b = make_pubkey(2);

        let access_sets: Vec<AccessSet> = (0..6)
            .map(|i| {
                let mut a = AccessSet::new();
                if i < 3 {
                    a.add_write(pk_a);
                } else {
                    a.add_write(pk_b);
                }
                a
            })
            .collect();

        let scheduler = BatchScheduler::new(1000, 100);
        let batches = scheduler.schedule(&access_sets);

        // Should be 3 batches: (0,3), (1,4), (2,5)
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            assert_eq!(batch.len(), 2);
        }
    }

    #[test]
    fn test_stats() {
        let access_sets: Vec<AccessSet> = (0..10)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let scheduler = BatchScheduler::new(1000, 100);
        let batches = scheduler.schedule(&access_sets);
        let stats = scheduler.get_stats(&batches);

        assert_eq!(stats.total_transactions, 10);
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.parallelism_factor, 10.0);
    }
}
