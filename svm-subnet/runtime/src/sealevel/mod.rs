//! Sealevel - Parallel Transaction Scheduler
//!
//! This module implements Solana's Sealevel parallel runtime, which enables
//! concurrent execution of non-conflicting transactions.
//!
//! Key components:
//! - `AccessSet`: Tracks read/write account access for each transaction
//! - `ConflictGraph`: Builds dependency graph between transactions
//! - `BatchScheduler`: Groups non-conflicting transactions into parallel batches
//! - `ParallelExecutor`: Executes batches using rayon thread pool

mod access_set;
mod batch_scheduler;
mod conflict_graph;
mod parallel_executor;
mod account_locks;
mod qmdb_executor;
pub mod benchmark;

pub use access_set::{AccessSet, AccessType};
pub use batch_scheduler::{BatchScheduler, TransactionBatch};
pub use conflict_graph::ConflictGraph;
pub use parallel_executor::{ParallelExecutor, ParallelExecutorConfig, BatchExecutionResult};
pub use account_locks::{AccountLocks, AccountLockGuard};
pub use qmdb_executor::{QMDBParallelExecutor, QMDBExecutorConfig, BlockExecutionResult, QMDBExecutorStats};

use crate::types::{Pubkey, Transaction};

/// Analyze a transaction to extract its read/write sets
pub fn analyze_transaction(tx: &Transaction) -> AccessSet {
    AccessSet::from_transaction(tx)
}

/// Check if two transactions conflict (share writable accounts)
pub fn transactions_conflict(tx1: &Transaction, tx2: &Transaction) -> bool {
    let access1 = AccessSet::from_transaction(tx1);
    let access2 = AccessSet::from_transaction(tx2);
    access1.conflicts_with(&access2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Ensure all types are accessible
        let _: AccessType = AccessType::Read;
    }
}
