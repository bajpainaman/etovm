//! Parallel Executor - Rayon-Powered Transaction Execution
//!
//! Executes batches of transactions in parallel using the rayon thread pool.
//! This is the core of Sealevel's performance.

use super::{AccessSet, AccountLocks, BatchScheduler, TransactionBatch};
use crate::accounts::{AccountsDB, AccountsManager};
use crate::error::{RuntimeError, RuntimeResult};
use crate::executor::{ExecutionContext, ExecutionResult, Executor, ExecutorConfig};
use crate::types::{Account, Pubkey, Transaction};
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for parallel executor
#[derive(Clone, Debug)]
pub struct ParallelExecutorConfig {
    /// Maximum transactions per batch
    pub max_batch_size: usize,
    /// Maximum number of worker threads (0 = use rayon default)
    pub num_threads: usize,
    /// Enable speculative execution (rollback on conflict)
    pub speculative_execution: bool,
    /// Base executor config
    pub executor_config: ExecutorConfig,
}

impl Default for ParallelExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 30000, // Increased for high TPS
            num_threads: 0, // Use rayon default (num CPUs)
            speculative_execution: false,
            executor_config: ExecutorConfig::default(),
        }
    }
}

/// Result of executing a batch of transactions
#[derive(Debug)]
pub struct BatchExecutionResult {
    /// Results for each transaction (in original order)
    pub results: Vec<ExecutionResult>,
    /// Combined state changes from all successful transactions
    pub state_changes: HashMap<Pubkey, Account>,
    /// Total compute units used
    pub total_compute_units: u64,
    /// Total fees collected
    pub total_fees: u64,
    /// Number of successful transactions
    pub successful: usize,
    /// Number of failed transactions
    pub failed: usize,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Parallel transaction executor
///
/// Implements Sealevel's parallel execution model using rayon.
pub struct ParallelExecutor<DB: AccountsDB + Clone + 'static> {
    config: ParallelExecutorConfig,
    accounts: AccountsManager<DB>,
    locks: Arc<AccountLocks>,
    scheduler: BatchScheduler,
    // Metrics
    total_txs_processed: AtomicU64,
    total_batches_processed: AtomicU64,
}

impl<DB: AccountsDB + Clone + Send + Sync + 'static> ParallelExecutor<DB> {
    /// Create a new parallel executor
    pub fn new(accounts: AccountsManager<DB>, config: ParallelExecutorConfig) -> Self {
        // Configure rayon thread pool if specified
        if config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        Self {
            scheduler: BatchScheduler::new(config.max_batch_size, 100),
            config,
            accounts,
            locks: Arc::new(AccountLocks::new()),
            total_txs_processed: AtomicU64::new(0),
            total_batches_processed: AtomicU64::new(0),
        }
    }

    /// Execute a batch of transactions in parallel
    ///
    /// 1. Analyzes access sets for all transactions
    /// 2. Schedules into non-conflicting batches
    /// 3. Executes batches in parallel using rayon
    /// 4. Commits successful state changes
    pub fn execute_transactions(
        &self,
        transactions: &[Transaction],
        ctx: &ExecutionContext,
    ) -> BatchExecutionResult {
        let start_time = std::time::Instant::now();

        if transactions.is_empty() {
            return BatchExecutionResult {
                results: vec![],
                state_changes: HashMap::new(),
                total_compute_units: 0,
                total_fees: 0,
                successful: 0,
                failed: 0,
                execution_time_us: 0,
            };
        }

        // Step 1: Analyze access sets
        let access_sets: Vec<AccessSet> = transactions
            .par_iter()
            .map(AccessSet::from_transaction)
            .collect();

        // Step 2: Schedule into batches
        let batches = self.scheduler.schedule_with_dependencies(&access_sets);

        // Step 3: Execute batches
        // Use indices to track which tx results go where
        let mut results_map: HashMap<usize, ExecutionResult> = HashMap::new();
        let mut combined_state_changes: HashMap<Pubkey, Account> = HashMap::new();
        let mut total_compute = 0u64;
        let mut total_fees = 0u64;
        let mut successful = 0usize;
        let mut failed = 0usize;

        // Process batches sequentially (within batch is parallel)
        for batch in &batches {
            let mut batch_results = self.execute_batch(transactions, &access_sets, batch, ctx);

            for (local_idx, tx_idx) in batch.indices.iter().enumerate() {
                if local_idx < batch_results.len() {
                    // Take ownership by swapping with a dummy result
                    let result = std::mem::replace(&mut batch_results[local_idx], ExecutionResult {
                        success: false,
                        compute_units_used: 0,
                        fee: 0,
                        state_changes: HashMap::new(),
                        logs: vec![],
                        error: None,
                    });

                    total_compute += result.compute_units_used;
                    total_fees += result.fee;

                    if result.success {
                        successful += 1;
                        // Merge state changes
                        for (pk, acc) in &result.state_changes {
                            combined_state_changes.insert(*pk, acc.clone());
                        }
                    } else {
                        failed += 1;
                    }

                    results_map.insert(*tx_idx, result);
                }
            }
        }

        // Update metrics
        self.total_txs_processed
            .fetch_add(transactions.len() as u64, Ordering::Relaxed);
        self.total_batches_processed
            .fetch_add(batches.len() as u64, Ordering::Relaxed);

        let execution_time_us = start_time.elapsed().as_micros() as u64;

        // Convert to ordered Vec
        let results: Vec<ExecutionResult> = (0..transactions.len())
            .map(|idx| {
                results_map.remove(&idx).unwrap_or_else(|| ExecutionResult {
                    success: false,
                    compute_units_used: 0,
                    fee: 0,
                    state_changes: HashMap::new(),
                    logs: vec!["Transaction not executed".to_string()],
                    error: Some(RuntimeError::Program("Not executed".into())),
                })
            })
            .collect();

        BatchExecutionResult {
            results,
            state_changes: combined_state_changes,
            total_compute_units: total_compute,
            total_fees: total_fees,
            successful,
            failed,
            execution_time_us,
        }
    }

    /// Execute a single batch in parallel
    fn execute_batch(
        &self,
        transactions: &[Transaction],
        access_sets: &[AccessSet],
        batch: &TransactionBatch,
        ctx: &ExecutionContext,
    ) -> Vec<ExecutionResult> {
        if batch.indices.len() == 1 {
            // Single transaction - execute directly
            let tx_idx = batch.indices[0];
            let result = self.execute_single_transaction(&transactions[tx_idx], ctx);
            return vec![result];
        }

        // Parallel execution using rayon
        batch
            .indices
            .par_iter()
            .map(|&tx_idx| {
                let tx = &transactions[tx_idx];
                let access = &access_sets[tx_idx];

                // Acquire locks
                let _guard = self.locks.lock_accounts(&access.reads, &access.writes);

                // Execute transaction
                self.execute_single_transaction(tx, ctx)
            })
            .collect()
    }

    /// Execute a single transaction
    fn execute_single_transaction(&self, tx: &Transaction, ctx: &ExecutionContext) -> ExecutionResult {
        // Create executor for this transaction
        let executor = Executor::new(self.accounts.clone(), self.config.executor_config.clone());

        // Create mutable context copy
        let mut tx_ctx = ExecutionContext::new(
            ctx.slot,
            ctx.timestamp,
            self.config.executor_config.clone(),
        );
        tx_ctx.recent_blockhashes = ctx.recent_blockhashes.clone();

        // Execute
        let result = executor.execute_transaction(tx, &mut tx_ctx);

        // Commit if successful
        if result.success {
            let _ = executor.commit(&result);
        }

        result
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> ExecutorStats {
        ExecutorStats {
            total_txs_processed: self.total_txs_processed.load(Ordering::Relaxed),
            total_batches_processed: self.total_batches_processed.load(Ordering::Relaxed),
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Get account locks (for debugging/monitoring)
    pub fn account_locks(&self) -> &AccountLocks {
        &self.locks
    }
}

/// Executor statistics
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub total_txs_processed: u64,
    pub total_batches_processed: u64,
    pub num_threads: usize,
}

/// Execute transactions in parallel (convenience function)
pub fn execute_parallel<DB: AccountsDB + Clone + Send + Sync + 'static>(
    transactions: &[Transaction],
    accounts: AccountsManager<DB>,
    ctx: &ExecutionContext,
) -> BatchExecutionResult {
    let executor = ParallelExecutor::new(accounts, ParallelExecutorConfig::default());
    executor.execute_transactions(transactions, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accounts::InMemoryAccountsDB;
    use crate::types::{CompiledInstruction, Message, MessageHeader};

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    fn make_simple_tx(from: Pubkey, to: Pubkey) -> Transaction {
        Transaction {
            signatures: vec![[0u8; 64]],
            message: Message {
                header: MessageHeader {
                    num_required_signatures: 1,
                    num_readonly_signed_accounts: 0,
                    num_readonly_unsigned_accounts: 1,
                },
                account_keys: vec![from, to, Pubkey::system_program()],
                recent_blockhash: [0u8; 32],
                instructions: vec![CompiledInstruction {
                    program_id_index: 2,
                    accounts: vec![0, 1],
                    data: vec![2, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0], // Transfer 100
                }],
            },
        }
    }

    #[test]
    fn test_parallel_executor_creation() {
        let accounts = AccountsManager::new(InMemoryAccountsDB::new());
        let executor = ParallelExecutor::new(accounts, ParallelExecutorConfig::default());

        let stats = executor.get_stats();
        assert_eq!(stats.total_txs_processed, 0);
        assert!(stats.num_threads > 0);
    }

    #[test]
    fn test_empty_batch() {
        let accounts = AccountsManager::new(InMemoryAccountsDB::new());
        let executor = ParallelExecutor::new(accounts, ParallelExecutorConfig::default());
        let ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());

        let result = executor.execute_transactions(&[], &ctx);
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_parallel_non_conflicting() {
        // Test that non-conflicting transactions get batched together
        let access_sets: Vec<AccessSet> = (0..4)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let scheduler = BatchScheduler::new(1000, 100);
        let batches = scheduler.schedule(&access_sets);

        // All 4 should be in a single batch (no conflicts)
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn test_conflicting_transactions() {
        // Test that conflicting transactions get separate batches
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

        // Should be 4 batches (all conflict on same account)
        assert_eq!(batches.len(), 4);
        for batch in &batches {
            assert_eq!(batch.len(), 1);
        }
    }
}
