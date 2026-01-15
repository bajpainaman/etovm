//! QMDB-Optimized Parallel Executor
//!
//! Parallel executor designed for QMDB's block-level batching model.
//! Key differences from standard ParallelExecutor:
//! - Block-level state batching (begin_block/commit_block)
//! - Deferred state commits until block end
//! - Optimized for QMDB's Prefetcher-Updater-Flusher pipeline

use super::{AccessSet, AccountLocks, BatchScheduler, TransactionBatch};
use crate::error::{RuntimeError, RuntimeResult};
use crate::executor::{ExecutionContext, ExecutionResult, ExecutorConfig};
use crate::qmdb_state::{InMemoryQMDBState, QMDBState, StateChangeSet};
use crate::types::{Account, Pubkey, Transaction};
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for QMDB parallel executor
#[derive(Clone, Debug)]
pub struct QMDBExecutorConfig {
    /// Maximum transactions per batch
    pub max_batch_size: usize,
    /// Maximum number of worker threads (0 = use rayon default)
    pub num_threads: usize,
    /// Enable speculative execution
    pub speculative_execution: bool,
    /// Base executor config
    pub executor_config: ExecutorConfig,
}

impl Default for QMDBExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 30000,
            num_threads: 0,
            speculative_execution: false,
            executor_config: ExecutorConfig::default(),
        }
    }
}

/// Result of executing a block of transactions
#[derive(Debug)]
pub struct BlockExecutionResult {
    /// Results for each transaction (in original order)
    pub results: Vec<ExecutionResult>,
    /// Merkle root after block commit
    pub merkle_root: [u8; 32],
    /// Block height
    pub block_height: u64,
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
    /// Number of batches used
    pub num_batches: usize,
}

/// QMDB-optimized parallel executor
///
/// Designed for block-level execution with QMDB state storage.
/// All state changes are batched and committed atomically at block end.
pub struct QMDBParallelExecutor {
    config: QMDBExecutorConfig,
    state: Arc<InMemoryQMDBState>,
    locks: Arc<AccountLocks>,
    scheduler: BatchScheduler,
    // Metrics
    total_txs_processed: AtomicU64,
    total_blocks_processed: AtomicU64,
}

impl QMDBParallelExecutor {
    /// Create a new QMDB parallel executor
    pub fn new(state: Arc<InMemoryQMDBState>, config: QMDBExecutorConfig) -> Self {
        // Configure rayon thread pool if specified
        if config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build_global()
                .ok();
        }

        Self {
            scheduler: BatchScheduler::new(config.max_batch_size, 1000),
            config,
            state,
            locks: Arc::new(AccountLocks::new()),
            total_txs_processed: AtomicU64::new(0),
            total_blocks_processed: AtomicU64::new(0),
        }
    }

    /// Execute a block of transactions
    ///
    /// 1. Begins QMDB block
    /// 2. Analyzes and schedules transactions
    /// 3. Executes in parallel, collecting changesets
    /// 4. Commits block atomically
    pub fn execute_block(
        &self,
        block_height: u64,
        transactions: &[Transaction],
        ctx: &ExecutionContext,
    ) -> RuntimeResult<BlockExecutionResult> {
        let start_time = std::time::Instant::now();

        if transactions.is_empty() {
            return Ok(BlockExecutionResult {
                results: vec![],
                merkle_root: self.state.merkle_root()?,
                block_height,
                total_compute_units: 0,
                total_fees: 0,
                successful: 0,
                failed: 0,
                execution_time_us: 0,
                num_batches: 0,
            });
        }

        // Begin block
        self.state.begin_block(block_height)?;

        // Analyze access sets in parallel
        let access_sets: Vec<AccessSet> = transactions
            .par_iter()
            .map(AccessSet::from_transaction)
            .collect();

        // Schedule into batches
        let batches = self.scheduler.schedule_with_dependencies(&access_sets);
        let num_batches = batches.len();

        // Thread-safe results collection
        let results_map: DashMap<usize, (ExecutionResult, StateChangeSet)> = DashMap::new();

        // Execute batches
        for batch in &batches {
            self.execute_batch(transactions, &access_sets, batch, ctx, &results_map);
        }

        // Collect results in order and commit changesets
        let mut results = Vec::with_capacity(transactions.len());
        let mut total_compute = 0u64;
        let mut total_fees = 0u64;
        let mut successful = 0usize;
        let mut failed = 0usize;

        for idx in 0..transactions.len() {
            if let Some((_, (result, changeset))) = results_map.remove(&idx) {
                total_compute += result.compute_units_used;
                total_fees += result.fee;

                if result.success {
                    successful += 1;
                    // Add changeset to block batch
                    self.state.add_tx_changes(changeset)?;
                } else {
                    failed += 1;
                }

                results.push(result);
            } else {
                // Transaction wasn't executed
                failed += 1;
                results.push(ExecutionResult {
                    success: false,
                    compute_units_used: 0,
                    fee: 0,
                    state_changes: HashMap::new(),
                    logs: vec!["Transaction not scheduled".to_string()],
                    error: Some(RuntimeError::Program("Not executed".into())),
                });
            }
        }

        // Commit block
        let state_root = self.state.commit_block()?;

        // Update metrics
        self.total_txs_processed
            .fetch_add(transactions.len() as u64, Ordering::Relaxed);
        self.total_blocks_processed.fetch_add(1, Ordering::Relaxed);

        let execution_time_us = start_time.elapsed().as_micros() as u64;

        Ok(BlockExecutionResult {
            results,
            merkle_root: state_root,
            block_height,
            total_compute_units: total_compute,
            total_fees,
            successful,
            failed,
            execution_time_us,
            num_batches,
        })
    }

    /// Execute a single batch in parallel
    fn execute_batch(
        &self,
        transactions: &[Transaction],
        access_sets: &[AccessSet],
        batch: &TransactionBatch,
        ctx: &ExecutionContext,
        results_map: &DashMap<usize, (ExecutionResult, StateChangeSet)>,
    ) {
        batch.indices.par_iter().for_each(|&tx_idx| {
            let tx = &transactions[tx_idx];
            let access = &access_sets[tx_idx];

            // Acquire locks
            let _guard = self.locks.lock_accounts(&access.reads, &access.writes);

            // Execute transaction
            let (result, changeset) = self.execute_single_transaction(tx, ctx);

            results_map.insert(tx_idx, (result, changeset));
        });
    }

    /// Execute a single transaction
    fn execute_single_transaction(
        &self,
        tx: &Transaction,
        ctx: &ExecutionContext,
    ) -> (ExecutionResult, StateChangeSet) {
        // Load accounts
        let mut accounts: Vec<(Pubkey, Account)> = Vec::new();
        for pubkey in &tx.message.account_keys {
            let account = self.state.get_account(pubkey)
                .unwrap_or(None)
                .unwrap_or_default();
            accounts.push((*pubkey, account));
        }

        // Create changeset for this transaction
        let mut changeset = StateChangeSet::new();

        // Simplified execution - just process system program transfers
        // In full implementation, this would call the actual executor
        let result = self.execute_tx_simple(tx, &mut accounts, &mut changeset, ctx);

        (result, changeset)
    }

    /// Simplified transaction execution (for benchmarking)
    fn execute_tx_simple(
        &self,
        tx: &Transaction,
        accounts: &mut [(Pubkey, Account)],
        changeset: &mut StateChangeSet,
        _ctx: &ExecutionContext,
    ) -> ExecutionResult {
        // Basic validation
        if tx.message.instructions.is_empty() {
            return ExecutionResult {
                success: false,
                compute_units_used: 0,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec![],
                error: Some(RuntimeError::Program("No instructions".into())),
            };
        }

        // Process first instruction only (system transfer)
        let ix = &tx.message.instructions[0];

        // Check if it's a system program transfer
        if ix.program_id_index as usize >= tx.message.account_keys.len() {
            return ExecutionResult {
                success: false,
                compute_units_used: 0,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec![],
                error: Some(RuntimeError::Program("Invalid program index".into())),
            };
        }

        let program_id = &tx.message.account_keys[ix.program_id_index as usize];

        // Only handle system program
        if program_id != &Pubkey::system_program() {
            return ExecutionResult {
                success: true,
                compute_units_used: 150, // Base CU
                fee: 5000,
                state_changes: HashMap::new(),
                logs: vec!["Skipped non-system instruction".to_string()],
                error: None,
            };
        }

        // Parse transfer instruction
        if ix.data.len() < 12 || ix.data[0] != 2 {
            return ExecutionResult {
                success: false,
                compute_units_used: 0,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec![],
                error: Some(RuntimeError::Program("Invalid transfer instruction".into())),
            };
        }

        // Get transfer amount
        let amount = u64::from_le_bytes(ix.data[4..12].try_into().unwrap());

        // Get source and dest indices
        if ix.accounts.len() < 2 {
            return ExecutionResult {
                success: false,
                compute_units_used: 0,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec![],
                error: Some(RuntimeError::Program("Missing accounts".into())),
            };
        }

        let from_idx = ix.accounts[0] as usize;
        let to_idx = ix.accounts[1] as usize;

        if from_idx >= accounts.len() || to_idx >= accounts.len() {
            return ExecutionResult {
                success: false,
                compute_units_used: 0,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec![],
                error: Some(RuntimeError::Program("Invalid account indices".into())),
            };
        }

        // Execute transfer
        let from_balance = accounts[from_idx].1.lamports;
        if from_balance < amount {
            return ExecutionResult {
                success: false,
                compute_units_used: 150,
                fee: 0,
                state_changes: HashMap::new(),
                logs: vec!["Insufficient balance".to_string()],
                error: Some(RuntimeError::Program("Insufficient lamports".into())),
            };
        }

        // Update balances
        accounts[from_idx].1.lamports -= amount;
        accounts[to_idx].1.lamports += amount;

        // Record changes
        let mut state_changes = HashMap::new();
        state_changes.insert(accounts[from_idx].0, accounts[from_idx].1.clone());
        state_changes.insert(accounts[to_idx].0, accounts[to_idx].1.clone());

        changeset.update(accounts[from_idx].0, accounts[from_idx].1.clone());
        changeset.update(accounts[to_idx].0, accounts[to_idx].1.clone());

        ExecutionResult {
            success: true,
            compute_units_used: 450,
            fee: 5000,
            state_changes,
            logs: vec![format!("Transfer {} lamports", amount)],
            error: None,
        }
    }

    /// Get executor statistics
    pub fn get_stats(&self) -> QMDBExecutorStats {
        QMDBExecutorStats {
            total_txs_processed: self.total_txs_processed.load(Ordering::Relaxed),
            total_blocks_processed: self.total_blocks_processed.load(Ordering::Relaxed),
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Get current state root / merkle root
    pub fn get_merkle_root(&self) -> RuntimeResult<[u8; 32]> {
        self.state.merkle_root()
    }

    /// Get account from state
    pub fn get_account(&self, pubkey: &Pubkey) -> Option<Account> {
        self.state.get_account(pubkey).ok().flatten()
    }

    /// Load account into state (for initialization/testing)
    pub fn load_account(&self, pubkey: Pubkey, account: Account) {
        let _ = self.state.set_account(&pubkey, &account);
    }
}

/// QMDB executor statistics
#[derive(Debug, Clone)]
pub struct QMDBExecutorStats {
    pub total_txs_processed: u64,
    pub total_blocks_processed: u64,
    pub num_threads: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompiledInstruction, Message, MessageHeader};

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    fn setup_state_with_accounts(accounts: Vec<(Pubkey, u64)>) -> Arc<InMemoryQMDBState> {
        let state = Arc::new(InMemoryQMDBState::new());
        for (pubkey, lamports) in accounts {
            state.set_account(&pubkey, &Account {
                lamports,
                data: vec![],
                owner: Pubkey::system_program(),
                executable: false,
                rent_epoch: 0,
            }).unwrap();
        }
        state
    }

    fn make_transfer_tx(from: Pubkey, to: Pubkey, amount: u64) -> Transaction {
        let mut data = vec![2, 0, 0, 0]; // Transfer instruction
        data.extend_from_slice(&amount.to_le_bytes());

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
                    data,
                }],
            },
        }
    }

    #[test]
    fn test_qmdb_executor_creation() {
        let state = Arc::new(InMemoryQMDBState::new());
        let executor = QMDBParallelExecutor::new(state, QMDBExecutorConfig::default());

        let stats = executor.get_stats();
        assert_eq!(stats.total_txs_processed, 0);
        assert_eq!(stats.total_blocks_processed, 0);
    }

    #[test]
    fn test_execute_empty_block() {
        let state = Arc::new(InMemoryQMDBState::new());
        let executor = QMDBParallelExecutor::new(state, QMDBExecutorConfig::default());
        let ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());

        let result = executor.execute_block(1, &[], &ctx).unwrap();
        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(result.num_batches, 0);
    }

    #[test]
    fn test_execute_block_with_transfers() {
        let pk1 = make_pubkey(1);
        let pk2 = make_pubkey(2);
        let pk3 = make_pubkey(3);
        let pk4 = make_pubkey(4);

        let state = setup_state_with_accounts(vec![
            (pk1, 1_000_000),
            (pk2, 1_000_000),
            (pk3, 0),
            (pk4, 0),
        ]);

        let executor = QMDBParallelExecutor::new(state, QMDBExecutorConfig::default());
        let ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());

        // Two non-conflicting transfers
        let txs = vec![
            make_transfer_tx(pk1, pk3, 100),
            make_transfer_tx(pk2, pk4, 200),
        ];

        let result = executor.execute_block(1, &txs, &ctx).unwrap();

        assert_eq!(result.successful, 2);
        assert_eq!(result.failed, 0);
        assert_eq!(result.num_batches, 1); // Should batch together

        // Verify state changes
        let acc1 = executor.get_account(&pk1).unwrap();
        assert_eq!(acc1.lamports, 1_000_000 - 100);

        let acc3 = executor.get_account(&pk3).unwrap();
        assert_eq!(acc3.lamports, 100);
    }

    #[test]
    fn test_conflicting_transactions_separate_batches() {
        let pk1 = make_pubkey(1);
        let pk2 = make_pubkey(2);

        let state = setup_state_with_accounts(vec![
            (pk1, 1_000_000),
            (pk2, 0),
        ]);

        let executor = QMDBParallelExecutor::new(state, QMDBExecutorConfig::default());
        let ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());

        // Two conflicting transfers (same source)
        let txs = vec![
            make_transfer_tx(pk1, pk2, 100),
            make_transfer_tx(pk1, pk2, 200),
        ];

        let result = executor.execute_block(1, &txs, &ctx).unwrap();

        // Both should succeed but in separate batches
        assert_eq!(result.successful, 2);
        assert_eq!(result.num_batches, 2); // Conflicting = separate batches
    }

    #[test]
    fn test_insufficient_balance() {
        let pk1 = make_pubkey(1);
        let pk2 = make_pubkey(2);

        let state = setup_state_with_accounts(vec![
            (pk1, 50), // Only 50 lamports
            (pk2, 0),
        ]);

        let executor = QMDBParallelExecutor::new(state, QMDBExecutorConfig::default());
        let ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());

        let txs = vec![
            make_transfer_tx(pk1, pk2, 100), // Try to transfer 100
        ];

        let result = executor.execute_block(1, &txs, &ctx).unwrap();

        assert_eq!(result.successful, 0);
        assert_eq!(result.failed, 1);

        // State should be unchanged
        let acc1 = executor.get_account(&pk1).unwrap();
        assert_eq!(acc1.lamports, 50);
    }
}
