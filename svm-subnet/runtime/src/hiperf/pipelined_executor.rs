//! Pipelined Block Executor
//!
//! Overlaps execution stages across consecutive blocks for maximum throughput.
//!
//! Architecture:
//! ```text
//! Block N:   [Verify][Execute][Commit][Merkle]
//! Block N+1:         [Verify][Execute][Commit][Merkle]
//! Block N+2:                 [Verify][Execute][Commit][Merkle]
//! ```
//!
//! Each stage runs on a dedicated thread pool, allowing full overlap.

use crate::error::{RuntimeError, RuntimeResult};
use crate::executor::ExecutionContext;
use crate::qmdb_state::{InMemoryQMDBState, StateChangeSet};
use crate::sealevel::AccessSet;
use crate::types::{Account, Pubkey, Transaction};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

/// Pipeline stage data passed between stages
struct StageData {
    block_height: u64,
    transactions: Vec<Transaction>,
    ctx: ExecutionContext,
}

/// Verified stage output
struct VerifiedData {
    block_height: u64,
    transactions: Vec<Transaction>,
    valid_indices: Vec<usize>,
    access_sets: Vec<AccessSet>,
    ctx: ExecutionContext,
}

/// Executed stage output
struct ExecutedData {
    block_height: u64,
    changesets: Vec<StateChangeSet>,
    successful: u64,
    failed: u64,
}

/// Committed stage output
struct CommittedData {
    block_height: u64,
    merged_changeset: StateChangeSet,
}

/// Pipeline configuration
#[derive(Clone, Debug)]
pub struct BlockPipelineConfig {
    /// Channel buffer size between stages
    pub channel_buffer: usize,
    /// Number of verify threads
    pub verify_threads: usize,
    /// Number of execute threads
    pub execute_threads: usize,
    /// Number of commit threads
    pub commit_threads: usize,
}

impl Default for BlockPipelineConfig {
    fn default() -> Self {
        let total_threads = rayon::current_num_threads();
        Self {
            channel_buffer: 4,
            verify_threads: total_threads / 4,
            execute_threads: total_threads / 2,
            commit_threads: total_threads / 4,
        }
    }
}

/// Block result from pipeline
#[derive(Clone, Debug)]
pub struct PipelineBlockResult {
    pub block_height: u64,
    pub merkle_root: [u8; 32],
    pub successful: u64,
    pub failed: u64,
}

/// High-performance pipelined executor
///
/// Runs 4 stages in parallel:
/// 1. Verify - Ed25519 signature verification
/// 2. Execute - Transaction execution
/// 3. Commit - Parallel changeset merge
/// 4. Merkle - Root computation
pub struct PipelinedExecutor {
    state: Arc<InMemoryQMDBState>,
    config: BlockPipelineConfig,

    // Channels between stages
    input_tx: Sender<StageData>,
    output_rx: Receiver<PipelineBlockResult>,

    // Control
    running: Arc<AtomicBool>,

    // Stats
    blocks_processed: Arc<AtomicU64>,
}

impl PipelinedExecutor {
    pub fn new(state: Arc<InMemoryQMDBState>, config: BlockPipelineConfig) -> Self {
        let (input_tx, input_rx) = bounded(config.channel_buffer);
        let (verified_tx, verified_rx) = bounded(config.channel_buffer);
        let (executed_tx, executed_rx) = bounded(config.channel_buffer);
        let (committed_tx, committed_rx) = bounded(config.channel_buffer);
        let (output_tx, output_rx) = bounded(config.channel_buffer);

        let running = Arc::new(AtomicBool::new(true));
        let blocks_processed = Arc::new(AtomicU64::new(0));

        // Stage 1: Verify
        {
            let running = running.clone();
            let state = state.clone();
            thread::spawn(move || {
                verify_stage(input_rx, verified_tx, running);
            });
        }

        // Stage 2: Execute
        {
            let running = running.clone();
            let state = state.clone();
            thread::spawn(move || {
                execute_stage(verified_rx, executed_tx, state, running);
            });
        }

        // Stage 3: Commit (parallel merge)
        {
            let running = running.clone();
            thread::spawn(move || {
                commit_stage(executed_rx, committed_tx, running);
            });
        }

        // Stage 4: Merkle
        {
            let running = running.clone();
            let state = state.clone();
            let blocks_processed = blocks_processed.clone();
            thread::spawn(move || {
                merkle_stage(committed_rx, output_tx, state, blocks_processed, running);
            });
        }

        Self {
            state,
            config,
            input_tx,
            output_rx,
            running,
            blocks_processed,
        }
    }

    /// Submit a block for pipelined execution
    pub fn submit_block(
        &self,
        block_height: u64,
        transactions: Vec<Transaction>,
        ctx: ExecutionContext,
    ) -> RuntimeResult<()> {
        self.input_tx.send(StageData {
            block_height,
            transactions,
            ctx,
        }).map_err(|_| RuntimeError::State("Pipeline input full".to_string()))
    }

    /// Get next completed block result
    pub fn get_result(&self) -> RuntimeResult<PipelineBlockResult> {
        self.output_rx.recv()
            .map_err(|_| RuntimeError::State("Pipeline closed".to_string()))
    }

    /// Try to get result without blocking
    pub fn try_get_result(&self) -> Option<PipelineBlockResult> {
        self.output_rx.try_recv().ok()
    }

    /// Blocks processed
    pub fn blocks_processed(&self) -> u64 {
        self.blocks_processed.load(Ordering::Relaxed)
    }

    /// Shutdown the pipeline
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

impl Drop for PipelinedExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// Stage Implementations
// ============================================================================

/// Stage 1: Signature verification
fn verify_stage(
    input: Receiver<StageData>,
    output: Sender<VerifiedData>,
    running: Arc<AtomicBool>,
) {
    use ed25519_dalek::{VerifyingKey, Signature, Verifier};
    use sha2::{Sha256, Digest};

    while running.load(Ordering::Relaxed) {
        match input.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(data) => {
                // Parallel signature verification
                let results: Vec<bool> = data.transactions
                    .par_iter()
                    .map(|tx| verify_transaction_sigs(tx))
                    .collect();

                // Extract valid transactions and build access sets
                let (valid_indices, access_sets): (Vec<usize>, Vec<AccessSet>) = results
                    .iter()
                    .enumerate()
                    .filter(|(_, valid)| **valid)
                    .map(|(i, _)| (i, AccessSet::from_transaction(&data.transactions[i])))
                    .unzip();

                let _ = output.send(VerifiedData {
                    block_height: data.block_height,
                    transactions: data.transactions,
                    valid_indices,
                    access_sets,
                    ctx: data.ctx,
                });
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Stage 2: Transaction execution
fn execute_stage(
    input: Receiver<VerifiedData>,
    output: Sender<ExecutedData>,
    state: Arc<InMemoryQMDBState>,
    running: Arc<AtomicBool>,
) {
    use super::FastScheduler;

    let scheduler = FastScheduler::new(10000);

    while running.load(Ordering::Relaxed) {
        match input.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(data) => {
                // Begin block
                let _ = state.begin_block(data.block_height);

                // Schedule transactions
                let batches = scheduler.schedule(&data.access_sets);

                // Execute in parallel batches
                let mut all_changesets = Vec::new();
                let mut successful = 0u64;
                let mut failed = 0u64;

                for batch in &batches {
                    let batch_results: Vec<(StateChangeSet, bool)> = batch.indices
                        .par_iter()
                        .filter_map(|&batch_idx| data.valid_indices.get(batch_idx).copied())
                        .map(|tx_idx| {
                            let tx = &data.transactions[tx_idx];
                            execute_single_tx(tx, &state)
                        })
                        .collect();

                    for (cs, success) in batch_results {
                        all_changesets.push(cs);
                        if success {
                            successful += 1;
                        } else {
                            failed += 1;
                        }
                    }
                }

                let _ = output.send(ExecutedData {
                    block_height: data.block_height,
                    changesets: all_changesets,
                    successful,
                    failed,
                });
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Stage 3: Parallel changeset merge
fn commit_stage(
    input: Receiver<ExecutedData>,
    output: Sender<CommittedData>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        match input.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(data) => {
                // Parallel merge all changesets
                let merged = data.changesets
                    .into_par_iter()
                    .reduce(
                        StateChangeSet::new,
                        |mut acc, cs| { acc.merge(cs); acc }
                    );

                let _ = output.send(CommittedData {
                    block_height: data.block_height,
                    merged_changeset: merged,
                });
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Stage 4: Merkle root computation
fn merkle_stage(
    input: Receiver<CommittedData>,
    output: Sender<PipelineBlockResult>,
    state: Arc<InMemoryQMDBState>,
    blocks_processed: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        match input.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(data) => {
                // Apply merged changeset
                let _ = state.add_merged_changes(data.merged_changeset);

                // Compute merkle root (parallel internally)
                let merkle_root = state.commit_block().unwrap_or([0u8; 32]);

                blocks_processed.fetch_add(1, Ordering::Relaxed);

                let _ = output.send(PipelineBlockResult {
                    block_height: data.block_height,
                    merkle_root,
                    successful: 0, // TODO: pass through
                    failed: 0,
                });
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn verify_transaction_sigs(tx: &Transaction) -> bool {
    use ed25519_dalek::{VerifyingKey, Signature, Verifier};
    use sha2::{Sha256, Digest};

    let num_signers = tx.message.header.num_required_signatures as usize;

    if tx.signatures.len() < num_signers || tx.message.account_keys.len() < num_signers {
        return false;
    }

    // Compute message hash
    let mut hasher = Sha256::new();
    hasher.update(&[tx.message.header.num_required_signatures]);
    hasher.update(&[tx.message.header.num_readonly_signed_accounts]);
    hasher.update(&[tx.message.header.num_readonly_unsigned_accounts]);
    for key in &tx.message.account_keys {
        hasher.update(&key.0);
    }
    hasher.update(&tx.message.recent_blockhash);
    for ix in &tx.message.instructions {
        hasher.update(&[ix.program_id_index]);
        hasher.update(&[ix.accounts.len() as u8]);
        hasher.update(&ix.accounts);
        hasher.update(&(ix.data.len() as u16).to_le_bytes());
        hasher.update(&ix.data);
    }
    let msg_hash: [u8; 32] = hasher.finalize().into();

    for i in 0..num_signers {
        let Ok(pubkey) = VerifyingKey::from_bytes(&tx.message.account_keys[i].0) else {
            return false;
        };
        let signature = Signature::from_bytes(&tx.signatures[i]);
        if pubkey.verify(&msg_hash, &signature).is_err() {
            return false;
        }
    }

    true
}

fn execute_single_tx(tx: &Transaction, state: &InMemoryQMDBState) -> (StateChangeSet, bool) {
    let mut changeset = StateChangeSet::new();

    // Simple transfer execution (native)
    if tx.message.instructions.len() == 1 {
        let ix = &tx.message.instructions[0];

        // Check if system program transfer
        if ix.program_id_index < tx.message.account_keys.len() as u8 {
            let program = &tx.message.account_keys[ix.program_id_index as usize];

            if *program == Pubkey::system_program() && ix.data.len() >= 12 && ix.data[0] == 2 {
                // Transfer instruction
                let amount = u64::from_le_bytes(ix.data[4..12].try_into().unwrap_or([0; 8]));

                if ix.accounts.len() >= 2 {
                    let from_idx = ix.accounts[0] as usize;
                    let to_idx = ix.accounts[1] as usize;

                    if from_idx < tx.message.account_keys.len() && to_idx < tx.message.account_keys.len() {
                        let from_pk = &tx.message.account_keys[from_idx];
                        let to_pk = &tx.message.account_keys[to_idx];

                        if let (Ok(Some(mut from_acc)), Ok(Some(mut to_acc))) =
                            (state.get_account(from_pk), state.get_account(to_pk))
                        {
                            if from_acc.lamports >= amount {
                                from_acc.lamports -= amount;
                                to_acc.lamports += amount;

                                changeset.update(from_pk.clone(), from_acc);
                                changeset.update(to_pk.clone(), to_acc);

                                return (changeset, true);
                            }
                        }
                    }
                }
            }
        }
    }

    (changeset, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config() {
        let config = BlockPipelineConfig::default();
        assert!(config.verify_threads > 0);
        assert!(config.execute_threads > 0);
    }
}
