//! QMDB State Storage
//!
//! High-performance state storage using QMDB (Quick Merkle Database).
//! Provides O(1) I/O per update and efficient Merkle proofs.
//!
//! Architecture:
//! - Prefetcher-Updater-Flusher pipeline for parallel I/O
//! - Twig-based Merkle tree minimizes RAM usage
//! - Append-only design optimized for SSD write patterns

use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, Pubkey};
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use parking_lot::Mutex;

/// High-performance DashMap with FxHasher (3-5x faster than SipHash)
type FastDashMap<K, V> = DashMap<K, V, FxBuildHasher>;

// Re-export QMDB types when the feature is enabled
#[cfg(feature = "qmdb")]
use qmdb::{
    config::Config as QMDBConfig,
    entryfile::entry::{Entry, EntryBz},
    tasks::{Task, TaskHub},
};

/// Configuration for QMDB state storage
#[derive(Clone, Debug)]
pub struct QMDBStateConfig {
    /// Directory for QMDB data files
    pub data_dir: PathBuf,
    /// Number of shards (default: 16)
    pub shard_count: usize,
    /// Maximum entries per twig (default: 2048)
    pub twig_size: usize,
    /// Enable direct I/O (Linux only)
    pub direct_io: bool,
    /// In-memory mode for testing
    pub in_memory: bool,
}

impl Default for QMDBStateConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./qmdb_data"),
            shard_count: 16,
            twig_size: 2048,
            direct_io: true,
            in_memory: false,
        }
    }
}

/// Account serialization format for QMDB
///
/// Layout:
/// - 8 bytes: lamports (little-endian u64)
/// - 4 bytes: data length (little-endian u32)
/// - N bytes: account data
/// - 32 bytes: owner pubkey
/// - 1 byte: executable flag
/// - 8 bytes: rent_epoch (little-endian u64)
const ACCOUNT_HEADER_SIZE: usize = 8 + 4 + 32 + 1 + 8; // 53 bytes

fn serialize_account(account: &Account) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ACCOUNT_HEADER_SIZE + account.data.len());

    // Lamports (8 bytes)
    buf.extend_from_slice(&account.lamports.to_le_bytes());

    // Data length (4 bytes)
    buf.extend_from_slice(&(account.data.len() as u32).to_le_bytes());

    // Data (variable)
    buf.extend_from_slice(&account.data);

    // Owner (32 bytes)
    buf.extend_from_slice(&account.owner.0);

    // Executable (1 byte)
    buf.push(if account.executable { 1 } else { 0 });

    // Rent epoch (8 bytes)
    buf.extend_from_slice(&account.rent_epoch.to_le_bytes());

    buf
}

fn deserialize_account(data: &[u8]) -> RuntimeResult<Account> {
    if data.len() < ACCOUNT_HEADER_SIZE {
        return Err(RuntimeError::State("Account data too short".to_string()));
    }

    let lamports = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let data_len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

    if data.len() < 12 + data_len + 32 + 1 + 8 {
        return Err(RuntimeError::State("Account data truncated".to_string()));
    }

    let account_data = data[12..12 + data_len].to_vec();

    let offset = 12 + data_len;
    let mut owner = [0u8; 32];
    owner.copy_from_slice(&data[offset..offset + 32]);

    let executable = data[offset + 32] != 0;
    let rent_epoch = u64::from_le_bytes(data[offset + 33..offset + 41].try_into().unwrap());

    Ok(Account {
        lamports,
        data: account_data,
        owner: Pubkey(owner),
        executable,
        rent_epoch,
    })
}

/// Change set for a single transaction
#[derive(Clone, Debug, Default)]
pub struct StateChangeSet {
    /// Account updates (pubkey -> new account state)
    pub updates: HashMap<Pubkey, Account>,
    /// Account deletions
    pub deletions: Vec<Pubkey>,
}

impl StateChangeSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, pubkey: Pubkey, account: Account) {
        self.updates.insert(pubkey, account);
    }

    pub fn delete(&mut self, pubkey: Pubkey) {
        self.updates.remove(&pubkey);
        self.deletions.push(pubkey);
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty() && self.deletions.is_empty()
    }

    pub fn merge(&mut self, other: StateChangeSet) {
        for (pubkey, account) in other.updates {
            self.updates.insert(pubkey, account);
        }
        for pubkey in other.deletions {
            self.updates.remove(&pubkey);
            self.deletions.push(pubkey);
        }
    }
}

/// Block-level state batch for QMDB
///
/// Collects all state changes for a block before committing.
/// This matches QMDB's Task model.
#[derive(Clone, Debug)]
pub struct BlockStateBatch {
    pub height: u64,
    pub changes: Vec<StateChangeSet>,
}

impl BlockStateBatch {
    pub fn new(height: u64) -> Self {
        Self {
            height,
            changes: Vec::new(),
        }
    }

    pub fn add_changeset(&mut self, changeset: StateChangeSet) {
        self.changes.push(changeset);
    }

    /// Flatten all changesets into a single merged changeset
    pub fn flatten(&self) -> StateChangeSet {
        let mut merged = StateChangeSet::new();
        for cs in &self.changes {
            merged.merge(cs.clone());
        }
        merged
    }
}

/// In-memory QMDB state storage (for testing without full QMDB)
///
/// This provides QMDB-like semantics with in-memory storage.
/// Used when QMDB feature is disabled or for unit tests.
/// Uses DashMap with FxHasher for lock-free concurrent reads and writes.
#[derive(Clone)]
pub struct InMemoryQMDBState {
    /// Current accounts state - FastDashMap (FxHasher) for lock-free concurrent access
    accounts: Arc<FastDashMap<Pubkey, Account>>,
    /// Pending block batch
    pending_batch: Arc<Mutex<Option<BlockStateBatch>>>,
    /// Current block height
    current_height: Arc<RwLock<u64>>,
    /// Merkle root (cached)
    merkle_root: Arc<RwLock<[u8; 32]>>,
}

impl InMemoryQMDBState {
    pub fn new() -> Self {
        Self {
            accounts: Arc::new(DashMap::with_hasher(FxBuildHasher)),
            pending_batch: Arc::new(Mutex::new(None)),
            current_height: Arc::new(RwLock::new(0)),
            merkle_root: Arc::new(RwLock::new([0u8; 32])),
        }
    }

    pub fn with_accounts(accounts: Vec<(Pubkey, Account)>) -> Self {
        let state = Self::new();
        for (pubkey, account) in accounts {
            state.accounts.insert(pubkey, account);
        }
        state
    }

    /// Start a new block batch
    pub fn begin_block(&self, height: u64) -> RuntimeResult<()> {
        let mut pending = self.pending_batch.lock();
        if pending.is_some() {
            return Err(RuntimeError::State("Block already in progress".to_string()));
        }
        *pending = Some(BlockStateBatch::new(height));
        Ok(())
    }

    /// Add a transaction's changes to the current block
    pub fn add_tx_changes(&self, changeset: StateChangeSet) -> RuntimeResult<()> {
        let mut pending = self.pending_batch.lock();
        match pending.as_mut() {
            Some(batch) => {
                batch.add_changeset(changeset);
                Ok(())
            }
            None => Err(RuntimeError::State("No block in progress".to_string())),
        }
    }

    /// Add a pre-merged changeset to the current block (for parallel merge optimization)
    pub fn add_merged_changes(&self, merged: StateChangeSet) -> RuntimeResult<()> {
        let mut pending = self.pending_batch.lock();
        match pending.as_mut() {
            Some(batch) => {
                batch.add_changeset(merged);
                Ok(())
            }
            None => Err(RuntimeError::State("No block in progress".to_string())),
        }
    }

    /// Commit the current block
    pub fn commit_block(&self) -> RuntimeResult<[u8; 32]> {
        let batch = {
            let mut pending = self.pending_batch.lock();
            pending.take()
        };

        let batch = match batch {
            Some(b) => b,
            None => return Err(RuntimeError::State("No block to commit".to_string())),
        };

        // Apply all changes - DashMap allows concurrent writes
        let merged = batch.flatten();
        let _num_changes = merged.updates.len();

        // Parallel deletions
        use rayon::prelude::*;
        merged.deletions.par_iter().for_each(|pubkey| {
            self.accounts.remove(pubkey);
        });

        // Parallel inserts
        merged.updates.par_iter().for_each(|(pubkey, account)| {
            self.accounts.insert(*pubkey, account.clone());
        });

        // Update height
        {
            let mut height = self.current_height.write()
                .map_err(|e| RuntimeError::State(e.to_string()))?;
            *height = batch.height;
        }

        // OPTIMIZATION: Compute merkle ONLY on changed accounts (not all)
        let root = self.compute_merkle_from_changes(&merged)?;
        {
            let mut merkle_root = self.merkle_root.write()
                .map_err(|e| RuntimeError::State(e.to_string()))?;
            *merkle_root = root;
        }

        Ok(root)
    }

    /// Compute incremental state commitment using parallel accumulator
    /// This is O(n/p) where n = changes, p = cores - much faster than merkle tree
    fn compute_merkle_from_changes(&self, changes: &StateChangeSet) -> RuntimeResult<[u8; 32]> {
        use crate::hiperf::sha256_pair;
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU64, Ordering};

        if changes.updates.is_empty() && changes.deletions.is_empty() {
            let root = self.merkle_root.read()
                .map_err(|e| RuntimeError::State(e.to_string()))?;
            return Ok(*root);
        }

        // Parallel XOR accumulator - O(n/p) instead of O(n log n)
        // Each thread computes local XOR, then combine atomically
        let acc: [AtomicU64; 4] = Default::default();

        changes.updates.par_iter().for_each(|(pubkey, account)| {
            let hash = sha256_pair(&pubkey.0, &serialize_account(account));

            // XOR into accumulator (split into 4 u64s for parallel access)
            for i in 0..4 {
                let chunk = u64::from_le_bytes(hash[i*8..(i+1)*8].try_into().unwrap());
                acc[i].fetch_xor(chunk, Ordering::Relaxed);
            }
        });

        // Combine with previous root
        let prev_root = self.merkle_root.read()
            .map_err(|e| RuntimeError::State(e.to_string()))?;

        let mut result = [0u8; 32];
        for i in 0..4 {
            let prev_chunk = u64::from_le_bytes(prev_root[i*8..(i+1)*8].try_into().unwrap());
            let new_chunk = acc[i].load(Ordering::Relaxed) ^ prev_chunk;
            result[i*8..(i+1)*8].copy_from_slice(&new_chunk.to_le_bytes());
        }

        Ok(result)
    }

    /// Abort the current block
    pub fn abort_block(&self) -> RuntimeResult<()> {
        let mut pending = self.pending_batch.lock();
        *pending = None;
        Ok(())
    }

    /// Get account (reads committed state) - lock-free with DashMap
    pub fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        Ok(self.accounts.get(pubkey).map(|r| r.value().clone()))
    }

    /// Get multiple accounts - parallel with DashMap
    pub fn get_accounts(&self, pubkeys: &[Pubkey]) -> RuntimeResult<Vec<Option<Account>>> {
        use rayon::prelude::*;
        Ok(pubkeys.par_iter()
            .map(|pk| self.accounts.get(pk).map(|r| r.value().clone()))
            .collect())
    }

    /// Check if account exists - lock-free with DashMap
    pub fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool> {
        Ok(self.accounts.contains_key(pubkey))
    }

    /// Get current merkle root
    pub fn merkle_root(&self) -> RuntimeResult<[u8; 32]> {
        let root = self.merkle_root.read()
            .map_err(|e| RuntimeError::State(e.to_string()))?;
        Ok(*root)
    }

    /// Get current block height
    pub fn current_height(&self) -> RuntimeResult<u64> {
        let height = self.current_height.read()
            .map_err(|e| RuntimeError::State(e.to_string()))?;
        Ok(*height)
    }

    /// Compute full merkle root over all accounts (used for initial state only)
    fn compute_merkle_root(&self) -> RuntimeResult<[u8; 32]> {
        use crate::hiperf::{parallel_merkle_root, sha256_pair};
        use rayon::prelude::*;

        if self.accounts.is_empty() {
            return Ok([0u8; 32]);
        }

        // Collect and sort entries from DashMap
        let mut sorted: Vec<_> = self.accounts.iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect();
        sorted.sort_by_key(|(pk, _)| pk.0);

        let leaves: Vec<[u8; 32]> = sorted
            .par_iter()
            .map(|(pubkey, account)| {
                sha256_pair(&pubkey.0, &serialize_account(account))
            })
            .collect();

        Ok(parallel_merkle_root(&leaves))
    }

    /// Direct set account (for testing/initialization) - lock-free with DashMap
    pub fn set_account(&self, pubkey: &Pubkey, account: &Account) -> RuntimeResult<()> {
        self.accounts.insert(*pubkey, account.clone());
        Ok(())
    }

    /// Direct delete account (for testing) - lock-free with DashMap
    pub fn delete_account(&self, pubkey: &Pubkey) -> RuntimeResult<()> {
        self.accounts.remove(pubkey);
        Ok(())
    }
}

impl Default for InMemoryQMDBState {
    fn default() -> Self {
        Self::new()
    }
}

/// QMDB state storage trait
///
/// Abstracts over in-memory and full QMDB implementations.
pub trait QMDBState: Send + Sync {
    fn begin_block(&self, height: u64) -> RuntimeResult<()>;
    fn add_tx_changes(&self, changeset: StateChangeSet) -> RuntimeResult<()>;
    fn add_merged_changes(&self, merged: StateChangeSet) -> RuntimeResult<()>;
    fn commit_block(&self) -> RuntimeResult<[u8; 32]>;
    fn abort_block(&self) -> RuntimeResult<()>;
    fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>>;
    fn get_accounts(&self, pubkeys: &[Pubkey]) -> RuntimeResult<Vec<Option<Account>>>;
    fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool>;
    fn merkle_root(&self) -> RuntimeResult<[u8; 32]>;
    fn current_height(&self) -> RuntimeResult<u64>;
}

impl QMDBState for InMemoryQMDBState {
    fn begin_block(&self, height: u64) -> RuntimeResult<()> {
        self.begin_block(height)
    }

    fn add_tx_changes(&self, changeset: StateChangeSet) -> RuntimeResult<()> {
        self.add_tx_changes(changeset)
    }

    fn add_merged_changes(&self, merged: StateChangeSet) -> RuntimeResult<()> {
        self.add_merged_changes(merged)
    }

    fn commit_block(&self) -> RuntimeResult<[u8; 32]> {
        self.commit_block()
    }

    fn abort_block(&self) -> RuntimeResult<()> {
        self.abort_block()
    }

    fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        self.get_account(pubkey)
    }

    fn get_accounts(&self, pubkeys: &[Pubkey]) -> RuntimeResult<Vec<Option<Account>>> {
        self.get_accounts(pubkeys)
    }

    fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool> {
        self.account_exists(pubkey)
    }

    fn merkle_root(&self) -> RuntimeResult<[u8; 32]> {
        self.merkle_root()
    }

    fn current_height(&self) -> RuntimeResult<u64> {
        self.current_height()
    }
}

// ============================================================================
// AccountsDB compatibility layer
// ============================================================================

use crate::accounts::AccountsDB;

impl AccountsDB for InMemoryQMDBState {
    fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        InMemoryQMDBState::get_account(self, pubkey)
    }

    fn set_account(&mut self, pubkey: &Pubkey, account: &Account) -> RuntimeResult<()> {
        InMemoryQMDBState::set_account(self, pubkey, account)
    }

    fn delete_account(&mut self, pubkey: &Pubkey) -> RuntimeResult<()> {
        InMemoryQMDBState::delete_account(self, pubkey)
    }

    fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool> {
        InMemoryQMDBState::account_exists(self, pubkey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_serialization() {
        let account = Account {
            lamports: 1_000_000,
            data: vec![1, 2, 3, 4, 5],
            owner: Pubkey([0xAB; 32]),
            executable: true,
            rent_epoch: 123,
        };

        let serialized = serialize_account(&account);
        let deserialized = deserialize_account(&serialized).unwrap();

        assert_eq!(account.lamports, deserialized.lamports);
        assert_eq!(account.data, deserialized.data);
        assert_eq!(account.owner.0, deserialized.owner.0);
        assert_eq!(account.executable, deserialized.executable);
        assert_eq!(account.rent_epoch, deserialized.rent_epoch);
    }

    #[test]
    fn test_block_batch() {
        let state = InMemoryQMDBState::new();

        // Set initial state
        let pk1 = Pubkey([1; 32]);
        let pk2 = Pubkey([2; 32]);
        state.set_account(&pk1, &Account {
            lamports: 100,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();

        // Begin block
        state.begin_block(1).unwrap();

        // Add changes
        let mut cs = StateChangeSet::new();
        cs.update(pk1, Account {
            lamports: 50,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        cs.update(pk2, Account {
            lamports: 50,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        state.add_tx_changes(cs).unwrap();

        // Commit
        let root = state.commit_block().unwrap();
        assert_ne!(root, [0u8; 32]);

        // Verify state
        let acc1 = state.get_account(&pk1).unwrap().unwrap();
        assert_eq!(acc1.lamports, 50);

        let acc2 = state.get_account(&pk2).unwrap().unwrap();
        assert_eq!(acc2.lamports, 50);
    }

    #[test]
    fn test_abort_block() {
        let state = InMemoryQMDBState::new();

        let pk = Pubkey([1; 32]);
        state.set_account(&pk, &Account {
            lamports: 100,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();

        // Begin block
        state.begin_block(1).unwrap();

        // Add changes
        let mut cs = StateChangeSet::new();
        cs.update(pk, Account {
            lamports: 0,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        state.add_tx_changes(cs).unwrap();

        // Abort
        state.abort_block().unwrap();

        // State should be unchanged
        let acc = state.get_account(&pk).unwrap().unwrap();
        assert_eq!(acc.lamports, 100);
    }
}
