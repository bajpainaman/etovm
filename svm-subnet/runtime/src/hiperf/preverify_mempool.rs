//! Pre-Verification Mempool
//!
//! Verifies transaction signatures in background as they arrive.
//! Block execution can then skip verification entirely.
//!
//! Architecture:
//! ```text
//! Transactions → [Background Verify Pool] → Verified Queue → Block Builder
//! ```

use crate::types::{Transaction, Pubkey};
use crate::error::{RuntimeError, RuntimeResult};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Verification result for a transaction
#[derive(Clone, Debug)]
pub struct VerifiedTx {
    pub tx: Transaction,
    pub tx_hash: [u8; 32],
    pub valid: bool,
    pub verified_at_ns: u64,
}

/// Pre-verification mempool configuration
#[derive(Clone, Debug)]
pub struct MempoolConfig {
    /// Max transactions in mempool
    pub max_size: usize,
    /// Number of verification threads (0 = auto)
    pub verify_threads: usize,
    /// Channel buffer size
    pub channel_buffer: usize,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1_000_000,
            verify_threads: 0,
            channel_buffer: 10_000,
        }
    }
}

/// High-performance pre-verification mempool
pub struct PreVerifyMempool {
    /// Verified transactions ready for block building
    verified: DashMap<[u8; 32], VerifiedTx>,
    /// Pending verification queue
    pending_tx: Sender<Transaction>,
    /// Verification complete receiver
    verified_rx: Receiver<VerifiedTx>,
    /// Statistics
    stats: Arc<MempoolStats>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Config
    config: MempoolConfig,
}

/// Mempool statistics
#[derive(Default)]
pub struct MempoolStats {
    pub submitted: AtomicU64,
    pub verified_valid: AtomicU64,
    pub verified_invalid: AtomicU64,
    pub pending: AtomicU64,
}

impl PreVerifyMempool {
    pub fn new(config: MempoolConfig) -> Self {
        let (pending_tx, pending_rx) = bounded(config.channel_buffer);
        let (verified_tx, verified_rx) = bounded(config.channel_buffer);

        let stats = Arc::new(MempoolStats::default());
        let running = Arc::new(AtomicBool::new(true));

        // Spawn verification workers
        let num_threads = if config.verify_threads == 0 {
            rayon::current_num_threads() / 2  // Use half cores for verification
        } else {
            config.verify_threads
        };

        for _ in 0..num_threads {
            let rx = pending_rx.clone();
            let tx = verified_tx.clone();
            let stats = stats.clone();
            let running = running.clone();

            thread::spawn(move || {
                verification_worker(rx, tx, stats, running);
            });
        }

        Self {
            verified: DashMap::new(),
            pending_tx,
            verified_rx,
            stats,
            running,
            config,
        }
    }

    /// Submit transaction for background verification
    pub fn submit(&self, tx: Transaction) -> RuntimeResult<()> {
        if self.verified.len() >= self.config.max_size {
            return Err(RuntimeError::Mempool("Mempool full".to_string()));
        }

        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        self.stats.pending.fetch_add(1, Ordering::Relaxed);

        self.pending_tx.send(tx)
            .map_err(|_| RuntimeError::Mempool("Verification queue full".to_string()))?;

        Ok(())
    }

    /// Submit batch of transactions
    pub fn submit_batch(&self, txs: Vec<Transaction>) -> RuntimeResult<usize> {
        let mut submitted = 0;
        for tx in txs {
            if self.submit(tx).is_ok() {
                submitted += 1;
            }
        }
        Ok(submitted)
    }

    /// Drain verified transactions into the verified map
    pub fn drain_verified(&self) {
        while let Ok(verified) = self.verified_rx.try_recv() {
            if verified.valid {
                self.stats.verified_valid.fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats.verified_invalid.fetch_add(1, Ordering::Relaxed);
            }
            self.stats.pending.fetch_sub(1, Ordering::Relaxed);
            self.verified.insert(verified.tx_hash, verified);
        }
    }

    /// Get N verified valid transactions for block building
    pub fn get_verified_batch(&self, max_count: usize) -> Vec<VerifiedTx> {
        self.drain_verified();

        let mut batch = Vec::with_capacity(max_count);
        let mut to_remove = Vec::new();

        for entry in self.verified.iter() {
            if batch.len() >= max_count {
                break;
            }
            if entry.valid {
                batch.push(entry.value().clone());
                to_remove.push(*entry.key());
            }
        }

        // Remove from mempool
        for hash in to_remove {
            self.verified.remove(&hash);
        }

        batch
    }

    /// Get verified transactions by hash (for specific tx requests)
    pub fn get_by_hash(&self, hash: &[u8; 32]) -> Option<VerifiedTx> {
        self.drain_verified();
        self.verified.get(hash).map(|v| v.clone())
    }

    /// Current mempool size
    pub fn len(&self) -> usize {
        self.drain_verified();
        self.verified.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &MempoolStats {
        &self.stats
    }

    /// Shutdown mempool
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

impl Drop for PreVerifyMempool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Background verification worker
fn verification_worker(
    rx: Receiver<Transaction>,
    tx: Sender<VerifiedTx>,
    stats: Arc<MempoolStats>,
    running: Arc<AtomicBool>,
) {
    use ed25519_dalek::{VerifyingKey, Signature, Verifier};

    while running.load(Ordering::Relaxed) {
        // Try to receive with timeout
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(transaction) => {
                let start = std::time::Instant::now();

                // Compute transaction hash
                let tx_hash = compute_tx_hash(&transaction);

                // Verify signature
                let valid = verify_transaction(&transaction);

                let verified = VerifiedTx {
                    tx: transaction,
                    tx_hash,
                    valid,
                    verified_at_ns: start.elapsed().as_nanos() as u64,
                };

                let _ = tx.send(verified);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Compute transaction hash
fn compute_tx_hash(tx: &Transaction) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Hash signatures
    for sig in &tx.signatures {
        hasher.update(sig);
    }

    // Hash message
    hasher.update(&[tx.message.header.num_required_signatures]);
    hasher.update(&[tx.message.header.num_readonly_signed_accounts]);
    hasher.update(&[tx.message.header.num_readonly_unsigned_accounts]);

    for key in &tx.message.account_keys {
        hasher.update(&key.0);
    }

    hasher.update(&tx.message.recent_blockhash);

    hasher.finalize().into()
}

/// Verify a single transaction's signatures
fn verify_transaction(tx: &Transaction) -> bool {
    use ed25519_dalek::{VerifyingKey, Signature, Verifier};

    let num_signers = tx.message.header.num_required_signatures as usize;

    if tx.signatures.len() < num_signers {
        return false;
    }

    if tx.message.account_keys.len() < num_signers {
        return false;
    }

    // Compute message hash
    let msg_hash = compute_message_hash(&tx.message);

    // Verify each required signature
    for i in 0..num_signers {
        let pubkey_bytes = &tx.message.account_keys[i].0;
        let sig_bytes = &tx.signatures[i];

        let Ok(pubkey) = VerifyingKey::from_bytes(pubkey_bytes) else {
            return false;
        };

        let signature = Signature::from_bytes(sig_bytes);

        if pubkey.verify(&msg_hash, &signature).is_err() {
            return false;
        }
    }

    true
}

/// Compute message hash for verification
fn compute_message_hash(msg: &crate::types::Message) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Header
    hasher.update(&[msg.header.num_required_signatures]);
    hasher.update(&[msg.header.num_readonly_signed_accounts]);
    hasher.update(&[msg.header.num_readonly_unsigned_accounts]);

    // Account keys
    for key in &msg.account_keys {
        hasher.update(&key.0);
    }

    // Recent blockhash
    hasher.update(&msg.recent_blockhash);

    // Instructions
    for ix in &msg.instructions {
        hasher.update(&[ix.program_id_index]);
        hasher.update(&[ix.accounts.len() as u8]);
        hasher.update(&ix.accounts);
        hasher.update(&(ix.data.len() as u16).to_le_bytes());
        hasher.update(&ix.data);
    }

    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageHeader, CompiledInstruction};
    use ed25519_dalek::{SigningKey, Signer};

    fn make_signed_tx(seed: u64) -> Transaction {
        let mut seed_bytes = [0u8; 32];
        seed_bytes[0..8].copy_from_slice(&seed.to_le_bytes());
        let hash: [u8; 32] = Sha256::digest(&seed_bytes).into();
        let keypair = SigningKey::from_bytes(&hash);

        let from = keypair.verifying_key().to_bytes();
        let mut to = [0u8; 32];
        to[0..8].copy_from_slice(&(seed + 1).to_le_bytes());

        let message = Message {
            header: MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: 1,
            },
            account_keys: vec![Pubkey(from), Pubkey(to), Pubkey::system_program()],
            recent_blockhash: [0u8; 32],
            instructions: vec![CompiledInstruction {
                program_id_index: 2,
                accounts: vec![0, 1],
                data: vec![2, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
            }],
        };

        let msg_hash = compute_message_hash(&message);
        let signature = keypair.sign(&msg_hash);

        Transaction {
            signatures: vec![signature.to_bytes()],
            message,
        }
    }

    #[test]
    fn test_verification() {
        let tx = make_signed_tx(42);
        assert!(verify_transaction(&tx));
    }

    #[test]
    fn test_mempool_basic() {
        let config = MempoolConfig {
            verify_threads: 2,
            ..Default::default()
        };
        let mempool = PreVerifyMempool::new(config);

        // Submit transactions
        for i in 0..100 {
            let tx = make_signed_tx(i);
            mempool.submit(tx).unwrap();
        }

        // Wait for verification
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Get verified batch
        let batch = mempool.get_verified_batch(100);
        assert!(!batch.is_empty());

        // All should be valid
        for vtx in &batch {
            assert!(vtx.valid);
        }

        mempool.shutdown();
    }
}
