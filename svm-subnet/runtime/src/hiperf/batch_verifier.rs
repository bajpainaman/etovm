//! Parallel Batch Signature Verification
//!
//! Uses Ed25519 batch verification for ~8x speedup over individual verification.
//! Processes signatures in parallel batches across all CPU cores.

use crate::types::Transaction;
use ed25519_dalek::{Signature, VerifyingKey};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Batch size for signature verification (tuned for L1 cache)
const BATCH_SIZE: usize = 64;

/// Result of batch verification
#[derive(Debug)]
pub struct VerificationResult {
    /// Bitmap of valid transactions (true = valid)
    pub valid: Vec<bool>,
    /// Number of valid signatures
    pub num_valid: usize,
    /// Number of invalid signatures
    pub num_invalid: usize,
    /// Verification time in microseconds
    pub time_us: u64,
}

/// High-performance batch signature verifier
pub struct BatchVerifier {
    /// Number of threads to use (0 = auto)
    num_threads: usize,
    /// Counter for verified signatures
    verified_count: AtomicUsize,
}

impl BatchVerifier {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            verified_count: AtomicUsize::new(0),
        }
    }

    /// Verify all signatures in a batch of transactions
    ///
    /// Uses parallel batch verification for maximum throughput.
    /// Returns a bitmap indicating which transactions have valid signatures.
    pub fn verify_batch(&self, transactions: &[Transaction]) -> VerificationResult {
        let start = std::time::Instant::now();

        if transactions.is_empty() {
            return VerificationResult {
                valid: vec![],
                num_valid: 0,
                num_invalid: 0,
                time_us: 0,
            };
        }

        // Parallel verification with chunking for cache efficiency
        let results: Vec<bool> = transactions
            .par_chunks(BATCH_SIZE)
            .flat_map(|chunk| {
                self.verify_chunk(chunk)
            })
            .collect();

        let num_valid = results.iter().filter(|&&v| v).count();
        let num_invalid = results.len() - num_valid;

        self.verified_count.fetch_add(results.len(), Ordering::Relaxed);

        VerificationResult {
            valid: results,
            num_valid,
            num_invalid,
            time_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Verify a chunk of transactions using batch verification
    fn verify_chunk(&self, transactions: &[Transaction]) -> Vec<bool> {
        // Collect all signature data for batch verification
        let mut messages: SmallVec<[[u8; 32]; BATCH_SIZE]> = SmallVec::new();
        let mut signatures: SmallVec<[Option<Signature>; BATCH_SIZE]> = SmallVec::new();
        let mut pubkeys: SmallVec<[Option<VerifyingKey>; BATCH_SIZE]> = SmallVec::new();

        for tx in transactions {
            // Compute message hash (what was signed)
            let msg_hash = self.compute_message_hash(tx);
            messages.push(msg_hash);

            // Parse signature
            if tx.signatures.is_empty() {
                signatures.push(None);
                pubkeys.push(None);
                continue;
            }

            // ed25519-dalek 2.x: from_bytes returns Signature directly, use try_from for fallible
            let sig = Signature::try_from(&tx.signatures[0][..]).ok();
            signatures.push(sig);

            // Get first signer's pubkey
            if tx.message.account_keys.is_empty() {
                pubkeys.push(None);
                continue;
            }

            let pk = match VerifyingKey::from_bytes(&tx.message.account_keys[0].0) {
                Ok(k) => Some(k),
                Err(_) => None,
            };
            pubkeys.push(pk);
        }

        // Try batch verification first (most efficient)
        // Fall back to individual verification if batch fails
        self.batch_verify_or_individual(&messages, &signatures, &pubkeys)
    }

    /// Attempt batch verification, fall back to individual if needed
    fn batch_verify_or_individual(
        &self,
        messages: &[[u8; 32]],
        signatures: &[Option<Signature>],
        pubkeys: &[Option<VerifyingKey>],
    ) -> Vec<bool> {
        // Collect valid items for batch verification
        let mut valid_indices: SmallVec<[usize; BATCH_SIZE]> = SmallVec::new();
        let mut batch_messages: SmallVec<[&[u8]; BATCH_SIZE]> = SmallVec::new();
        let mut batch_signatures: SmallVec<[Signature; BATCH_SIZE]> = SmallVec::new();
        let mut batch_pubkeys: SmallVec<[VerifyingKey; BATCH_SIZE]> = SmallVec::new();

        let mut results = vec![false; messages.len()];

        for i in 0..messages.len() {
            if let (Some(sig), Some(pk)) = (&signatures[i], &pubkeys[i]) {
                valid_indices.push(i);
                batch_messages.push(&messages[i][..]);
                batch_signatures.push(*sig);
                batch_pubkeys.push(*pk);
            }
        }

        if batch_signatures.is_empty() {
            return results;
        }

        // Try batch verification
        let batch_result = ed25519_dalek::verify_batch(
            &batch_messages,
            &batch_signatures,
            &batch_pubkeys,
        );

        match batch_result {
            Ok(()) => {
                // All valid - mark all as valid
                for &idx in &valid_indices {
                    results[idx] = true;
                }
            }
            Err(_) => {
                // Batch failed - verify individually to find which ones are valid
                for (i, &idx) in valid_indices.iter().enumerate() {
                    let pk = &batch_pubkeys[i];
                    let sig = &batch_signatures[i];
                    let msg = &messages[idx];

                    if pk.verify_strict(msg, sig).is_ok() {
                        results[idx] = true;
                    }
                }
            }
        }

        results
    }

    /// Compute the message hash (what the signature signs)
    #[inline]
    fn compute_message_hash(&self, tx: &Transaction) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash the serialized message
        // In production, this should match Solana's message serialization exactly
        hasher.update(&[tx.message.header.num_required_signatures]);
        hasher.update(&[tx.message.header.num_readonly_signed_accounts]);
        hasher.update(&[tx.message.header.num_readonly_unsigned_accounts]);

        // Account keys
        for key in &tx.message.account_keys {
            hasher.update(&key.0);
        }

        // Recent blockhash
        hasher.update(&tx.message.recent_blockhash);

        // Instructions
        for ix in &tx.message.instructions {
            hasher.update(&[ix.program_id_index]);
            hasher.update(&[ix.accounts.len() as u8]);
            hasher.update(&ix.accounts);
            hasher.update(&(ix.data.len() as u16).to_le_bytes());
            hasher.update(&ix.data);
        }

        hasher.finalize().into()
    }

    /// Get total signatures verified
    pub fn total_verified(&self) -> usize {
        self.verified_count.load(Ordering::Relaxed)
    }
}

/// Verify signatures with skip option (for pre-verified transactions)
pub struct SkipVerifier;

impl SkipVerifier {
    /// Create verification result marking all as valid (for testing/benchmarks)
    pub fn skip_all(count: usize) -> VerificationResult {
        VerificationResult {
            valid: vec![true; count],
            num_valid: count,
            num_invalid: 0,
            time_us: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageHeader, CompiledInstruction, Pubkey};
    use ed25519_dalek::{SigningKey, Signer};
    use rand::rngs::OsRng;

    fn make_signed_tx(signer: &SigningKey) -> Transaction {
        let pubkey = Pubkey(signer.verifying_key().to_bytes());

        let message = Message {
            header: MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: 0,
            },
            account_keys: vec![pubkey],
            recent_blockhash: [0u8; 32],
            instructions: vec![CompiledInstruction {
                program_id_index: 0,
                accounts: vec![],
                data: vec![1, 2, 3],
            }],
        };

        // Create a verifier to compute the message hash
        let verifier = BatchVerifier::new(0);
        let mut tx = Transaction {
            signatures: vec![[0u8; 64]],
            message,
        };

        let msg_hash = verifier.compute_message_hash(&tx);
        let signature = signer.sign(&msg_hash);
        tx.signatures[0] = signature.to_bytes();

        tx
    }

    #[test]
    fn test_batch_verification() {
        let verifier = BatchVerifier::new(0);

        // Generate signed transactions
        let mut transactions = Vec::new();
        for _ in 0..100 {
            let signer = SigningKey::generate(&mut OsRng);
            transactions.push(make_signed_tx(&signer));
        }

        let result = verifier.verify_batch(&transactions);

        assert_eq!(result.num_valid, 100);
        assert_eq!(result.num_invalid, 0);
    }

    #[test]
    fn test_invalid_signature_detection() {
        let verifier = BatchVerifier::new(0);

        let signer = SigningKey::generate(&mut OsRng);
        let mut tx = make_signed_tx(&signer);

        // Corrupt the signature
        tx.signatures[0][0] ^= 0xFF;

        let result = verifier.verify_batch(&[tx]);

        assert_eq!(result.num_valid, 0);
        assert_eq!(result.num_invalid, 1);
    }
}
