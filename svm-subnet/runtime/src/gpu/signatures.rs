//! GPU-Accelerated Ed25519 Signature Verification
//!
//! Provides high-throughput signature verification using:
//! - CPU batch verification with ed25519-dalek (baseline)
//! - GPU-accelerated verification for massive batches (CUDA)
//!
//! Performance targets:
//! - CPU batch: ~1.7M verifications/sec (128 cores)
//! - GPU batch: ~10M+ verifications/sec (H100)
//!
//! The GPU implementation uses 5x51-bit limb representation for field elements
//! and parallel Edwards curve arithmetic for maximum throughput.

use ed25519_dalek::{Signature, VerifyingKey, Verifier};
use rayon::prelude::*;
use sha2::{Sha512, Digest};
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cuda")]
use crate::hiperf::GpuEd25519Verifier;

/// Result of batch signature verification
#[derive(Debug, Clone)]
pub struct BatchVerifyResult {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub elapsed_ms: f64,
    pub verifications_per_sec: f64,
}

/// Signature verification request
#[derive(Clone)]
pub struct SigVerifyRequest {
    /// 32-byte public key
    pub pubkey: [u8; 32],
    /// 64-byte signature
    pub signature: [u8; 64],
    /// Message to verify (transaction message bytes)
    pub message: Vec<u8>,
}

/// CPU-based batch signature verification using ed25519-dalek
/// Uses parallel verification across all available cores
pub fn batch_verify_cpu(requests: &[SigVerifyRequest]) -> BatchVerifyResult {
    let start = std::time::Instant::now();
    let valid_count = AtomicUsize::new(0);
    let invalid_count = AtomicUsize::new(0);

    requests.par_iter().for_each(|req| {
        let is_valid = verify_single(&req.pubkey, &req.signature, &req.message);
        if is_valid {
            valid_count.fetch_add(1, Ordering::Relaxed);
        } else {
            invalid_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let elapsed = start.elapsed();
    let valid = valid_count.load(Ordering::Relaxed);
    let invalid = invalid_count.load(Ordering::Relaxed);
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let verifications_per_sec = requests.len() as f64 / elapsed.as_secs_f64();

    BatchVerifyResult {
        total: requests.len(),
        valid,
        invalid,
        elapsed_ms,
        verifications_per_sec,
    }
}

/// Verify a single ed25519 signature
#[inline]
pub fn verify_single(pubkey: &[u8; 32], signature: &[u8; 64], message: &[u8]) -> bool {
    let Ok(verifying_key) = VerifyingKey::from_bytes(pubkey) else {
        return false;
    };

    let sig = Signature::from_bytes(signature);

    verifying_key.verify(message, &sig).is_ok()
}

/// Pre-hash message using SHA512 (ed25519 uses SHA512 internally)
/// This can be GPU-accelerated for massive batches
#[inline]
pub fn prehash_message(message: &[u8]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(message);
    let result = hasher.finalize();
    let mut hash = [0u8; 64];
    hash.copy_from_slice(&result);
    hash
}

/// Batch prehash messages in parallel (CPU)
/// Returns vector of (pubkey, signature, message_hash) for GPU verification
pub fn batch_prehash_cpu(requests: &[SigVerifyRequest]) -> Vec<([u8; 32], [u8; 64], [u8; 64])> {
    requests
        .par_iter()
        .map(|req| {
            let hash = prehash_message(&req.message);
            (req.pubkey, req.signature, hash)
        })
        .collect()
}

/// GPU batch verification using CUDA kernels
/// Uses the full Ed25519 implementation with Edwards curve arithmetic
#[cfg(feature = "cuda")]
pub fn batch_verify_gpu(requests: &[SigVerifyRequest]) -> BatchVerifyResult {
    let start = std::time::Instant::now();

    if requests.is_empty() {
        return BatchVerifyResult {
            total: 0,
            valid: 0,
            invalid: 0,
            elapsed_ms: 0.0,
            verifications_per_sec: 0.0,
        };
    }

    // Initialize GPU verifier
    let verifier = match GpuEd25519Verifier::new(0) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("GPU init failed: {}, falling back to CPU", e);
            return batch_verify_cpu(requests);
        }
    };

    // Prepare data for GPU: need prehashed messages
    let messages: Vec<[u8; 32]> = requests
        .par_iter()
        .map(|req| {
            // Hash the message to 32 bytes (SHA256 for GPU kernel)
            use sha2::Sha256;
            let mut hasher = Sha256::new();
            hasher.update(&req.message);
            let result = hasher.finalize();
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&result);
            hash
        })
        .collect();

    let signatures: Vec<[u8; 64]> = requests.iter().map(|r| r.signature).collect();
    let pubkeys: Vec<[u8; 32]> = requests.iter().map(|r| r.pubkey).collect();

    // Run GPU verification
    let results = match verifier.batch_verify(&messages, &signatures, &pubkeys) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("GPU verification failed: {}, falling back to CPU", e);
            return batch_verify_cpu(requests);
        }
    };

    let valid = results.iter().filter(|&&v| v).count();
    let invalid = results.len() - valid;

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let verifications_per_sec = requests.len() as f64 / elapsed.as_secs_f64();

    BatchVerifyResult {
        total: requests.len(),
        valid,
        invalid,
        elapsed_ms,
        verifications_per_sec,
    }
}

/// Non-CUDA fallback
#[cfg(not(feature = "cuda"))]
pub fn batch_verify_gpu(requests: &[SigVerifyRequest]) -> BatchVerifyResult {
    // Fall back to CPU batch verification
    batch_verify_cpu(requests)
}

/// Smart batch verification - chooses GPU or CPU based on batch size
/// GPU has overhead that only pays off for large batches (>10K)
pub fn batch_verify_auto(requests: &[SigVerifyRequest]) -> BatchVerifyResult {
    const GPU_THRESHOLD: usize = 10_000;

    #[cfg(feature = "cuda")]
    if requests.len() >= GPU_THRESHOLD {
        return batch_verify_gpu(requests);
    }

    batch_verify_cpu(requests)
}

/// Multi-GPU batch verification for maximum throughput
/// Distributes work across all available GPUs
#[cfg(feature = "cuda")]
pub fn batch_verify_multi_gpu(requests: &[SigVerifyRequest], num_gpus: usize) -> BatchVerifyResult {
    use std::sync::Arc;
    use std::thread;

    let start = std::time::Instant::now();

    if requests.is_empty() || num_gpus == 0 {
        return BatchVerifyResult {
            total: 0,
            valid: 0,
            invalid: 0,
            elapsed_ms: 0.0,
            verifications_per_sec: 0.0,
        };
    }

    let chunk_size = (requests.len() + num_gpus - 1) / num_gpus;
    let requests = Arc::new(requests.to_vec());

    let handles: Vec<_> = (0..num_gpus)
        .map(|gpu_id| {
            let requests = Arc::clone(&requests);
            thread::spawn(move || {
                let start_idx = gpu_id * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, requests.len());

                if start_idx >= requests.len() {
                    return BatchVerifyResult {
                        total: 0,
                        valid: 0,
                        invalid: 0,
                        elapsed_ms: 0.0,
                        verifications_per_sec: 0.0,
                    };
                }

                let chunk = &requests[start_idx..end_idx];

                // Initialize GPU for this thread
                let verifier = match GpuEd25519Verifier::new(gpu_id) {
                    Ok(v) => v,
                    Err(_) => return batch_verify_cpu(chunk),
                };

                // Prepare data
                let messages: Vec<[u8; 32]> = chunk
                    .iter()
                    .map(|req| {
                        use sha2::Sha256;
                        let mut hasher = Sha256::new();
                        hasher.update(&req.message);
                        let result = hasher.finalize();
                        let mut hash = [0u8; 32];
                        hash.copy_from_slice(&result);
                        hash
                    })
                    .collect();

                let signatures: Vec<[u8; 64]> = chunk.iter().map(|r| r.signature).collect();
                let pubkeys: Vec<[u8; 32]> = chunk.iter().map(|r| r.pubkey).collect();

                match verifier.batch_verify(&messages, &signatures, &pubkeys) {
                    Ok(results) => {
                        let valid = results.iter().filter(|&&v| v).count();
                        BatchVerifyResult {
                            total: chunk.len(),
                            valid,
                            invalid: chunk.len() - valid,
                            elapsed_ms: 0.0,
                            verifications_per_sec: 0.0,
                        }
                    }
                    Err(_) => batch_verify_cpu(chunk),
                }
            })
        })
        .collect();

    // Aggregate results
    let mut total = 0;
    let mut valid = 0;
    let mut invalid = 0;

    for handle in handles {
        if let Ok(result) = handle.join() {
            total += result.total;
            valid += result.valid;
            invalid += result.invalid;
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let verifications_per_sec = total as f64 / elapsed.as_secs_f64();

    BatchVerifyResult {
        total,
        valid,
        invalid,
        elapsed_ms,
        verifications_per_sec,
    }
}

#[cfg(not(feature = "cuda"))]
pub fn batch_verify_multi_gpu(requests: &[SigVerifyRequest], _num_gpus: usize) -> BatchVerifyResult {
    batch_verify_cpu(requests)
}

/// Generate a valid signature for testing
pub fn generate_test_signature(seed: u64) -> SigVerifyRequest {
    use ed25519_dalek::SigningKey;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    // Deterministic keypair from seed
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let signing_key = SigningKey::generate(&mut rng);
    let verifying_key = signing_key.verifying_key();

    // Create message
    let message = format!("test message {}", seed).into_bytes();

    // Sign
    use ed25519_dalek::Signer;
    let signature = signing_key.sign(&message);

    SigVerifyRequest {
        pubkey: verifying_key.to_bytes(),
        signature: signature.to_bytes(),
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_verification() {
        let req = generate_test_signature(42);
        assert!(verify_single(&req.pubkey, &req.signature, &req.message));
    }

    #[test]
    fn test_batch_verification() {
        let requests: Vec<_> = (0..1000).map(generate_test_signature).collect();
        let result = batch_verify_cpu(&requests);
        assert_eq!(result.valid, 1000);
        assert_eq!(result.invalid, 0);
    }

    #[test]
    fn test_invalid_signature() {
        let mut req = generate_test_signature(42);
        req.signature[0] ^= 0xFF; // Corrupt signature
        assert!(!verify_single(&req.pubkey, &req.signature, &req.message));
    }
}
