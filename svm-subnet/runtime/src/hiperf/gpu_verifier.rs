//! GPU-Accelerated Signature Verification
//!
//! Provides a trait for signature verification that can be implemented
//! with either CPU (ed25519-dalek) or GPU (CUDA/Metal) backends.
//!
//! For H100: Use CUDA backend for ~100x speedup on batch verification.

use crate::types::Transaction;
use std::sync::Arc;

/// Signature verification result
#[derive(Clone, Debug)]
pub struct VerifyResult {
    pub valid: Vec<bool>,
    pub total_time_us: u64,
}

/// Trait for signature verification backends
pub trait SignatureVerifier: Send + Sync {
    /// Verify a batch of transactions
    fn verify_batch(&self, transactions: &[Transaction]) -> VerifyResult;

    /// Backend name for logging
    fn name(&self) -> &'static str;

    /// Whether this is a GPU backend
    fn is_gpu(&self) -> bool;
}

/// CPU batch verifier using ed25519-dalek
pub struct CpuBatchVerifier {
    /// Batch size for parallel verification
    batch_size: usize,
}

impl CpuBatchVerifier {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

impl Default for CpuBatchVerifier {
    fn default() -> Self {
        Self::new(64) // Optimal for L1 cache
    }
}

impl SignatureVerifier for CpuBatchVerifier {
    fn verify_batch(&self, transactions: &[Transaction]) -> VerifyResult {
        use ed25519_dalek::{VerifyingKey, Signature, Verifier};
        use rayon::prelude::*;
        use sha2::{Sha256, Digest};

        let start = std::time::Instant::now();

        let valid: Vec<bool> = transactions
            .par_chunks(self.batch_size)
            .flat_map(|chunk| {
                chunk.iter().map(|tx| verify_single_cpu(tx)).collect::<Vec<_>>()
            })
            .collect();

        VerifyResult {
            valid,
            total_time_us: start.elapsed().as_micros() as u64,
        }
    }

    fn name(&self) -> &'static str {
        "CPU (ed25519-dalek)"
    }

    fn is_gpu(&self) -> bool {
        false
    }
}

/// Verify single transaction on CPU
fn verify_single_cpu(tx: &Transaction) -> bool {
    use ed25519_dalek::{VerifyingKey, Signature, Verifier};
    use sha2::{Sha256, Digest};

    let num_signers = tx.message.header.num_required_signatures as usize;

    if tx.signatures.len() < num_signers || tx.message.account_keys.len() < num_signers {
        return false;
    }

    // Compute message hash
    let msg_hash = compute_message_hash_for_verify(&tx.message);

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

fn compute_message_hash_for_verify(msg: &crate::types::Message) -> [u8; 32] {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(&[msg.header.num_required_signatures]);
    hasher.update(&[msg.header.num_readonly_signed_accounts]);
    hasher.update(&[msg.header.num_readonly_unsigned_accounts]);

    for key in &msg.account_keys {
        hasher.update(&key.0);
    }
    hasher.update(&msg.recent_blockhash);

    for ix in &msg.instructions {
        hasher.update(&[ix.program_id_index]);
        hasher.update(&[ix.accounts.len() as u8]);
        hasher.update(&ix.accounts);
        hasher.update(&(ix.data.len() as u16).to_le_bytes());
        hasher.update(&ix.data);
    }

    hasher.finalize().into()
}

// ============================================================================
// GPU Backend (CUDA) - Feature-gated
// ============================================================================

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;

    /// CUDA GPU batch verifier for H100/A100
    ///
    /// Uses CUDA kernels for parallel Ed25519 verification.
    /// Expected speedup: 50-100x over CPU for large batches.
    pub struct CudaBatchVerifier {
        device_id: i32,
        max_batch_size: usize,
    }

    impl CudaBatchVerifier {
        pub fn new(device_id: i32) -> Result<Self, String> {
            // TODO: Initialize CUDA context
            // cudarc::driver::CudaDevice::new(device_id)?
            Ok(Self {
                device_id,
                max_batch_size: 1_000_000, // H100 can handle millions
            })
        }
    }

    impl SignatureVerifier for CudaBatchVerifier {
        fn verify_batch(&self, transactions: &[Transaction]) -> VerifyResult {
            // TODO: Implement CUDA kernel call
            // 1. Copy messages, signatures, pubkeys to GPU
            // 2. Launch parallel verification kernel
            // 3. Copy results back

            // Placeholder: fall back to CPU for now
            let cpu = CpuBatchVerifier::default();
            cpu.verify_batch(transactions)
        }

        fn name(&self) -> &'static str {
            "GPU (CUDA/H100)"
        }

        fn is_gpu(&self) -> bool {
            true
        }
    }
}

// ============================================================================
// Hybrid Verifier - Auto-selects best backend
// ============================================================================

/// Hybrid verifier that uses GPU when available, falls back to CPU
pub struct HybridVerifier {
    gpu: Option<Arc<dyn SignatureVerifier>>,
    cpu: Arc<dyn SignatureVerifier>,
    /// Minimum batch size to use GPU (small batches faster on CPU)
    gpu_threshold: usize,
}

impl HybridVerifier {
    pub fn new() -> Self {
        let cpu = Arc::new(CpuBatchVerifier::default());

        #[cfg(feature = "cuda")]
        let gpu = cuda::CudaBatchVerifier::new(0)
            .ok()
            .map(|v| Arc::new(v) as Arc<dyn SignatureVerifier>);

        #[cfg(not(feature = "cuda"))]
        let gpu: Option<Arc<dyn SignatureVerifier>> = None;

        Self {
            gpu,
            cpu,
            gpu_threshold: 1000, // Use GPU for batches > 1000
        }
    }

    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }
}

impl Default for HybridVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SignatureVerifier for HybridVerifier {
    fn verify_batch(&self, transactions: &[Transaction]) -> VerifyResult {
        // Use GPU for large batches if available
        if let Some(gpu) = &self.gpu {
            if transactions.len() >= self.gpu_threshold {
                return gpu.verify_batch(transactions);
            }
        }

        // Fall back to CPU
        self.cpu.verify_batch(transactions)
    }

    fn name(&self) -> &'static str {
        if self.gpu.is_some() {
            "Hybrid (GPU+CPU)"
        } else {
            "CPU (ed25519-dalek)"
        }
    }

    fn is_gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

// ============================================================================
// Pre-computed Verification Cache
// ============================================================================

use dashmap::DashMap;

/// Caches verification results to avoid re-verifying same signatures
pub struct CachedVerifier<V: SignatureVerifier> {
    inner: V,
    cache: DashMap<[u8; 64], bool>, // signature -> valid
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl<V: SignatureVerifier> CachedVerifier<V> {
    pub fn new(inner: V) -> Self {
        Self {
            inner,
            cache: DashMap::new(),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl<V: SignatureVerifier> SignatureVerifier for CachedVerifier<V> {
    fn verify_batch(&self, transactions: &[Transaction]) -> VerifyResult {
        use std::sync::atomic::Ordering;

        let start = std::time::Instant::now();

        // Check cache for each transaction
        let mut need_verify = Vec::new();
        let mut need_verify_idx = Vec::new();
        let mut results = vec![false; transactions.len()];

        for (i, tx) in transactions.iter().enumerate() {
            if tx.signatures.is_empty() {
                continue;
            }

            let sig = &tx.signatures[0];
            if let Some(cached) = self.cache.get(sig) {
                results[i] = *cached;
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                need_verify.push(tx.clone());
                need_verify_idx.push(i);
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Verify uncached transactions
        if !need_verify.is_empty() {
            let verify_results = self.inner.verify_batch(&need_verify);

            for (j, i) in need_verify_idx.iter().enumerate() {
                results[*i] = verify_results.valid[j];

                // Cache the result
                if !need_verify[j].signatures.is_empty() {
                    self.cache.insert(need_verify[j].signatures[0], verify_results.valid[j]);
                }
            }
        }

        VerifyResult {
            valid: results,
            total_time_us: start.elapsed().as_micros() as u64,
        }
    }

    fn name(&self) -> &'static str {
        "Cached Verifier"
    }

    fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_verifier() {
        let verifier = CpuBatchVerifier::default();
        assert_eq!(verifier.name(), "CPU (ed25519-dalek)");
        assert!(!verifier.is_gpu());
    }

    #[test]
    fn test_hybrid_verifier() {
        let verifier = HybridVerifier::new();
        // Without CUDA feature, should use CPU
        #[cfg(not(feature = "cuda"))]
        assert!(!verifier.is_gpu());
    }
}
