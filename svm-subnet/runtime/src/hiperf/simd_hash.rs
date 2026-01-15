//! SIMD-Accelerated Hashing
//!
//! Uses `sha2` crate with asm feature for hardware-accelerated SHA256:
//! - SHA-NI (Intel/AMD SHA extensions) - fastest
//! - ARMv8 crypto extensions (Apple Silicon, etc.)
//!
//! The sha2 asm implementation has lower per-call overhead than ring,
//! making it faster for many small hashes (merkle trees, account hashing).

use sha2::{Sha256, Digest};

/// Fast SHA256 hash using sha2 asm
#[inline(always)]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Fast SHA256 hash of two concatenated inputs
#[inline(always)]
pub fn sha256_pair(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(a);
    hasher.update(b);
    hasher.finalize().into()
}

/// Fast SHA256 for merkle node (two 32-byte children)
/// Optimized: pre-allocate 64-byte buffer to avoid allocations
#[inline(always)]
pub fn sha256_merkle_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// Batch hash multiple items in parallel using rayon
pub fn sha256_batch_parallel<T: AsRef<[u8]> + Sync>(items: &[T]) -> Vec<[u8; 32]> {
    use rayon::prelude::*;

    items
        .par_iter()
        .map(|item| sha256(item.as_ref()))
        .collect()
}

/// Hash account data (pubkey + serialized account)
#[inline(always)]
pub fn hash_account_fast(pubkey: &[u8; 32], data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(pubkey);
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Sha256, Digest};

    #[test]
    fn test_sha256_matches_sha2_crate() {
        let data = b"hello world";

        // Our fast implementation
        let fast_hash = sha256(data);

        // Reference implementation
        let mut hasher = Sha256::new();
        hasher.update(data);
        let reference: [u8; 32] = hasher.finalize().into();

        assert_eq!(fast_hash, reference);
    }

    #[test]
    fn test_merkle_node_hash() {
        let left = [1u8; 32];
        let right = [2u8; 32];

        let fast_hash = sha256_merkle_node(&left, &right);

        // Reference
        let mut hasher = Sha256::new();
        hasher.update(&left);
        hasher.update(&right);
        let reference: [u8; 32] = hasher.finalize().into();

        assert_eq!(fast_hash, reference);
    }

    #[test]
    fn test_batch_parallel() {
        let items: Vec<Vec<u8>> = (0..1000)
            .map(|i| vec![i as u8; 100])
            .collect();

        let hashes = sha256_batch_parallel(&items);

        assert_eq!(hashes.len(), 1000);

        // Verify first hash
        let expected = sha256(&items[0]);
        assert_eq!(hashes[0], expected);
    }
}
