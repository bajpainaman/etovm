//! Parallel Merkle Tree Computation
//!
//! Computes merkle roots using parallel hashing for maximum throughput.
//! Uses a divide-and-conquer approach with rayon for work-stealing parallelism.
//! SIMD-accelerated SHA256 via sha2 asm (SHA-NI/ARMv8 crypto).

use super::simd_hash::{sha256_merkle_node, sha256_pair};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Threshold for switching to sequential hashing
/// Lowered from 1024 to 64 for better parallelism on mid-range workloads.
/// Rayon's work-stealing handles small batches efficiently.
const PARALLEL_THRESHOLD: usize = 64;

/// Compute merkle root from leaf data in parallel
///
/// Uses an iterative parallel approach:
/// 1. Compute each tree level in parallel using par_chunks
/// 2. Iterate until we reach the root
/// 3. Falls back to sequential for small inputs (< threshold)
pub fn parallel_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }

    if leaves.len() == 1 {
        return leaves[0];
    }

    // Pad to power of 2 for balanced tree
    let padded_len = leaves.len().next_power_of_two();
    let mut current: Vec<[u8; 32]> = Vec::with_capacity(padded_len);
    current.extend_from_slice(leaves);
    current.resize(padded_len, [0u8; 32]);

    // Iterative parallel tree construction
    while current.len() > 1 {
        if current.len() <= PARALLEL_THRESHOLD {
            // Small enough: finish sequentially
            return compute_merkle_level_sequential(&current);
        }

        // Parallel level computation
        current = current
            .par_chunks(2)
            .map(|pair| sha256_merkle_node(&pair[0], &pair[1]))
            .collect();
    }

    current[0]
}

/// Sequential merkle level computation (for small inputs, SIMD-accelerated)
#[inline]
fn compute_merkle_level_sequential(nodes: &[[u8; 32]]) -> [u8; 32] {
    let mut current = nodes.to_vec();

    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len() / 2);

        for pair in current.chunks_exact(2) {
            next.push(sha256_merkle_node(&pair[0], &pair[1]));
        }

        current = next;
    }

    current[0]
}

/// High-performance merkle tree builder
///
/// Features:
/// - Parallel leaf hashing
/// - Parallel tree construction
/// - Incremental updates (for future optimization)
pub struct ParallelMerkleTree {
    /// Number of leaves hashed
    leaves_hashed: AtomicUsize,
}

impl ParallelMerkleTree {
    pub fn new() -> Self {
        Self {
            leaves_hashed: AtomicUsize::new(0),
        }
    }

    /// Hash account data to leaf nodes in parallel (SIMD-accelerated)
    pub fn hash_accounts_parallel<T: AsRef<[u8]> + Sync>(
        &self,
        accounts: &[(impl AsRef<[u8]> + Sync, T)],
    ) -> Vec<[u8; 32]> {
        let leaves: Vec<[u8; 32]> = accounts
            .par_iter()
            .map(|(key, data)| sha256_pair(key.as_ref(), data.as_ref()))
            .collect();

        self.leaves_hashed.fetch_add(leaves.len(), Ordering::Relaxed);
        leaves
    }

    /// Compute root from pre-hashed leaves
    pub fn compute_root(&self, leaves: &[[u8; 32]]) -> [u8; 32] {
        parallel_merkle_root(leaves)
    }

    /// Full pipeline: hash accounts and compute root
    pub fn compute_root_from_accounts<T: AsRef<[u8]> + Sync>(
        &self,
        accounts: &[(impl AsRef<[u8]> + Sync, T)],
    ) -> [u8; 32] {
        if accounts.is_empty() {
            return [0u8; 32];
        }

        let leaves = self.hash_accounts_parallel(accounts);
        self.compute_root(&leaves)
    }

    /// Get total leaves hashed
    pub fn total_hashed(&self) -> usize {
        self.leaves_hashed.load(Ordering::Relaxed)
    }
}

impl Default for ParallelMerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute merkle root with sorted key ordering (SIMD-accelerated)
///
/// Ensures deterministic results by sorting by key first.
pub fn merkle_root_sorted<K: AsRef<[u8]> + Ord + Sync, V: AsRef<[u8]> + Sync>(
    items: &mut [(K, V)],
) -> [u8; 32] {
    if items.is_empty() {
        return [0u8; 32];
    }

    // Sort by key for deterministic ordering
    items.sort_by(|a, b| a.0.cmp(&b.0));

    // Hash leaves in parallel (SIMD-accelerated)
    let leaves: Vec<[u8; 32]> = items
        .par_iter()
        .map(|(key, value)| sha256_pair(key.as_ref(), value.as_ref()))
        .collect();

    parallel_merkle_root(&leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_merkle() {
        let root = parallel_merkle_root(&[]);
        assert_eq!(root, [0u8; 32]);
    }

    #[test]
    fn test_single_leaf() {
        let leaf = [1u8; 32];
        let root = parallel_merkle_root(&[leaf]);
        assert_eq!(root, leaf);
    }

    #[test]
    fn test_two_leaves() {
        let leaf1 = [1u8; 32];
        let leaf2 = [2u8; 32];

        let root = parallel_merkle_root(&[leaf1, leaf2]);

        // Verify manually using SIMD hasher
        let expected = sha256_merkle_node(&leaf1, &leaf2);

        assert_eq!(root, expected);
    }

    #[test]
    fn test_power_of_two() {
        let leaves: Vec<[u8; 32]> = (0..8u8)
            .map(|i| {
                let mut l = [0u8; 32];
                l[0] = i;
                l
            })
            .collect();

        let root = parallel_merkle_root(&leaves);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_non_power_of_two() {
        let leaves: Vec<[u8; 32]> = (0..5u8)
            .map(|i| {
                let mut l = [0u8; 32];
                l[0] = i;
                l
            })
            .collect();

        let root = parallel_merkle_root(&leaves);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_large_tree() {
        let leaves: Vec<[u8; 32]> = (0..10000u32)
            .map(|i| {
                let mut l = [0u8; 32];
                l[0..4].copy_from_slice(&i.to_le_bytes());
                l
            })
            .collect();

        let root = parallel_merkle_root(&leaves);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_deterministic() {
        let leaves: Vec<[u8; 32]> = (0..100u8)
            .map(|i| {
                let mut l = [0u8; 32];
                l[0] = i;
                l
            })
            .collect();

        let root1 = parallel_merkle_root(&leaves);
        let root2 = parallel_merkle_root(&leaves);

        assert_eq!(root1, root2);
    }

    #[test]
    fn test_merkle_tree_builder() {
        let tree = ParallelMerkleTree::new();

        let accounts: Vec<([u8; 32], Vec<u8>)> = (0..1000u32)
            .map(|i| {
                let mut key = [0u8; 32];
                key[0..4].copy_from_slice(&i.to_le_bytes());
                (key, vec![i as u8; 100])
            })
            .collect();

        let root = tree.compute_root_from_accounts(&accounts);
        assert_ne!(root, [0u8; 32]);
        assert_eq!(tree.total_hashed(), 1000);
    }
}
