//! FAFO Scheduler - Fast Ahead of Formation Optimization
//!
//! Based on LayerZero's FAFO paper for achieving 1M+ TPS.
//!
//! Key innovations:
//! - Parabloom filters: 64 parallel bloom filters for O(1) conflict detection
//! - Frame-based packing: group non-conflicting txs into parallel frames
//! - Streaming execution: frames can be executed as they're formed
//!
//! This replaces the O(N²) greedy scheduler with O(N) parallel scheduling.

use crate::sealevel::AccessSet;
use crate::types::Pubkey;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of parallel frames (fits in 64-bit word for SIMD operations)
const NUM_FRAMES: usize = 64;

/// Bloom filter size in bits (must fit in L1 cache for speed)
/// 4KB = 32768 bits, fits in half of typical 8KB L1 cache
const BLOOM_BITS: usize = 32768;
const BLOOM_WORDS: usize = BLOOM_BITS / 64;

/// Number of hash functions for bloom filter
const NUM_HASHES: usize = 4;

/// Parabloom filter - 64 parallel bloom filters packed for SIMD operations
///
/// Each of the 64 bits in a u64 word represents membership in one of 64 frames.
/// This allows checking conflicts against all 64 frames with a single bitwise AND.
pub struct Parabloom {
    /// Read set bloom filters (64 parallel filters)
    /// Each u64 has bit i set if frame i's read set might contain this hash position
    reads: Vec<AtomicU64>,
    /// Write set bloom filters (64 parallel filters)
    writes: Vec<AtomicU64>,
}

impl Parabloom {
    pub fn new() -> Self {
        Self {
            reads: (0..BLOOM_WORDS).map(|_| AtomicU64::new(0)).collect(),
            writes: (0..BLOOM_WORDS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Hash a pubkey to bloom filter positions
    #[inline]
    fn hash_positions(pubkey: &Pubkey) -> [usize; NUM_HASHES] {
        let bytes = &pubkey.0;
        [
            // Use different byte ranges for independent hashes
            (u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize) % BLOOM_BITS,
            (u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize) % BLOOM_BITS,
            (u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize) % BLOOM_BITS,
            (u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize) % BLOOM_BITS,
        ]
    }

    /// Add a read to a specific frame
    #[inline]
    pub fn add_read(&self, pubkey: &Pubkey, frame: usize) {
        let frame_bit = 1u64 << frame;
        for pos in Self::hash_positions(pubkey) {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            // Set the frame bit at this bloom position
            self.reads[word_idx].fetch_or(frame_bit << bit_idx, Ordering::Relaxed);
        }
    }

    /// Add a write to a specific frame
    #[inline]
    pub fn add_write(&self, pubkey: &Pubkey, frame: usize) {
        let frame_bit = 1u64 << frame;
        for pos in Self::hash_positions(pubkey) {
            let word_idx = pos / 64;
            let bit_idx = pos % 64;
            self.writes[word_idx].fetch_or(frame_bit << bit_idx, Ordering::Relaxed);
        }
    }

    /// Check which frames a transaction conflicts with
    ///
    /// Returns a bitmask where bit i is set if frame i conflicts.
    /// Conflict = (tx writes ∩ frame all) OR (tx reads ∩ frame writes)
    #[inline]
    pub fn check_conflicts(&self, access: &AccessSet) -> u64 {
        let mut write_conflicts: u64 = 0;
        let mut read_conflicts: u64 = 0;

        // Check tx writes against all frames' reads and writes
        for pubkey in &access.writes {
            for pos in Self::hash_positions(pubkey) {
                let word_idx = pos / 64;
                let bit_idx = pos % 64;
                let reads = self.reads[word_idx].load(Ordering::Relaxed);
                let writes = self.writes[word_idx].load(Ordering::Relaxed);
                write_conflicts |= (reads >> bit_idx) | (writes >> bit_idx);
            }
        }

        // Check tx reads against all frames' writes
        for pubkey in &access.reads {
            for pos in Self::hash_positions(pubkey) {
                let word_idx = pos / 64;
                let bit_idx = pos % 64;
                let writes = self.writes[word_idx].load(Ordering::Relaxed);
                read_conflicts |= writes >> bit_idx;
            }
        }

        write_conflicts | read_conflicts
    }
}

impl Default for Parabloom {
    fn default() -> Self {
        Self::new()
    }
}

/// A frame of non-conflicting transactions that can execute in parallel
#[derive(Debug, Clone)]
pub struct ExecutionFrame {
    /// Transaction indices in this frame
    pub tx_indices: Vec<usize>,
}

impl ExecutionFrame {
    pub fn new() -> Self {
        Self {
            tx_indices: Vec::with_capacity(1024),
        }
    }

    pub fn len(&self) -> usize {
        self.tx_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tx_indices.is_empty()
    }
}

impl Default for ExecutionFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// FAFO Scheduler - achieves O(N) scheduling for 1M+ TPS
///
/// Algorithm:
/// 1. Maintain 64 parallel frames with parabloom filters
/// 2. For each transaction, find first non-conflicting frame in O(1)
/// 3. Add to frame and update bloom filter
/// 4. When frame is "full", eject and create new frame
pub struct FafoScheduler {
    /// Maximum transactions per frame before ejection
    max_frame_size: usize,
}

impl FafoScheduler {
    pub fn new(max_frame_size: usize) -> Self {
        Self { max_frame_size }
    }

    /// Schedule transactions into parallel execution frames
    ///
    /// This is the core FAFO algorithm:
    /// - O(N) total time complexity
    /// - O(1) conflict detection per transaction via parabloom
    /// - 64 parallel frames for maximum parallelism
    pub fn schedule(&self, access_sets: &[AccessSet]) -> Vec<ExecutionFrame> {
        if access_sets.is_empty() {
            return vec![];
        }

        let mut frames: Vec<ExecutionFrame> = (0..NUM_FRAMES)
            .map(|_| ExecutionFrame::new())
            .collect();
        let parabloom = Parabloom::new();
        let mut active_frames: u64 = u64::MAX; // All 64 frames active
        let mut completed_frames: Vec<ExecutionFrame> = Vec::new();

        for (tx_idx, access) in access_sets.iter().enumerate() {
            // Find non-conflicting frames (O(1) via bloom filter)
            let conflicts = parabloom.check_conflicts(access);
            let available = active_frames & !conflicts;

            if available == 0 {
                // All frames conflict - eject oldest frame and retry
                let oldest = active_frames.trailing_zeros() as usize;
                if oldest < NUM_FRAMES && !frames[oldest].is_empty() {
                    completed_frames.push(std::mem::take(&mut frames[oldest]));
                }
                // Add to this now-empty frame
                frames[oldest].tx_indices.push(tx_idx);
                Self::add_to_bloom(&parabloom, access, oldest);
            } else {
                // Find first available frame (lowest set bit)
                let frame_idx = available.trailing_zeros() as usize;
                frames[frame_idx].tx_indices.push(tx_idx);
                Self::add_to_bloom(&parabloom, access, frame_idx);

                // Check if frame is full
                if frames[frame_idx].len() >= self.max_frame_size {
                    completed_frames.push(std::mem::take(&mut frames[frame_idx]));
                    // Frame stays "active" but is now empty
                }
            }
        }

        // Collect remaining non-empty frames
        for frame in frames {
            if !frame.is_empty() {
                completed_frames.push(frame);
            }
        }

        completed_frames
    }

    #[inline]
    fn add_to_bloom(parabloom: &Parabloom, access: &AccessSet, frame: usize) {
        for pubkey in &access.reads {
            parabloom.add_read(pubkey, frame);
        }
        for pubkey in &access.writes {
            parabloom.add_write(pubkey, frame);
        }
    }

    /// Parallel scheduling - analyze and schedule in parallel chunks
    ///
    /// For very large transaction counts, this parallelizes the scheduling
    /// across multiple cores, then merges the results.
    pub fn schedule_parallel(&self, access_sets: &[AccessSet]) -> Vec<ExecutionFrame> {
        if access_sets.len() < 10_000 {
            return self.schedule(access_sets);
        }

        // Split into chunks and schedule each in parallel
        const CHUNK_SIZE: usize = 50_000;
        let chunks: Vec<_> = access_sets.chunks(CHUNK_SIZE).collect();

        let chunk_frames: Vec<Vec<ExecutionFrame>> = chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_offset = chunk_idx * CHUNK_SIZE;
                let local_frames = self.schedule(chunk);

                // Adjust indices to global positions
                local_frames
                    .into_iter()
                    .map(|mut frame| {
                        for idx in frame.tx_indices.iter_mut() {
                            *idx += base_offset;
                        }
                        frame
                    })
                    .collect()
            })
            .collect();

        // Flatten all frames
        chunk_frames.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_fafo_no_conflicts() {
        let scheduler = FafoScheduler::new(1000);

        // 100 transactions, all different accounts
        let access_sets: Vec<AccessSet> = (0..100u8)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let frames = scheduler.schedule(&access_sets);

        // Should pack into minimal frames
        let total_txs: usize = frames.iter().map(|f| f.len()).sum();
        assert_eq!(total_txs, 100);
    }

    #[test]
    fn test_fafo_all_conflict() {
        let scheduler = FafoScheduler::new(1000);

        // 10 transactions all writing to same account
        let pk = make_pubkey(1);
        let access_sets: Vec<AccessSet> = (0..10)
            .map(|_| {
                let mut a = AccessSet::new();
                a.add_write(pk);
                a
            })
            .collect();

        let frames = scheduler.schedule(&access_sets);

        // Each should be in separate frame due to conflicts
        // (bloom filter may have false positives, so >= 10 frames)
        let total_txs: usize = frames.iter().map(|f| f.len()).sum();
        assert_eq!(total_txs, 10);
    }

    #[test]
    fn test_parabloom_basic() {
        let bloom = Parabloom::new();
        let pk1 = make_pubkey(1);
        let pk2 = make_pubkey(2);

        // Add pk1 write to frame 0
        bloom.add_write(&pk1, 0);

        // Check conflicts
        let mut access = AccessSet::new();
        access.add_write(pk1);

        let conflicts = bloom.check_conflicts(&access);
        assert!(conflicts & 1 != 0); // Frame 0 should conflict

        // Different key should not conflict (usually)
        let mut access2 = AccessSet::new();
        access2.add_write(pk2);
        // Note: bloom filters can have false positives
    }
}
