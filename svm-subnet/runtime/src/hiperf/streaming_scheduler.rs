//! Streaming FAFO Scheduler - Based on LayerZero's actual implementation
//!
//! Key optimizations from real FAFO:
//! 1. Short hashes (8 bytes) instead of bloom filters
//! 2. Sorted two-pointer collision detection O(n+m)
//! 3. Streaming frame dispatch - execute as frames form
//! 4. Lock-free frame management

use crate::sealevel::AccessSet;
use crate::types::Pubkey;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Number of parallel frames
const NUM_FRAMES: usize = 64;

/// Short hash for compact access set representation (8 bytes like LayerZero)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShortHash(u64);

impl ShortHash {
    #[inline]
    pub fn from_pubkey(pk: &Pubkey) -> Self {
        // Use first 8 bytes as hash - very fast, good distribution
        let hash = u64::from_le_bytes(pk.0[0..8].try_into().unwrap());
        ShortHash(hash)
    }
}

/// Compact access set using sorted short hashes
#[derive(Clone, Default)]
pub struct CompactAccessSet {
    /// Sorted read hashes
    pub reads: Vec<ShortHash>,
    /// Sorted write hashes
    pub writes: Vec<ShortHash>,
}

impl CompactAccessSet {
    pub fn new() -> Self {
        Self {
            reads: Vec::with_capacity(4),
            writes: Vec::with_capacity(4),
        }
    }

    pub fn from_access_set(access: &AccessSet) -> Self {
        let mut reads: Vec<ShortHash> = access.reads.iter()
            .map(ShortHash::from_pubkey)
            .collect();
        let mut writes: Vec<ShortHash> = access.writes.iter()
            .map(ShortHash::from_pubkey)
            .collect();

        reads.sort_unstable();
        writes.sort_unstable();
        reads.dedup();
        writes.dedup();

        Self { reads, writes }
    }

    /// Two-pointer collision detection - O(n+m) like LayerZero
    /// Returns true if this access set conflicts with other
    #[inline]
    pub fn conflicts_with(&self, other: &CompactAccessSet) -> bool {
        // Check: our writes vs their reads+writes
        if Self::sorted_intersects(&self.writes, &other.reads) {
            return true;
        }
        if Self::sorted_intersects(&self.writes, &other.writes) {
            return true;
        }
        // Check: our reads vs their writes
        if Self::sorted_intersects(&self.reads, &other.writes) {
            return true;
        }
        false
    }

    /// Two-pointer intersection check on sorted arrays - O(n+m)
    #[inline]
    fn sorted_intersects(a: &[ShortHash], b: &[ShortHash]) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }

        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                return true;
            } else if a[i] < b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        false
    }
}

/// Frame accumulator for streaming execution
#[derive(Default)]
pub struct FrameAccumulator {
    /// Combined read hashes for this frame (sorted)
    reads: Vec<ShortHash>,
    /// Combined write hashes for this frame (sorted)
    writes: Vec<ShortHash>,
    /// Transaction indices in this frame
    tx_indices: Vec<usize>,
}

impl FrameAccumulator {
    pub fn new() -> Self {
        Self {
            reads: Vec::with_capacity(1024),
            writes: Vec::with_capacity(1024),
            tx_indices: Vec::with_capacity(1024),
        }
    }

    /// Check if a transaction conflicts with this frame
    #[inline]
    pub fn conflicts(&self, access: &CompactAccessSet) -> bool {
        // tx writes vs frame all
        if Self::sorted_intersects(&access.writes, &self.reads) {
            return true;
        }
        if Self::sorted_intersects(&access.writes, &self.writes) {
            return true;
        }
        // tx reads vs frame writes
        if Self::sorted_intersects(&access.reads, &self.writes) {
            return true;
        }
        false
    }

    /// Add transaction to this frame
    #[inline]
    pub fn add(&mut self, tx_idx: usize, access: &CompactAccessSet) {
        self.tx_indices.push(tx_idx);

        // Merge reads
        self.reads.extend_from_slice(&access.reads);
        // Merge writes
        self.writes.extend_from_slice(&access.writes);
    }

    /// Finalize frame - sort and dedup for next conflict checks
    pub fn finalize(&mut self) {
        self.reads.sort_unstable();
        self.reads.dedup();
        self.writes.sort_unstable();
        self.writes.dedup();
    }

    pub fn len(&self) -> usize {
        self.tx_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tx_indices.is_empty()
    }

    pub fn take_indices(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.tx_indices)
    }

    #[inline]
    fn sorted_intersects(a: &[ShortHash], b: &[ShortHash]) -> bool {
        CompactAccessSet::sorted_intersects(a, b)
    }
}

/// Streaming execution frame
pub struct StreamFrame {
    pub tx_indices: Vec<usize>,
}

/// Streaming FAFO Scheduler
///
/// Processes transactions and emits frames as they're formed,
/// allowing pipelined execution.
pub struct StreamingScheduler {
    max_frame_size: usize,
}

impl StreamingScheduler {
    pub fn new(max_frame_size: usize) -> Self {
        Self { max_frame_size }
    }

    /// Schedule with streaming - emits frames via channel as they're formed
    pub fn schedule_streaming(
        &self,
        access_sets: &[AccessSet],
        frame_tx: Sender<StreamFrame>,
    ) {
        if access_sets.is_empty() {
            return;
        }

        // Convert to compact access sets in parallel
        let compact: Vec<CompactAccessSet> = access_sets
            .par_iter()
            .map(CompactAccessSet::from_access_set)
            .collect();

        // Process with streaming frames
        let mut frames: Vec<FrameAccumulator> = (0..NUM_FRAMES)
            .map(|_| FrameAccumulator::new())
            .collect();

        let mut active_mask: u64 = u64::MAX;

        for (tx_idx, access) in compact.iter().enumerate() {
            // Find first non-conflicting frame
            let mut found_frame = None;
            let mut test_mask = active_mask;

            while test_mask != 0 {
                let frame_idx = test_mask.trailing_zeros() as usize;
                if frame_idx >= NUM_FRAMES {
                    break;
                }

                if !frames[frame_idx].conflicts(access) {
                    found_frame = Some(frame_idx);
                    break;
                }

                test_mask &= !(1u64 << frame_idx);
            }

            match found_frame {
                Some(frame_idx) => {
                    frames[frame_idx].add(tx_idx, access);

                    // Check if frame should be flushed
                    if frames[frame_idx].len() >= self.max_frame_size {
                        frames[frame_idx].finalize();
                        let indices = frames[frame_idx].take_indices();
                        let _ = frame_tx.send(StreamFrame { tx_indices: indices });
                        frames[frame_idx] = FrameAccumulator::new();
                    }
                }
                None => {
                    // All frames conflict - flush oldest and retry
                    let oldest = active_mask.trailing_zeros() as usize;
                    if oldest < NUM_FRAMES && !frames[oldest].is_empty() {
                        frames[oldest].finalize();
                        let indices = frames[oldest].take_indices();
                        let _ = frame_tx.send(StreamFrame { tx_indices: indices });
                    }
                    frames[oldest] = FrameAccumulator::new();
                    frames[oldest].add(tx_idx, access);
                }
            }
        }

        // Flush remaining frames
        for frame in &mut frames {
            if !frame.is_empty() {
                frame.finalize();
                let indices = frame.take_indices();
                let _ = frame_tx.send(StreamFrame { tx_indices: indices });
            }
        }
    }

    /// Non-streaming schedule for comparison
    pub fn schedule(&self, access_sets: &[AccessSet]) -> Vec<StreamFrame> {
        if access_sets.is_empty() {
            return vec![];
        }

        // Convert to compact access sets in parallel
        let compact: Vec<CompactAccessSet> = access_sets
            .par_iter()
            .map(CompactAccessSet::from_access_set)
            .collect();

        let mut frames: Vec<FrameAccumulator> = (0..NUM_FRAMES)
            .map(|_| FrameAccumulator::new())
            .collect();
        let mut completed: Vec<StreamFrame> = Vec::new();
        let mut active_mask: u64 = u64::MAX;

        for (tx_idx, access) in compact.iter().enumerate() {
            let mut found_frame = None;
            let mut test_mask = active_mask;

            while test_mask != 0 {
                let frame_idx = test_mask.trailing_zeros() as usize;
                if frame_idx >= NUM_FRAMES {
                    break;
                }

                if !frames[frame_idx].conflicts(access) {
                    found_frame = Some(frame_idx);
                    break;
                }

                test_mask &= !(1u64 << frame_idx);
            }

            match found_frame {
                Some(frame_idx) => {
                    frames[frame_idx].add(tx_idx, access);

                    if frames[frame_idx].len() >= self.max_frame_size {
                        let indices = frames[frame_idx].take_indices();
                        completed.push(StreamFrame { tx_indices: indices });
                        frames[frame_idx] = FrameAccumulator::new();
                    }
                }
                None => {
                    let oldest = active_mask.trailing_zeros() as usize;
                    if oldest < NUM_FRAMES && !frames[oldest].is_empty() {
                        let indices = frames[oldest].take_indices();
                        completed.push(StreamFrame { tx_indices: indices });
                    }
                    frames[oldest] = FrameAccumulator::new();
                    frames[oldest].add(tx_idx, access);
                }
            }
        }

        for frame in &mut frames {
            if !frame.is_empty() {
                let indices = frame.take_indices();
                completed.push(StreamFrame { tx_indices: indices });
            }
        }

        completed
    }

    /// Ultra-fast schedule - minimal overhead path
    /// For non-conflicting workloads, just batch everything
    pub fn schedule_fast(&self, access_sets: &[AccessSet]) -> Vec<StreamFrame> {
        if access_sets.is_empty() {
            return vec![];
        }

        // For independent txs, single frame is optimal
        let n = access_sets.len();
        let frame_size = self.max_frame_size.max(10_000);

        let mut frames = Vec::with_capacity((n + frame_size - 1) / frame_size);

        for chunk_start in (0..n).step_by(frame_size) {
            let chunk_end = (chunk_start + frame_size).min(n);
            frames.push(StreamFrame {
                tx_indices: (chunk_start..chunk_end).collect(),
            });
        }

        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u64) -> Pubkey {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&seed.to_le_bytes());
        Pubkey(bytes)
    }

    #[test]
    fn test_compact_access_set() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));
        a.add_read(make_pubkey(2));

        let compact = CompactAccessSet::from_access_set(&a);
        assert_eq!(compact.writes.len(), 1);
        assert_eq!(compact.reads.len(), 1);
    }

    #[test]
    fn test_collision_detection() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));

        let mut b = AccessSet::new();
        b.add_read(make_pubkey(1));

        let ca = CompactAccessSet::from_access_set(&a);
        let cb = CompactAccessSet::from_access_set(&b);

        assert!(ca.conflicts_with(&cb));
    }

    #[test]
    fn test_no_collision() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));

        let mut b = AccessSet::new();
        b.add_write(make_pubkey(2));

        let ca = CompactAccessSet::from_access_set(&a);
        let cb = CompactAccessSet::from_access_set(&b);

        assert!(!ca.conflicts_with(&cb));
    }

    #[test]
    fn test_streaming_scheduler() {
        let scheduler = StreamingScheduler::new(1000);

        let access_sets: Vec<AccessSet> = (0..100u64)
            .map(|i| {
                let mut a = AccessSet::new();
                a.add_write(make_pubkey(i));
                a
            })
            .collect();

        let frames = scheduler.schedule(&access_sets);
        let total: usize = frames.iter().map(|f| f.tx_indices.len()).sum();
        assert_eq!(total, 100);
    }
}
