//! Transaction Access Set Analysis
//!
//! Extracts the read and write sets from transactions to enable
//! conflict detection and parallel scheduling.
//!
//! ## Performance Optimizations
//!
//! This module provides two access set implementations:
//! - `AccessSet`: HashSet-based, flexible but O(n) conflict check
//! - `BitsetAccessSet`: Bitset-based, O(1) conflict check via bitwise AND
//!
//! For batch scheduling with N transactions, bitsets reduce conflict
//! detection from O(N² × accounts) to O(N² × 1).

use crate::types::{Pubkey, Transaction};
use std::collections::HashMap;
use std::collections::HashSet;

/// Type of access to an account
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    /// Read-only access
    Read,
    /// Read-write access
    Write,
}

/// Access set for a single transaction
///
/// Contains the set of accounts that will be read and/or written.
/// Two transactions conflict if they both access the same account
/// and at least one access is a write.
#[derive(Debug, Clone)]
pub struct AccessSet {
    /// Accounts that are read (but not written)
    pub reads: HashSet<Pubkey>,
    /// Accounts that are written (may also be read)
    pub writes: HashSet<Pubkey>,
    /// Programs invoked (needed for CPI conflict detection)
    pub programs: HashSet<Pubkey>,
}

impl AccessSet {
    /// Create an empty access set
    pub fn new() -> Self {
        Self {
            reads: HashSet::new(),
            writes: HashSet::new(),
            programs: HashSet::new(),
        }
    }

    /// Extract access set from a transaction
    ///
    /// Uses the transaction message header to determine which accounts
    /// are signers (writable) vs read-only.
    pub fn from_transaction(tx: &Transaction) -> Self {
        let mut access = Self::new();
        let msg = &tx.message;
        let header = &msg.header;

        // Total counts from header
        let num_required_sigs = header.num_required_signatures as usize;
        let num_readonly_signed = header.num_readonly_signed_accounts as usize;
        let num_readonly_unsigned = header.num_readonly_unsigned_accounts as usize;

        let total_accounts = msg.account_keys.len();
        let num_signed_writable = num_required_sigs.saturating_sub(num_readonly_signed);
        let num_unsigned = total_accounts.saturating_sub(num_required_sigs);
        let num_unsigned_writable = num_unsigned.saturating_sub(num_readonly_unsigned);

        for (i, pubkey) in msg.account_keys.iter().enumerate() {
            let is_writable = if i < num_required_sigs {
                // Signed account: writable if not in readonly_signed section
                i < num_signed_writable
            } else {
                // Unsigned account: writable if not in readonly_unsigned section
                let unsigned_index = i - num_required_sigs;
                unsigned_index < num_unsigned_writable
            };

            if is_writable {
                access.writes.insert(*pubkey);
            } else {
                access.reads.insert(*pubkey);
            }
        }

        // Extract program IDs from instructions
        for ix in &msg.instructions {
            let program_idx = ix.program_id_index as usize;
            if program_idx < msg.account_keys.len() {
                access.programs.insert(msg.account_keys[program_idx]);
            }
        }

        access
    }

    /// Add a read access
    pub fn add_read(&mut self, pubkey: Pubkey) {
        if !self.writes.contains(&pubkey) {
            self.reads.insert(pubkey);
        }
    }

    /// Add a write access (removes from reads if present)
    pub fn add_write(&mut self, pubkey: Pubkey) {
        self.reads.remove(&pubkey);
        self.writes.insert(pubkey);
    }

    /// Check if this access set conflicts with another
    ///
    /// Conflict occurs when:
    /// - Both access the same account AND
    /// - At least one is a write
    pub fn conflicts_with(&self, other: &AccessSet) -> bool {
        // Check if our writes conflict with their reads or writes
        for pubkey in &self.writes {
            if other.writes.contains(pubkey) || other.reads.contains(pubkey) {
                return true;
            }
        }

        // Check if their writes conflict with our reads
        for pubkey in &other.writes {
            if self.reads.contains(pubkey) {
                return true;
            }
        }

        false
    }

    /// Get all accounts accessed (both reads and writes)
    pub fn all_accounts(&self) -> HashSet<Pubkey> {
        let mut all = self.reads.clone();
        all.extend(&self.writes);
        all
    }

    /// Get the number of accounts accessed
    pub fn account_count(&self) -> usize {
        self.reads.len() + self.writes.len()
    }

    /// Check if this transaction is read-only (no writes)
    pub fn is_readonly(&self) -> bool {
        self.writes.is_empty()
    }

    /// Merge another access set into this one
    pub fn merge(&mut self, other: &AccessSet) {
        for pubkey in &other.writes {
            self.add_write(*pubkey);
        }
        for pubkey in &other.reads {
            self.add_read(*pubkey);
        }
        self.programs.extend(&other.programs);
    }
}

impl Default for AccessSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// High-Performance Bitset-Based Access Sets
// ============================================================================

/// Maximum accounts supported in a single batch's bitset
/// 256 bits = 4 x u64, handles most real-world transactions
const BITSET_SIZE: usize = 256;
const WORDS: usize = BITSET_SIZE / 64;

/// Fixed-size bitset for O(1) conflict detection
#[derive(Clone, Copy, Default)]
pub struct Bitset256 {
    bits: [u64; WORDS],
}

impl Bitset256 {
    #[inline]
    pub fn new() -> Self {
        Self { bits: [0; WORDS] }
    }

    /// Set a bit at the given index
    #[inline]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < BITSET_SIZE);
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Check if a bit is set
    #[inline]
    pub fn test(&self, idx: usize) -> bool {
        debug_assert!(idx < BITSET_SIZE);
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Bitwise OR with another bitset
    #[inline]
    pub fn union(&self, other: &Bitset256) -> Bitset256 {
        let mut result = Bitset256::new();
        for i in 0..WORDS {
            result.bits[i] = self.bits[i] | other.bits[i];
        }
        result
    }

    /// Bitwise AND - returns true if intersection is non-empty
    #[inline]
    pub fn intersects(&self, other: &Bitset256) -> bool {
        for i in 0..WORDS {
            if (self.bits[i] & other.bits[i]) != 0 {
                return true;
            }
        }
        false
    }

    /// Union in place
    #[inline]
    pub fn union_inplace(&mut self, other: &Bitset256) {
        for i in 0..WORDS {
            self.bits[i] |= other.bits[i];
        }
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }
}

impl std::fmt::Debug for Bitset256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitset256({:016x}{:016x}{:016x}{:016x})",
            self.bits[3], self.bits[2], self.bits[1], self.bits[0])
    }
}

/// Maps pubkeys to sequential bit indices for a batch
///
/// Built once per batch, allows converting HashSet-based AccessSet
/// to BitsetAccessSet for O(1) conflict checks.
#[derive(Debug)]
pub struct AccountIndexer {
    /// Pubkey -> bit index mapping
    indices: HashMap<Pubkey, usize>,
    /// Next available index
    next_idx: usize,
}

impl AccountIndexer {
    pub fn new() -> Self {
        Self {
            indices: HashMap::with_capacity(1024),
            next_idx: 0,
        }
    }

    /// Build indexer from a slice of access sets
    pub fn from_access_sets(access_sets: &[AccessSet]) -> Self {
        let mut indexer = Self::new();
        for access in access_sets {
            for pk in &access.reads {
                indexer.get_or_assign(*pk);
            }
            for pk in &access.writes {
                indexer.get_or_assign(*pk);
            }
        }
        indexer
    }

    /// Get index for pubkey, assigning new index if needed
    #[inline]
    pub fn get_or_assign(&mut self, pubkey: Pubkey) -> usize {
        if let Some(&idx) = self.indices.get(&pubkey) {
            idx
        } else {
            let idx = self.next_idx;
            // Wrap around if we exceed BITSET_SIZE (rare case)
            self.next_idx = (self.next_idx + 1) % BITSET_SIZE;
            self.indices.insert(pubkey, idx);
            idx
        }
    }

    /// Get index for pubkey (returns None if not indexed)
    #[inline]
    pub fn get(&self, pubkey: &Pubkey) -> Option<usize> {
        self.indices.get(pubkey).copied()
    }

    /// Total unique accounts indexed
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl Default for AccountIndexer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bitset-based access set for O(1) conflict detection
///
/// This is the high-performance version of AccessSet used in
/// batch scheduling. Conflict detection is a single bitwise AND
/// operation instead of HashSet iteration.
#[derive(Debug, Clone, Copy, Default)]
pub struct BitsetAccessSet {
    /// Accounts read (but not written)
    pub reads: Bitset256,
    /// Accounts written
    pub writes: Bitset256,
    /// All accounts accessed (reads | writes) - precomputed for speed
    pub all: Bitset256,
}

impl BitsetAccessSet {
    pub fn new() -> Self {
        Self {
            reads: Bitset256::new(),
            writes: Bitset256::new(),
            all: Bitset256::new(),
        }
    }

    /// Convert from HashSet-based AccessSet using an indexer
    pub fn from_access_set(access: &AccessSet, indexer: &AccountIndexer) -> Self {
        let mut bitset = Self::new();

        for pk in &access.reads {
            if let Some(idx) = indexer.get(pk) {
                bitset.reads.set(idx);
                bitset.all.set(idx);
            }
        }

        for pk in &access.writes {
            if let Some(idx) = indexer.get(pk) {
                bitset.writes.set(idx);
                bitset.all.set(idx);
            }
        }

        bitset
    }

    /// Add a read at the given bit index
    #[inline]
    pub fn add_read(&mut self, idx: usize) {
        if !self.writes.test(idx) {
            self.reads.set(idx);
        }
        self.all.set(idx);
    }

    /// Add a write at the given bit index
    #[inline]
    pub fn add_write(&mut self, idx: usize) {
        self.writes.set(idx);
        self.all.set(idx);
        // Note: we don't clear from reads for bitset (not needed for conflict detection)
    }

    /// O(1) conflict detection via bitwise AND
    ///
    /// Conflict occurs when:
    /// - (our writes) AND (their reads OR writes) is non-empty, OR
    /// - (their writes) AND (our reads) is non-empty
    #[inline]
    pub fn conflicts_with(&self, other: &BitsetAccessSet) -> bool {
        // Check: our writes vs their all accounts
        if self.writes.intersects(&other.all) {
            return true;
        }
        // Check: their writes vs our reads
        other.writes.intersects(&self.reads)
    }

    /// Merge another bitset access set into this one
    #[inline]
    pub fn merge(&mut self, other: &BitsetAccessSet) {
        self.reads.union_inplace(&other.reads);
        self.writes.union_inplace(&other.writes);
        self.all.union_inplace(&other.all);
    }

    /// Check if this transaction is read-only
    #[inline]
    pub fn is_readonly(&self) -> bool {
        self.writes.is_empty()
    }
}

/// High-performance batch with bitset-based conflict tracking
#[derive(Debug, Clone)]
pub struct BitsetBatch {
    /// Transaction indices in this batch
    pub tx_indices: Vec<usize>,
    /// Combined read set for the batch
    pub reads: Bitset256,
    /// Combined write set for the batch
    pub writes: Bitset256,
    /// All accounts in the batch
    pub all: Bitset256,
}

impl BitsetBatch {
    pub fn new() -> Self {
        Self {
            tx_indices: Vec::with_capacity(64),
            reads: Bitset256::new(),
            writes: Bitset256::new(),
            all: Bitset256::new(),
        }
    }

    /// O(1) check if transaction can be added without conflicts
    #[inline]
    pub fn can_add(&self, access: &BitsetAccessSet) -> bool {
        // No write-write conflicts
        if self.writes.intersects(&access.writes) {
            return false;
        }
        // No write-read conflicts
        if self.writes.intersects(&access.reads) {
            return false;
        }
        // No read-write conflicts
        if self.reads.intersects(&access.writes) {
            return false;
        }
        true
    }

    /// Add transaction to batch (updates combined sets)
    #[inline]
    pub fn add(&mut self, tx_idx: usize, access: &BitsetAccessSet) {
        self.tx_indices.push(tx_idx);
        self.reads.union_inplace(&access.reads);
        self.writes.union_inplace(&access.writes);
        self.all.union_inplace(&access.all);
    }

    pub fn len(&self) -> usize {
        self.tx_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tx_indices.is_empty()
    }
}

impl Default for BitsetBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_no_conflict_disjoint() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));
        a.add_read(make_pubkey(2));

        let mut b = AccessSet::new();
        b.add_write(make_pubkey(3));
        b.add_read(make_pubkey(4));

        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_conflict_write_write() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));

        let mut b = AccessSet::new();
        b.add_write(make_pubkey(1));

        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_conflict_write_read() {
        let mut a = AccessSet::new();
        a.add_write(make_pubkey(1));

        let mut b = AccessSet::new();
        b.add_read(make_pubkey(1));

        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_no_conflict_read_read() {
        let mut a = AccessSet::new();
        a.add_read(make_pubkey(1));

        let mut b = AccessSet::new();
        b.add_read(make_pubkey(1));

        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_readonly_check() {
        let mut a = AccessSet::new();
        a.add_read(make_pubkey(1));
        a.add_read(make_pubkey(2));
        assert!(a.is_readonly());

        a.add_write(make_pubkey(3));
        assert!(!a.is_readonly());
    }

    // ========================================
    // Bitset Tests
    // ========================================

    #[test]
    fn test_bitset_basic_ops() {
        let mut bits = Bitset256::new();
        assert!(bits.is_empty());

        bits.set(0);
        bits.set(63);
        bits.set(64);
        bits.set(127);
        bits.set(255);

        assert!(bits.test(0));
        assert!(bits.test(63));
        assert!(bits.test(64));
        assert!(bits.test(127));
        assert!(bits.test(255));
        assert!(!bits.test(1));
        assert!(!bits.test(128));
    }

    #[test]
    fn test_bitset_intersects() {
        let mut a = Bitset256::new();
        a.set(10);
        a.set(20);

        let mut b = Bitset256::new();
        b.set(20);
        b.set(30);

        assert!(a.intersects(&b));  // Both have bit 20

        let mut c = Bitset256::new();
        c.set(100);
        c.set(200);

        assert!(!a.intersects(&c));  // No overlap
    }

    #[test]
    fn test_bitset_access_set_conflict() {
        let mut a = BitsetAccessSet::new();
        a.add_write(10);
        a.add_read(20);

        let mut b = BitsetAccessSet::new();
        b.add_write(30);
        b.add_read(40);

        assert!(!a.conflicts_with(&b));  // Disjoint

        let mut c = BitsetAccessSet::new();
        c.add_write(10);  // Conflicts with a's write

        assert!(a.conflicts_with(&c));

        let mut d = BitsetAccessSet::new();
        d.add_read(10);  // Conflicts with a's write

        assert!(a.conflicts_with(&d));
    }

    #[test]
    fn test_bitset_batch_can_add() {
        let mut batch = BitsetBatch::new();

        let mut tx1 = BitsetAccessSet::new();
        tx1.add_write(10);
        tx1.add_read(20);
        batch.add(0, &tx1);

        let mut tx2 = BitsetAccessSet::new();
        tx2.add_write(30);  // Different account
        assert!(batch.can_add(&tx2));
        batch.add(1, &tx2);

        let mut tx3 = BitsetAccessSet::new();
        tx3.add_write(10);  // Conflicts with tx1
        assert!(!batch.can_add(&tx3));

        let mut tx4 = BitsetAccessSet::new();
        tx4.add_read(10);  // Conflicts with tx1's write
        assert!(!batch.can_add(&tx4));
    }

    #[test]
    fn test_indexer_and_conversion() {
        let mut access1 = AccessSet::new();
        access1.add_write(make_pubkey(1));
        access1.add_read(make_pubkey(2));

        let mut access2 = AccessSet::new();
        access2.add_write(make_pubkey(2));  // Overlaps with access1's read
        access2.add_read(make_pubkey(3));

        let indexer = AccountIndexer::from_access_sets(&[access1.clone(), access2.clone()]);
        assert_eq!(indexer.len(), 3);  // 3 unique pubkeys

        let bitset1 = BitsetAccessSet::from_access_set(&access1, &indexer);
        let bitset2 = BitsetAccessSet::from_access_set(&access2, &indexer);

        // access1 reads pubkey(2), access2 writes pubkey(2) -> conflict
        assert!(bitset1.conflicts_with(&bitset2));
    }

    #[test]
    fn test_bitset_matches_hashset_behavior() {
        // Ensure bitset conflict detection matches HashSet version
        for i in 0..100u8 {
            let mut h1 = AccessSet::new();
            let mut h2 = AccessSet::new();

            // Varying patterns
            h1.add_write(make_pubkey(i));
            h1.add_read(make_pubkey(i.wrapping_add(1)));
            h2.add_write(make_pubkey(i.wrapping_add(2)));
            h2.add_read(make_pubkey(i.wrapping_mul(3)));

            let indexer = AccountIndexer::from_access_sets(&[h1.clone(), h2.clone()]);
            let b1 = BitsetAccessSet::from_access_set(&h1, &indexer);
            let b2 = BitsetAccessSet::from_access_set(&h2, &indexer);

            // Both implementations should agree
            assert_eq!(
                h1.conflicts_with(&h2),
                b1.conflicts_with(&b2),
                "Mismatch at i={}", i
            );
        }
    }
}
