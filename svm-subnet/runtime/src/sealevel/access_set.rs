//! Transaction Access Set Analysis
//!
//! Extracts the read and write sets from transactions to enable
//! conflict detection and parallel scheduling.

use crate::types::{Pubkey, Transaction};
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
}
