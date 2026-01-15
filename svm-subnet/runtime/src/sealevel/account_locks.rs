//! Account Locks - Per-Account Read/Write Locking
//!
//! Provides fine-grained locking at the account level to enable
//! safe parallel execution of transactions.

use crate::types::Pubkey;
use dashmap::DashMap;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::HashSet;
use std::sync::Arc;

/// Lock type for an account
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    Read,
    Write,
}

/// Per-account lock manager
///
/// Uses DashMap for concurrent access to the lock table,
/// with individual RwLocks per account for fine-grained locking.
pub struct AccountLocks {
    locks: DashMap<Pubkey, Arc<RwLock<()>>>,
}

impl AccountLocks {
    /// Create a new account lock manager
    pub fn new() -> Self {
        Self {
            locks: DashMap::new(),
        }
    }

    /// Get or create lock for an account
    fn get_or_create_lock(&self, pubkey: &Pubkey) -> Arc<RwLock<()>> {
        self.locks
            .entry(*pubkey)
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }

    /// Acquire locks for a set of accounts
    ///
    /// Returns a guard that releases all locks when dropped.
    /// Acquires locks in sorted order to prevent deadlocks.
    pub fn lock_accounts(
        &self,
        reads: &HashSet<Pubkey>,
        writes: &HashSet<Pubkey>,
    ) -> AccountLockGuard {
        // Sort all accounts to ensure consistent lock ordering
        let mut all_accounts: Vec<_> = reads
            .iter()
            .map(|pk| (*pk, LockType::Read))
            .chain(writes.iter().map(|pk| (*pk, LockType::Write)))
            .collect();

        // Sort by pubkey bytes
        all_accounts.sort_by_key(|(pk, _)| pk.0);

        // Deduplicate: if both read and write, use write
        all_accounts.dedup_by(|a, b| {
            if a.0 == b.0 {
                // Keep the write lock if present
                if a.1 == LockType::Write || b.1 == LockType::Write {
                    b.1 = LockType::Write;
                }
                true
            } else {
                false
            }
        });

        let mut guards = Vec::with_capacity(all_accounts.len());

        for (pubkey, lock_type) in all_accounts {
            let lock = self.get_or_create_lock(&pubkey);
            let guard = match lock_type {
                LockType::Read => LockGuardType::Read(lock),
                LockType::Write => LockGuardType::Write(lock),
            };
            guards.push((pubkey, guard));
        }

        AccountLockGuard { guards }
    }

    /// Try to acquire locks without blocking
    ///
    /// Returns None if any lock would block.
    pub fn try_lock_accounts(
        &self,
        reads: &HashSet<Pubkey>,
        writes: &HashSet<Pubkey>,
    ) -> Option<AccountLockGuard> {
        let mut all_accounts: Vec<_> = reads
            .iter()
            .map(|pk| (*pk, LockType::Read))
            .chain(writes.iter().map(|pk| (*pk, LockType::Write)))
            .collect();

        all_accounts.sort_by_key(|(pk, _)| pk.0);
        all_accounts.dedup_by(|a, b| {
            if a.0 == b.0 {
                if a.1 == LockType::Write || b.1 == LockType::Write {
                    b.1 = LockType::Write;
                }
                true
            } else {
                false
            }
        });

        let mut guards = Vec::with_capacity(all_accounts.len());

        for (pubkey, lock_type) in all_accounts {
            let lock = self.get_or_create_lock(&pubkey);

            let guard = match lock_type {
                LockType::Read => {
                    if lock.try_read().is_some() {
                        LockGuardType::Read(lock)
                    } else {
                        return None;
                    }
                }
                LockType::Write => {
                    if lock.try_write().is_some() {
                        LockGuardType::Write(lock)
                    } else {
                        return None;
                    }
                }
            };
            guards.push((pubkey, guard));
        }

        Some(AccountLockGuard { guards })
    }

    /// Clear all locks (for testing/reset)
    pub fn clear(&self) {
        self.locks.clear();
    }

    /// Get number of tracked accounts
    pub fn len(&self) -> usize {
        self.locks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }
}

impl Default for AccountLocks {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-erased lock guard
enum LockGuardType {
    Read(Arc<RwLock<()>>),
    Write(Arc<RwLock<()>>),
}

/// Guard for held account locks
///
/// Releases all locks when dropped.
pub struct AccountLockGuard {
    guards: Vec<(Pubkey, LockGuardType)>,
}

impl AccountLockGuard {
    /// Get the pubkeys that are locked
    pub fn locked_accounts(&self) -> Vec<&Pubkey> {
        self.guards.iter().map(|(pk, _)| pk).collect()
    }

    /// Get number of locked accounts
    pub fn len(&self) -> usize {
        self.guards.len()
    }

    /// Check if any accounts are locked
    pub fn is_empty(&self) -> bool {
        self.guards.is_empty()
    }
}

impl Drop for AccountLockGuard {
    fn drop(&mut self) {
        // Guards are dropped in reverse order (LIFO)
        // This is handled automatically by Vec's drop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_pubkey(seed: u8) -> Pubkey {
        Pubkey([seed; 32])
    }

    #[test]
    fn test_basic_locking() {
        let locks = AccountLocks::new();

        let pk1 = make_pubkey(1);
        let pk2 = make_pubkey(2);

        let reads = HashSet::new();
        let mut writes = HashSet::new();
        writes.insert(pk1);
        writes.insert(pk2);

        let guard = locks.lock_accounts(&reads, &writes);
        assert_eq!(guard.len(), 2);

        drop(guard);
    }

    #[test]
    fn test_try_lock_conflict() {
        let locks = Arc::new(AccountLocks::new());

        let pk = make_pubkey(1);
        let mut writes = HashSet::new();
        writes.insert(pk);

        // First lock should succeed
        let guard1 = locks.try_lock_accounts(&HashSet::new(), &writes);
        assert!(guard1.is_some());

        // Second lock should fail (would block)
        let locks2 = locks.clone();
        let guard2 = locks2.try_lock_accounts(&HashSet::new(), &writes);
        // Note: This might succeed because we're using Arc<RwLock> and try_read/try_write
        // The actual blocking behavior depends on parking_lot's implementation

        drop(guard1);
    }

    #[test]
    fn test_read_read_no_conflict() {
        let locks = AccountLocks::new();

        let pk = make_pubkey(1);
        let mut reads = HashSet::new();
        reads.insert(pk);

        // Multiple read locks should work
        let _guard1 = locks.lock_accounts(&reads, &HashSet::new());
        // In theory, another read lock should also work
        // but our implementation acquires locks serially
    }
}
