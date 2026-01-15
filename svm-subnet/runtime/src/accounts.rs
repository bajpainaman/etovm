use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, AccountMeta, Pubkey};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Account database interface - will be backed by AvalancheGo's merkledb
pub trait AccountsDB: Send + Sync {
    fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>>;
    fn set_account(&mut self, pubkey: &Pubkey, account: &Account) -> RuntimeResult<()>;
    fn delete_account(&mut self, pubkey: &Pubkey) -> RuntimeResult<()>;
    fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool>;
}

/// In-memory accounts database for testing
#[derive(Default, Clone)]
pub struct InMemoryAccountsDB {
    accounts: HashMap<Pubkey, Account>,
}

impl InMemoryAccountsDB {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_accounts(accounts: Vec<(Pubkey, Account)>) -> Self {
        Self {
            accounts: accounts.into_iter().collect(),
        }
    }
}

impl AccountsDB for InMemoryAccountsDB {
    fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        Ok(self.accounts.get(pubkey).cloned())
    }

    fn set_account(&mut self, pubkey: &Pubkey, account: &Account) -> RuntimeResult<()> {
        self.accounts.insert(*pubkey, account.clone());
        Ok(())
    }

    fn delete_account(&mut self, pubkey: &Pubkey) -> RuntimeResult<()> {
        self.accounts.remove(pubkey);
        Ok(())
    }

    fn account_exists(&self, pubkey: &Pubkey) -> RuntimeResult<bool> {
        Ok(self.accounts.contains_key(pubkey))
    }
}

/// Thread-safe wrapper for accounts database
pub struct AccountsManager<DB: AccountsDB> {
    db: Arc<RwLock<DB>>,
}

impl<DB: AccountsDB> AccountsManager<DB> {
    pub fn new(db: DB) -> Self {
        Self {
            db: Arc::new(RwLock::new(db)),
        }
    }

    pub fn get_account(&self, pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        let db = self
            .db
            .read()
            .map_err(|e| RuntimeError::State(e.to_string()))?;
        db.get_account(pubkey)
    }

    pub fn get_account_or_default(&self, pubkey: &Pubkey) -> RuntimeResult<Account> {
        self.get_account(pubkey).map(|opt| opt.unwrap_or_default())
    }

    pub fn set_account(&self, pubkey: &Pubkey, account: &Account) -> RuntimeResult<()> {
        let mut db = self
            .db
            .write()
            .map_err(|e| RuntimeError::State(e.to_string()))?;
        db.set_account(pubkey, account)
    }

    pub fn load_accounts(&self, metas: &[AccountMeta]) -> RuntimeResult<Vec<(Pubkey, Account)>> {
        let db = self
            .db
            .read()
            .map_err(|e| RuntimeError::State(e.to_string()))?;

        let mut accounts = Vec::with_capacity(metas.len());
        for meta in metas {
            let account = db.get_account(&meta.pubkey)?.unwrap_or_default();
            accounts.push((meta.pubkey, account));
        }
        Ok(accounts)
    }

    pub fn commit_accounts(&self, accounts: &[(Pubkey, Account)]) -> RuntimeResult<()> {
        let mut db = self
            .db
            .write()
            .map_err(|e| RuntimeError::State(e.to_string()))?;

        for (pubkey, account) in accounts {
            if account.lamports == 0 && account.data.is_empty() {
                db.delete_account(pubkey)?;
            } else {
                db.set_account(pubkey, account)?;
            }
        }
        Ok(())
    }
}

impl<DB: AccountsDB> Clone for AccountsManager<DB> {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
        }
    }
}

/// Loaded account with metadata for execution
#[derive(Clone, Debug)]
pub struct LoadedAccount {
    pub pubkey: Pubkey,
    pub account: Account,
    pub is_signer: bool,
    pub is_writable: bool,
}

impl LoadedAccount {
    pub fn new(pubkey: Pubkey, account: Account, meta: &AccountMeta) -> Self {
        Self {
            pubkey,
            account,
            is_signer: meta.is_signer,
            is_writable: meta.is_writable,
        }
    }
}
