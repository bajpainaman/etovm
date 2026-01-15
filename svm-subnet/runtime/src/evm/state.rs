use crate::{Account, Pubkey, RuntimeResult};
use super::{EvmAddress, evm_address_to_pubkey, pubkey_to_evm_address};
use std::collections::HashMap;

use revm::primitives::{
    AccountInfo, Address, Bytecode, B256, U256, KECCAK_EMPTY,
};
use revm::Database;

/// Adapter that presents SVM account state as EVM state
pub struct EvmStateAdapter {
    /// Cache of EVM accounts (loaded from SVM state)
    accounts: HashMap<Address, AccountInfo>,
    /// Storage cache
    storage: HashMap<(Address, U256), U256>,
    /// Code cache
    code: HashMap<Address, Bytecode>,
    /// Block hashes
    block_hashes: HashMap<u64, B256>,
    /// SVM account getter function
    svm_accounts: HashMap<[u8; 32], Account>,
}

impl EvmStateAdapter {
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            storage: HashMap::new(),
            code: HashMap::new(),
            block_hashes: HashMap::new(),
            svm_accounts: HashMap::new(),
        }
    }

    /// Load SVM accounts into the adapter
    pub fn load_accounts(&mut self, accounts: Vec<([u8; 32], Account)>) {
        for (pubkey, account) in accounts {
            // Also create EVM view if this is an EVM-mapped account
            let evm_addr = pubkey_to_evm_address(&pubkey);
            let address = Address::from_slice(&evm_addr);

            // Convert lamports to wei (1 lamport = 1 gwei = 1e9 wei)
            let balance = U256::from(account.lamports) * U256::from(1_000_000_000u64);

            // Insert SVM account (must happen after reading balance)
            self.svm_accounts.insert(pubkey, account);

            let account_info = AccountInfo {
                balance,
                nonce: 0,
                code_hash: KECCAK_EMPTY,
                code: None,
            };

            self.accounts.insert(address, account_info);
        }
    }

    /// Set a block hash
    pub fn set_block_hash(&mut self, number: u64, hash: [u8; 32]) {
        self.block_hashes.insert(number, B256::from(hash));
    }

    /// Get modified SVM accounts after EVM execution
    pub fn get_modified_accounts(&self) -> Vec<([u8; 32], Account)> {
        self.svm_accounts
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Apply EVM state changes back to SVM accounts
    pub fn apply_changes(&mut self, address: Address, info: AccountInfo) {
        let evm_addr: [u8; 20] = address.as_slice().try_into().unwrap();
        let pubkey = evm_address_to_pubkey(&evm_addr);

        // Convert balance back to lamports
        let lamports = (info.balance / U256::from(1_000_000_000u64)).as_limbs()[0];

        if let Some(account) = self.svm_accounts.get_mut(&pubkey) {
            account.lamports = lamports;
        } else {
            // Create new account
            self.svm_accounts.insert(pubkey, Account {
                lamports,
                data: vec![],
                owner: Pubkey([0u8; 32]), // EVM-owned
                executable: false,
                rent_epoch: 0,
            });
        }
    }

    /// Store EVM contract code
    pub fn store_code(&mut self, address: Address, code: Vec<u8>) {
        let bytecode = Bytecode::new_raw(code.into());
        self.code.insert(address, bytecode);
    }

    /// Store EVM storage
    pub fn store_storage(&mut self, address: Address, slot: U256, value: U256) {
        self.storage.insert((address, slot), value);
    }
}

impl Default for EvmStateAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Database for EvmStateAdapter {
    type Error = crate::RuntimeError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        Ok(self.accounts.get(&address).cloned())
    }

    fn code_by_hash(&mut self, _code_hash: B256) -> Result<Bytecode, Self::Error> {
        Ok(Bytecode::default())
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        Ok(self.storage.get(&(address, index)).cloned().unwrap_or(U256::ZERO))
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        Ok(self.block_hashes.get(&number).cloned().unwrap_or(B256::ZERO))
    }
}
