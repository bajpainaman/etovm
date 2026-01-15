use super::Pubkey;
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// Account structure compatible with Solana's account model
#[derive(
    Clone, Debug, Default, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct Account {
    /// Lamports held by this account
    pub lamports: u64,
    /// Data held by this account (arbitrary bytes)
    pub data: Vec<u8>,
    /// Program that owns this account
    pub owner: Pubkey,
    /// Whether this account's data is an executable program
    pub executable: bool,
    /// Epoch at which this account will next owe rent
    pub rent_epoch: u64,
}

impl Account {
    pub fn new(lamports: u64, space: usize, owner: &Pubkey) -> Self {
        Self {
            lamports,
            data: vec![0; space],
            owner: *owner,
            executable: false,
            rent_epoch: 0,
        }
    }

    pub fn new_data<T: BorshSerialize>(
        lamports: u64,
        state: &T,
        owner: &Pubkey,
    ) -> Result<Self, std::io::Error> {
        let data = borsh::to_vec(state)?;
        Ok(Self {
            lamports,
            data,
            owner: *owner,
            executable: false,
            rent_epoch: 0,
        })
    }

    /// Calculate the minimum lamports needed for rent exemption
    pub fn minimum_balance(data_len: usize) -> u64 {
        const LAMPORTS_PER_BYTE_YEAR: u64 = 3480;
        const EXEMPTION_THRESHOLD: u64 = 2;
        const ACCOUNT_STORAGE_OVERHEAD: usize = 128;

        let total_size = data_len + ACCOUNT_STORAGE_OVERHEAD;
        (total_size as u64) * LAMPORTS_PER_BYTE_YEAR * EXEMPTION_THRESHOLD
    }

    pub fn is_rent_exempt(&self, _rent_lamports_per_byte_year: u64) -> bool {
        self.lamports >= Self::minimum_balance(self.data.len())
    }

    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    pub fn set_data(&mut self, data: Vec<u8>) {
        self.data = data;
    }

    pub fn set_lamports(&mut self, lamports: u64) {
        self.lamports = lamports;
    }

    pub fn checked_add_lamports(&mut self, amount: u64) -> Result<(), AccountError> {
        self.lamports = self
            .lamports
            .checked_add(amount)
            .ok_or(AccountError::Overflow)?;
        Ok(())
    }

    pub fn checked_sub_lamports(&mut self, amount: u64) -> Result<(), AccountError> {
        self.lamports = self
            .lamports
            .checked_sub(amount)
            .ok_or(AccountError::InsufficientFunds)?;
        Ok(())
    }
}

/// Metadata for an account stored in the state trie
#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct AccountMeta {
    pub pubkey: Pubkey,
    pub is_signer: bool,
    pub is_writable: bool,
}

impl AccountMeta {
    pub fn new(pubkey: Pubkey, is_signer: bool) -> Self {
        Self {
            pubkey,
            is_signer,
            is_writable: true,
        }
    }

    pub fn new_readonly(pubkey: Pubkey, is_signer: bool) -> Self {
        Self {
            pubkey,
            is_signer,
            is_writable: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AccountError {
    #[error("Account lamports overflow")]
    Overflow,
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("Account not found")]
    NotFound,
    #[error("Account data too large")]
    DataTooLarge,
    #[error("Invalid account owner")]
    InvalidOwner,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_creation() {
        let owner = Pubkey::system_program();
        let account = Account::new(1000, 100, &owner);
        assert_eq!(account.lamports, 1000);
        assert_eq!(account.data.len(), 100);
        assert_eq!(account.owner, owner);
    }

    #[test]
    fn test_minimum_balance() {
        let min = Account::minimum_balance(100);
        assert!(min > 0);
    }
}
