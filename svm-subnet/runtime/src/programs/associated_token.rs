//! Associated Token Account Program
//!
//! Creates and manages Associated Token Accounts (ATAs) - deterministically
//! derived token accounts for any wallet/mint pair.
//!
//! ATA Address = PDA([wallet, token_program_id, mint], ata_program_id)
//!
//! This enables:
//! - Predictable token account addresses
//! - No need to pass token account in instructions
//! - Automatic account creation on first transfer

use borsh::{BorshDeserialize, BorshSerialize};
use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, Pubkey};
use crate::programs::token::{TokenAccount, TokenProgram, TOKEN_PROGRAM_ID};
use std::collections::HashMap;

/// ATA Program ID (SPL Associated Token Account)
/// Real Solana: ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL
pub const ATA_PROGRAM_ID: [u8; 32] = [
    0x8c, 0x97, 0x25, 0x8f, 0x4e, 0x24, 0x89, 0xf1,
    0xbb, 0x3d, 0x10, 0x29, 0x14, 0x8e, 0x0d, 0x83,
    0x0b, 0x5a, 0x13, 0x99, 0xda, 0xff, 0x10, 0x84,
    0x04, 0x8e, 0x7b, 0xd8, 0xdb, 0xe9, 0xf8, 0x59,
];

/// ATA instruction types
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub enum AtaInstruction {
    /// Create an associated token account for the given wallet and mint
    ///
    /// Accounts:
    /// 0. `[writeable, signer]` Funding account (payer)
    /// 1. `[writeable]` Associated token account to create
    /// 2. `[]` Wallet address (owner of the new account)
    /// 3. `[]` Mint
    /// 4. `[]` System Program
    /// 5. `[]` Token Program
    Create,

    /// Create an associated token account idempotently
    /// Same as Create but doesn't error if account already exists
    ///
    /// Accounts: Same as Create
    CreateIdempotent,

    /// Recover nested associated token account
    /// Transfers tokens from a nested ATA back to the wallet's ATA
    ///
    /// Accounts:
    /// 0. `[writeable]` Nested ATA (ATA of an ATA)
    /// 1. `[]` Mint of nested ATA
    /// 2. `[writeable]` Wallet's ATA for the nested mint
    /// 3. `[]` Owner of the nested ATA (which is itself an ATA)
    /// 4. `[]` Mint of the owner ATA
    /// 5. `[signer]` Wallet that owns the owner ATA
    /// 6. `[]` Token Program
    RecoverNested,
}

/// Associated Token Account Program
pub struct AtaProgram {
    token_program: TokenProgram,
}

impl AtaProgram {
    pub fn new() -> Self {
        Self {
            token_program: TokenProgram::new(),
        }
    }

    /// Get the ATA program ID
    pub fn program_id() -> Pubkey {
        Pubkey::new(ATA_PROGRAM_ID)
    }

    /// Derive the associated token account address for a wallet and mint
    pub fn get_associated_token_address(wallet: &Pubkey, mint: &Pubkey) -> Pubkey {
        let (address, _bump) = Self::find_program_address(wallet, mint);
        address
    }

    /// Find the PDA and bump seed for an ATA
    pub fn find_program_address(wallet: &Pubkey, mint: &Pubkey) -> (Pubkey, u8) {
        let token_program_id = Pubkey::new(TOKEN_PROGRAM_ID);
        let seeds: &[&[u8]] = &[
            wallet.as_ref(),
            token_program_id.as_ref(),
            mint.as_ref(),
        ];
        Pubkey::find_program_address(seeds, &Self::program_id())
    }

    /// Process an ATA instruction
    pub fn process(
        &self,
        instruction_data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
        account_keys: &[Pubkey],
        signers: &[bool],
    ) -> RuntimeResult<()> {
        // Empty instruction data means Create
        let instruction = if instruction_data.is_empty() {
            AtaInstruction::Create
        } else {
            AtaInstruction::try_from_slice(instruction_data)
                .map_err(|_| RuntimeError::InvalidInstruction("Invalid ATA instruction".to_string()))?
        };

        match instruction {
            AtaInstruction::Create => {
                self.process_create(accounts, account_keys, signers, false)
            }
            AtaInstruction::CreateIdempotent => {
                self.process_create(accounts, account_keys, signers, true)
            }
            AtaInstruction::RecoverNested => {
                self.process_recover_nested(accounts, account_keys, signers)
            }
        }
    }

    /// Process Create/CreateIdempotent instruction
    fn process_create(
        &self,
        accounts: &mut HashMap<Pubkey, Account>,
        account_keys: &[Pubkey],
        signers: &[bool],
        idempotent: bool,
    ) -> RuntimeResult<()> {
        // Validate account count
        if account_keys.len() < 6 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let payer_key = &account_keys[0];
        let ata_key = &account_keys[1];
        let wallet_key = &account_keys[2];
        let mint_key = &account_keys[3];
        // account_keys[4] = System Program (not used directly)
        // account_keys[5] = Token Program (not used directly)

        // Verify payer is signer
        if !signers[0] {
            return Err(RuntimeError::InvalidSigner);
        }

        // Derive expected ATA address
        let (expected_ata, _bump) = Self::find_program_address(wallet_key, mint_key);
        if *ata_key != expected_ata {
            return Err(RuntimeError::InvalidArgument(
                "ATA address doesn't match derived address".to_string()
            ));
        }

        // Check if account already exists
        if let Some(existing) = accounts.get(ata_key) {
            if !existing.data.is_empty() {
                if idempotent {
                    // Account exists, that's fine for idempotent
                    return Ok(());
                } else {
                    return Err(RuntimeError::AccountAlreadyInitialized);
                }
            }
        }

        // Get rent-exempt minimum (use token account size)
        let rent_lamports = 2_039_280; // ~0.00203928 SOL for token account

        // Deduct rent from payer
        let payer = accounts.get_mut(payer_key)
            .ok_or(RuntimeError::AccountNotFound(*payer_key))?;

        if payer.lamports < rent_lamports {
            return Err(RuntimeError::InsufficientFunds);
        }
        payer.lamports -= rent_lamports;

        // Create the token account
        let token_account = TokenAccount {
            mint: *mint_key,
            owner: *wallet_key,
            amount: 0,
            delegate: None,
            state: crate::programs::token::AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut data = vec![];
        token_account.serialize(&mut data)
            .map_err(|e| RuntimeError::Serialization(e.to_string()))?;

        // Create the account
        let ata_account = Account {
            lamports: rent_lamports,
            data,
            owner: Pubkey::new(TOKEN_PROGRAM_ID),
            executable: false,
            rent_epoch: 0,
        };

        accounts.insert(*ata_key, ata_account);

        Ok(())
    }

    /// Process RecoverNested instruction
    fn process_recover_nested(
        &self,
        accounts: &mut HashMap<Pubkey, Account>,
        account_keys: &[Pubkey],
        signers: &[bool],
    ) -> RuntimeResult<()> {
        // Validate account count
        if account_keys.len() < 7 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let nested_ata_key = &account_keys[0];
        let nested_mint_key = &account_keys[1];
        let wallet_ata_key = &account_keys[2];
        let owner_ata_key = &account_keys[3];
        let owner_mint_key = &account_keys[4];
        let wallet_key = &account_keys[5];
        // account_keys[6] = Token Program

        // Verify wallet is signer
        if !signers[5] {
            return Err(RuntimeError::InvalidSigner);
        }

        // Verify owner_ata is the ATA of wallet for owner_mint
        let (expected_owner_ata, _) = Self::find_program_address(wallet_key, owner_mint_key);
        if *owner_ata_key != expected_owner_ata {
            return Err(RuntimeError::InvalidArgument(
                "Owner ATA doesn't match derived address".to_string()
            ));
        }

        // Verify nested_ata is the ATA of owner_ata for nested_mint
        let (expected_nested_ata, _) = Self::find_program_address(owner_ata_key, nested_mint_key);
        if *nested_ata_key != expected_nested_ata {
            return Err(RuntimeError::InvalidArgument(
                "Nested ATA doesn't match derived address".to_string()
            ));
        }

        // Verify wallet_ata is the ATA of wallet for nested_mint
        let (expected_wallet_ata, _) = Self::find_program_address(wallet_key, nested_mint_key);
        if *wallet_ata_key != expected_wallet_ata {
            return Err(RuntimeError::InvalidArgument(
                "Wallet ATA doesn't match derived address".to_string()
            ));
        }

        // Get nested ATA and transfer all tokens to wallet ATA
        let nested_account = accounts.get(nested_ata_key)
            .ok_or(RuntimeError::AccountNotFound(*nested_ata_key))?;

        let nested_token = TokenAccount::try_from_slice(&nested_account.data)
            .map_err(|_| RuntimeError::InvalidAccountData)?;

        if nested_token.amount == 0 {
            return Ok(()); // Nothing to recover
        }

        let amount = nested_token.amount;

        // Transfer tokens from nested to wallet ATA
        // Update nested ATA (set to 0)
        {
            let nested = accounts.get_mut(nested_ata_key)
                .ok_or(RuntimeError::AccountNotFound(*nested_ata_key))?;
            let mut nested_token = TokenAccount::try_from_slice(&nested.data)
                .map_err(|_| RuntimeError::InvalidAccountData)?;
            nested_token.amount = 0;
            let mut data = vec![];
            nested_token.serialize(&mut data)
                .map_err(|e| RuntimeError::Serialization(e.to_string()))?;
            nested.data = data;
        }

        // Update wallet ATA (add tokens)
        {
            let wallet_ata = accounts.get_mut(wallet_ata_key)
                .ok_or(RuntimeError::AccountNotFound(*wallet_ata_key))?;
            let mut wallet_token = TokenAccount::try_from_slice(&wallet_ata.data)
                .map_err(|_| RuntimeError::InvalidAccountData)?;
            wallet_token.amount = wallet_token.amount.checked_add(amount)
                .ok_or(RuntimeError::Overflow)?;
            let mut data = vec![];
            wallet_token.serialize(&mut data)
                .map_err(|e| RuntimeError::Serialization(e.to_string()))?;
            wallet_ata.data = data;
        }

        Ok(())
    }
}

impl Default for AtaProgram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programs::token::Mint;

    fn create_test_mint(accounts: &mut HashMap<Pubkey, Account>, mint_key: &Pubkey) {
        let mint = Mint {
            mint_authority: None,
            supply: 1_000_000_000,
            decimals: 9,
            is_initialized: true,
            freeze_authority: None,
        };
        let mut data = vec![];
        mint.serialize(&mut data).unwrap();
        accounts.insert(*mint_key, Account {
            lamports: 1_000_000,
            data,
            owner: Pubkey::new(TOKEN_PROGRAM_ID),
            executable: false,
            rent_epoch: 0,
        });
    }

    #[test]
    fn test_derive_ata_address() {
        let wallet = Pubkey::new([1u8; 32]);
        let mint = Pubkey::new([2u8; 32]);

        let ata1 = AtaProgram::get_associated_token_address(&wallet, &mint);
        let ata2 = AtaProgram::get_associated_token_address(&wallet, &mint);

        // Should be deterministic
        assert_eq!(ata1, ata2);

        // Different wallet should give different ATA
        let wallet2 = Pubkey::new([3u8; 32]);
        let ata3 = AtaProgram::get_associated_token_address(&wallet2, &mint);
        assert_ne!(ata1, ata3);
    }

    #[test]
    fn test_create_ata() {
        let program = AtaProgram::new();
        let mut accounts = HashMap::new();

        let payer = Pubkey::new([1u8; 32]);
        let wallet = Pubkey::new([2u8; 32]);
        let mint = Pubkey::new([3u8; 32]);
        let (ata, _) = AtaProgram::find_program_address(&wallet, &mint);
        let system_program = Pubkey::system_program();
        let token_program = Pubkey::new(TOKEN_PROGRAM_ID);

        // Create payer with funds
        accounts.insert(payer, Account {
            lamports: 10_000_000_000, // 10 SOL
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });

        // Create mint
        create_test_mint(&mut accounts, &mint);

        let account_keys = vec![payer, ata, wallet, mint, system_program, token_program];
        let signers = vec![true, false, false, false, false, false];

        // Create ATA
        let result = program.process(&[], &mut accounts, &account_keys, &signers);
        assert!(result.is_ok(), "Failed to create ATA: {:?}", result);

        // Verify ATA was created
        let ata_account = accounts.get(&ata).unwrap();
        assert!(ata_account.lamports > 0);
        assert_eq!(ata_account.owner, token_program);

        // Verify token account data
        let token_account = TokenAccount::try_from_slice(&ata_account.data).unwrap();
        assert_eq!(token_account.mint, mint);
        assert_eq!(token_account.owner, wallet);
        assert_eq!(token_account.amount, 0);
    }

    #[test]
    fn test_create_ata_idempotent() {
        let program = AtaProgram::new();
        let mut accounts = HashMap::new();

        let payer = Pubkey::new([1u8; 32]);
        let wallet = Pubkey::new([2u8; 32]);
        let mint = Pubkey::new([3u8; 32]);
        let (ata, _) = AtaProgram::find_program_address(&wallet, &mint);
        let system_program = Pubkey::system_program();
        let token_program = Pubkey::new(TOKEN_PROGRAM_ID);

        accounts.insert(payer, Account {
            lamports: 10_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        create_test_mint(&mut accounts, &mint);

        let account_keys = vec![payer, ata, wallet, mint, system_program, token_program];
        let signers = vec![true, false, false, false, false, false];

        // Create ATA first time
        let instruction = AtaInstruction::CreateIdempotent;
        let mut data = vec![];
        instruction.serialize(&mut data).unwrap();

        let result = program.process(&data, &mut accounts, &account_keys, &signers);
        assert!(result.is_ok());

        let balance_after_first = accounts.get(&payer).unwrap().lamports;

        // Create again - should succeed without charging
        let result = program.process(&data, &mut accounts, &account_keys, &signers);
        assert!(result.is_ok());

        let balance_after_second = accounts.get(&payer).unwrap().lamports;
        assert_eq!(balance_after_first, balance_after_second, "Should not charge for existing ATA");
    }

    #[test]
    fn test_create_ata_wrong_address() {
        let program = AtaProgram::new();
        let mut accounts = HashMap::new();

        let payer = Pubkey::new([1u8; 32]);
        let wallet = Pubkey::new([2u8; 32]);
        let mint = Pubkey::new([3u8; 32]);
        let wrong_ata = Pubkey::new([99u8; 32]); // Wrong address
        let system_program = Pubkey::system_program();
        let token_program = Pubkey::new(TOKEN_PROGRAM_ID);

        accounts.insert(payer, Account {
            lamports: 10_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        create_test_mint(&mut accounts, &mint);

        let account_keys = vec![payer, wrong_ata, wallet, mint, system_program, token_program];
        let signers = vec![true, false, false, false, false, false];

        let result = program.process(&[], &mut accounts, &account_keys, &signers);
        assert!(result.is_err());
        assert!(matches!(result, Err(RuntimeError::InvalidArgument(_))));
    }

    #[test]
    fn test_insufficient_funds() {
        let program = AtaProgram::new();
        let mut accounts = HashMap::new();

        let payer = Pubkey::new([1u8; 32]);
        let wallet = Pubkey::new([2u8; 32]);
        let mint = Pubkey::new([3u8; 32]);
        let (ata, _) = AtaProgram::find_program_address(&wallet, &mint);
        let system_program = Pubkey::system_program();
        let token_program = Pubkey::new(TOKEN_PROGRAM_ID);

        // Payer with insufficient funds
        accounts.insert(payer, Account {
            lamports: 1000, // Not enough
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        });
        create_test_mint(&mut accounts, &mint);

        let account_keys = vec![payer, ata, wallet, mint, system_program, token_program];
        let signers = vec![true, false, false, false, false, false];

        let result = program.process(&[], &mut accounts, &account_keys, &signers);
        assert!(result.is_err());
        assert!(matches!(result, Err(RuntimeError::InsufficientFunds)));
    }
}
