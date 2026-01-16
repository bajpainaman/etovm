//! SPL Token Program
//!
//! Handles fungible token operations:
//! - Initialize mints and token accounts
//! - Transfer tokens between accounts
//! - Mint and burn tokens
//! - Approve and revoke delegates
//! - Freeze and thaw accounts

use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};

/// Token program ID
pub const TOKEN_PROGRAM_ID: [u8; 32] = [
    0x06, 0xdd, 0xf6, 0xe1, 0xd7, 0x65, 0xa1, 0x93,
    0xd9, 0xcb, 0xe1, 0x46, 0xce, 0xeb, 0x79, 0xac,
    0x1c, 0xb4, 0x85, 0xed, 0x5f, 0x5b, 0x37, 0x91,
    0x3a, 0x8c, 0xf5, 0x85, 0x7e, 0xff, 0x00, 0xa9,
]; // TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA

/// Mint account state - defines a token type
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct Mint {
    /// Optional authority to mint new tokens
    pub mint_authority: Option<Pubkey>,
    /// Total supply of tokens
    pub supply: u64,
    /// Number of decimals (e.g., 9 for SOL-like, 6 for USDC-like)
    pub decimals: u8,
    /// Is this mint initialized?
    pub is_initialized: bool,
    /// Optional authority to freeze token accounts
    pub freeze_authority: Option<Pubkey>,
}

impl Default for Mint {
    fn default() -> Self {
        Self {
            mint_authority: None,
            supply: 0,
            decimals: 0,
            is_initialized: false,
            freeze_authority: None,
        }
    }
}

impl Mint {
    pub const LEN: usize = 82; // Approximate size for rent calculation

    pub fn unpack(data: &[u8]) -> RuntimeResult<Self> {
        if data.is_empty() {
            return Err(RuntimeError::InvalidAccountData);
        }
        borsh::from_slice(data).map_err(|_| RuntimeError::InvalidAccountData)
    }

    pub fn pack(&self) -> Vec<u8> {
        borsh::to_vec(self).unwrap_or_default()
    }
}

/// Token account state - holds tokens for an owner
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct TokenAccount {
    /// The mint this account holds tokens for
    pub mint: Pubkey,
    /// Owner of this token account
    pub owner: Pubkey,
    /// Amount of tokens held
    pub amount: u64,
    /// Optional delegate approved to transfer tokens
    pub delegate: Option<Pubkey>,
    /// Account state
    pub state: AccountState,
    /// If delegate is set, max amount they can transfer
    pub delegated_amount: u64,
    /// Optional authority that can close this account
    pub close_authority: Option<Pubkey>,
}

impl Default for TokenAccount {
    fn default() -> Self {
        Self {
            mint: Pubkey::default(),
            owner: Pubkey::default(),
            amount: 0,
            delegate: None,
            state: AccountState::Uninitialized,
            delegated_amount: 0,
            close_authority: None,
        }
    }
}

impl TokenAccount {
    pub const LEN: usize = 165; // Approximate size for rent calculation

    pub fn unpack(data: &[u8]) -> RuntimeResult<Self> {
        if data.is_empty() {
            return Err(RuntimeError::InvalidAccountData);
        }
        borsh::from_slice(data).map_err(|_| RuntimeError::InvalidAccountData)
    }

    pub fn pack(&self) -> Vec<u8> {
        borsh::to_vec(self).unwrap_or_default()
    }

    pub fn is_frozen(&self) -> bool {
        self.state == AccountState::Frozen
    }

    pub fn is_initialized(&self) -> bool {
        self.state != AccountState::Uninitialized
    }
}

/// Token account state
#[derive(Clone, Copy, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum AccountState {
    /// Account is not yet initialized
    Uninitialized,
    /// Account is initialized and active
    Initialized,
    /// Account is frozen by freeze authority
    Frozen,
}

impl Default for AccountState {
    fn default() -> Self {
        AccountState::Uninitialized
    }
}

/// Authority types for SetAuthority instruction
#[derive(Clone, Copy, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub enum AuthorityType {
    /// Authority to mint new tokens
    MintTokens,
    /// Authority to freeze token accounts
    FreezeAccount,
    /// Owner of a token account
    AccountOwner,
    /// Authority to close a token account
    CloseAccount,
}

/// Token program instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum TokenInstruction {
    /// Initialize a new mint
    /// Accounts:
    /// 0. `[writable]` Mint account to initialize
    /// 1. `[]` Rent sysvar
    InitializeMint {
        decimals: u8,
        mint_authority: Pubkey,
        freeze_authority: Option<Pubkey>,
    },

    /// Initialize a new token account
    /// Accounts:
    /// 0. `[writable]` Token account to initialize
    /// 1. `[]` Mint account
    /// 2. `[]` Owner of the new account
    /// 3. `[]` Rent sysvar
    InitializeAccount,

    /// Transfer tokens
    /// Accounts:
    /// 0. `[writable]` Source token account
    /// 1. `[writable]` Destination token account
    /// 2. `[signer]` Source account owner or delegate
    Transfer { amount: u64 },

    /// Approve a delegate
    /// Accounts:
    /// 0. `[writable]` Token account
    /// 1. `[]` Delegate
    /// 2. `[signer]` Token account owner
    Approve { amount: u64 },

    /// Revoke delegate
    /// Accounts:
    /// 0. `[writable]` Token account
    /// 1. `[signer]` Token account owner
    Revoke,

    /// Set a new authority
    /// Accounts:
    /// 0. `[writable]` Mint or token account
    /// 1. `[signer]` Current authority
    SetAuthority {
        authority_type: AuthorityType,
        new_authority: Option<Pubkey>,
    },

    /// Mint new tokens
    /// Accounts:
    /// 0. `[writable]` Mint account
    /// 1. `[writable]` Destination token account
    /// 2. `[signer]` Mint authority
    MintTo { amount: u64 },

    /// Burn tokens
    /// Accounts:
    /// 0. `[writable]` Token account to burn from
    /// 1. `[writable]` Mint account
    /// 2. `[signer]` Token account owner or delegate
    Burn { amount: u64 },

    /// Close a token account
    /// Accounts:
    /// 0. `[writable]` Token account to close
    /// 1. `[writable]` Destination for remaining lamports
    /// 2. `[signer]` Token account owner
    CloseAccount,

    /// Freeze a token account
    /// Accounts:
    /// 0. `[writable]` Token account to freeze
    /// 1. `[]` Mint account
    /// 2. `[signer]` Freeze authority
    FreezeAccount,

    /// Thaw a frozen token account
    /// Accounts:
    /// 0. `[writable]` Token account to thaw
    /// 1. `[]` Mint account
    /// 2. `[signer]` Freeze authority
    ThawAccount,

    /// Transfer with checked decimals
    /// Accounts: same as Transfer
    TransferChecked { amount: u64, decimals: u8 },

    /// Mint with checked decimals
    /// Accounts: same as MintTo
    MintToChecked { amount: u64, decimals: u8 },

    /// Burn with checked decimals
    /// Accounts: same as Burn
    BurnChecked { amount: u64, decimals: u8 },

    /// Initialize a multisig account (simplified)
    /// Accounts:
    /// 0. `[writable]` Multisig account
    /// 1..n. `[]` Signer accounts
    InitializeMultisig { m: u8 },

    /// Sync native token account balance
    /// Accounts:
    /// 0. `[writable]` Native token account
    SyncNative,
}

/// SPL Token Program
pub struct TokenProgram;

impl TokenProgram {
    pub fn new() -> Self {
        Self
    }

    pub fn program_id() -> Pubkey {
        Pubkey::new(TOKEN_PROGRAM_ID)
    }

    /// Process a token instruction
    pub fn process(
        &self,
        instruction_data: &[u8],
        accounts: &mut [(Pubkey, Account)],
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        let instruction: TokenInstruction =
            borsh::from_slice(instruction_data).map_err(|_| RuntimeError::InvalidInstruction("Invalid token instruction".to_string()))?;

        match instruction {
            TokenInstruction::InitializeMint {
                decimals,
                mint_authority,
                freeze_authority,
            } => self.process_initialize_mint(accounts, decimals, mint_authority, freeze_authority),

            TokenInstruction::InitializeAccount => self.process_initialize_account(accounts),

            TokenInstruction::Transfer { amount } => {
                self.process_transfer(accounts, amount, signers)
            }

            TokenInstruction::Approve { amount } => {
                self.process_approve(accounts, amount, signers)
            }

            TokenInstruction::Revoke => self.process_revoke(accounts, signers),

            TokenInstruction::SetAuthority {
                authority_type,
                new_authority,
            } => self.process_set_authority(accounts, authority_type, new_authority, signers),

            TokenInstruction::MintTo { amount } => self.process_mint_to(accounts, amount, signers),

            TokenInstruction::Burn { amount } => self.process_burn(accounts, amount, signers),

            TokenInstruction::CloseAccount => self.process_close_account(accounts, signers),

            TokenInstruction::FreezeAccount => self.process_freeze_account(accounts, signers),

            TokenInstruction::ThawAccount => self.process_thaw_account(accounts, signers),

            TokenInstruction::TransferChecked { amount, decimals } => {
                self.process_transfer_checked(accounts, amount, decimals, signers)
            }

            TokenInstruction::MintToChecked { amount, decimals } => {
                self.process_mint_to_checked(accounts, amount, decimals, signers)
            }

            TokenInstruction::BurnChecked { amount, decimals } => {
                self.process_burn_checked(accounts, amount, decimals, signers)
            }

            TokenInstruction::InitializeMultisig { m } => {
                self.process_initialize_multisig(accounts, m)
            }

            TokenInstruction::SyncNative => self.process_sync_native(accounts),
        }
    }

    fn process_initialize_mint(
        &self,
        accounts: &mut [(Pubkey, Account)],
        decimals: u8,
        mint_authority: Pubkey,
        freeze_authority: Option<Pubkey>,
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_account = &mut accounts[0].1;

        // Check not already initialized
        if !mint_account.data.is_empty() {
            let existing = Mint::unpack(&mint_account.data)?;
            if existing.is_initialized {
                return Err(RuntimeError::AccountAlreadyInitialized);
            }
        }

        let mint = Mint {
            mint_authority: Some(mint_authority),
            supply: 0,
            decimals,
            is_initialized: true,
            freeze_authority,
        };

        mint_account.data = mint.pack();
        Ok(())
    }

    fn process_initialize_account(&self, accounts: &mut [(Pubkey, Account)]) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_pubkey = accounts[1].0;
        let owner_pubkey = accounts[2].0;

        // Verify mint is initialized
        let mint = Mint::unpack(&accounts[1].1.data)?;
        if !mint.is_initialized {
            return Err(RuntimeError::UninitializedAccount);
        }

        let token_account = &mut accounts[0].1;

        // Check not already initialized
        if !token_account.data.is_empty() && token_account.data.len() >= TokenAccount::LEN {
            let existing = TokenAccount::unpack(&token_account.data)?;
            if existing.is_initialized() {
                return Err(RuntimeError::AccountAlreadyInitialized);
            }
        }

        let account = TokenAccount {
            mint: mint_pubkey,
            owner: owner_pubkey,
            amount: 0,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        token_account.data = account.pack();
        Ok(())
    }

    fn process_transfer(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        // Extract data to avoid borrow conflicts
        let source_data = accounts[0].1.data.clone();
        let dest_data = accounts[1].1.data.clone();

        let mut source = TokenAccount::unpack(&source_data)?;
        let mut dest = TokenAccount::unpack(&dest_data)?;

        // Verify same mint
        if source.mint != dest.mint {
            return Err(RuntimeError::InvalidMint);
        }

        // Verify source is not frozen
        if source.is_frozen() {
            return Err(RuntimeError::AccountFrozen);
        }

        // Verify destination is not frozen
        if dest.is_frozen() {
            return Err(RuntimeError::AccountFrozen);
        }

        // Verify authority (owner or delegate)
        let authority = if signers.contains(&source.owner) {
            source.owner
        } else if let Some(delegate) = source.delegate {
            if signers.contains(&delegate) && source.delegated_amount >= amount {
                delegate
            } else {
                return Err(RuntimeError::InvalidSigner);
            }
        } else {
            return Err(RuntimeError::InvalidSigner);
        };

        // Check sufficient balance
        if source.amount < amount {
            return Err(RuntimeError::InsufficientFunds);
        }

        // Perform transfer
        source.amount = source.amount.checked_sub(amount).ok_or(RuntimeError::Overflow)?;
        dest.amount = dest.amount.checked_add(amount).ok_or(RuntimeError::Overflow)?;

        // Update delegated amount if delegate was used
        if source.delegate.is_some() && authority != source.owner {
            source.delegated_amount = source
                .delegated_amount
                .checked_sub(amount)
                .ok_or(RuntimeError::Overflow)?;
        }

        accounts[0].1.data = source.pack();
        accounts[1].1.data = dest.pack();

        Ok(())
    }

    fn process_approve(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let delegate_pubkey = accounts[1].0;
        let token_data = accounts[0].1.data.clone();
        let mut token_account = TokenAccount::unpack(&token_data)?;

        // Verify owner signed
        if !signers.contains(&token_account.owner) {
            return Err(RuntimeError::InvalidSigner);
        }

        // Verify not frozen
        if token_account.is_frozen() {
            return Err(RuntimeError::AccountFrozen);
        }

        token_account.delegate = Some(delegate_pubkey);
        token_account.delegated_amount = amount;

        accounts[0].1.data = token_account.pack();
        Ok(())
    }

    fn process_revoke(
        &self,
        accounts: &mut [(Pubkey, Account)],
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let token_data = accounts[0].1.data.clone();
        let mut token_account = TokenAccount::unpack(&token_data)?;

        // Verify owner signed
        if !signers.contains(&token_account.owner) {
            return Err(RuntimeError::InvalidSigner);
        }

        token_account.delegate = None;
        token_account.delegated_amount = 0;

        accounts[0].1.data = token_account.pack();
        Ok(())
    }

    fn process_set_authority(
        &self,
        accounts: &mut [(Pubkey, Account)],
        authority_type: AuthorityType,
        new_authority: Option<Pubkey>,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let account_data = accounts[0].1.data.clone();

        match authority_type {
            AuthorityType::MintTokens | AuthorityType::FreezeAccount => {
                let mut mint = Mint::unpack(&account_data)?;

                let current_authority = match authority_type {
                    AuthorityType::MintTokens => mint.mint_authority,
                    AuthorityType::FreezeAccount => mint.freeze_authority,
                    _ => unreachable!(),
                };

                // Verify current authority signed
                if let Some(auth) = current_authority {
                    if !signers.contains(&auth) {
                        return Err(RuntimeError::InvalidSigner);
                    }
                } else {
                    return Err(RuntimeError::InvalidAuthority);
                }

                match authority_type {
                    AuthorityType::MintTokens => mint.mint_authority = new_authority,
                    AuthorityType::FreezeAccount => mint.freeze_authority = new_authority,
                    _ => unreachable!(),
                }

                accounts[0].1.data = mint.pack();
            }
            AuthorityType::AccountOwner | AuthorityType::CloseAccount => {
                let mut token_account = TokenAccount::unpack(&account_data)?;

                let current_authority = match authority_type {
                    AuthorityType::AccountOwner => Some(token_account.owner),
                    AuthorityType::CloseAccount => token_account.close_authority,
                    _ => unreachable!(),
                };

                // Verify current authority signed
                if let Some(auth) = current_authority {
                    if !signers.contains(&auth) {
                        return Err(RuntimeError::InvalidSigner);
                    }
                } else if authority_type == AuthorityType::CloseAccount {
                    // Default close authority is owner
                    if !signers.contains(&token_account.owner) {
                        return Err(RuntimeError::InvalidSigner);
                    }
                } else {
                    return Err(RuntimeError::InvalidAuthority);
                }

                match authority_type {
                    AuthorityType::AccountOwner => {
                        token_account.owner = new_authority.ok_or(RuntimeError::InvalidAuthority)?
                    }
                    AuthorityType::CloseAccount => token_account.close_authority = new_authority,
                    _ => unreachable!(),
                }

                accounts[0].1.data = token_account.pack();
            }
        }

        Ok(())
    }

    fn process_mint_to(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[0].1.data.clone();
        let dest_data = accounts[1].1.data.clone();

        let mut mint = Mint::unpack(&mint_data)?;
        let mut dest = TokenAccount::unpack(&dest_data)?;

        // Verify mint authority
        let mint_authority = mint.mint_authority.ok_or(RuntimeError::InvalidAuthority)?;
        if !signers.contains(&mint_authority) {
            return Err(RuntimeError::InvalidSigner);
        }

        // Verify destination mint matches
        if dest.mint != accounts[0].0 {
            return Err(RuntimeError::InvalidMint);
        }

        // Verify destination not frozen
        if dest.is_frozen() {
            return Err(RuntimeError::AccountFrozen);
        }

        // Mint tokens
        mint.supply = mint.supply.checked_add(amount).ok_or(RuntimeError::Overflow)?;
        dest.amount = dest.amount.checked_add(amount).ok_or(RuntimeError::Overflow)?;

        accounts[0].1.data = mint.pack();
        accounts[1].1.data = dest.pack();

        Ok(())
    }

    fn process_burn(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let source_data = accounts[0].1.data.clone();
        let mint_data = accounts[1].1.data.clone();

        let mut source = TokenAccount::unpack(&source_data)?;
        let mut mint = Mint::unpack(&mint_data)?;

        // Verify source mint matches
        if source.mint != accounts[1].0 {
            return Err(RuntimeError::InvalidMint);
        }

        // Verify authority (owner or delegate)
        let is_delegate = if signers.contains(&source.owner) {
            false
        } else if let Some(delegate) = source.delegate {
            if signers.contains(&delegate) && source.delegated_amount >= amount {
                true
            } else {
                return Err(RuntimeError::InvalidSigner);
            }
        } else {
            return Err(RuntimeError::InvalidSigner);
        };

        // Verify not frozen
        if source.is_frozen() {
            return Err(RuntimeError::AccountFrozen);
        }

        // Check sufficient balance
        if source.amount < amount {
            return Err(RuntimeError::InsufficientFunds);
        }

        // Burn tokens
        source.amount = source.amount.checked_sub(amount).ok_or(RuntimeError::Overflow)?;
        mint.supply = mint.supply.checked_sub(amount).ok_or(RuntimeError::Overflow)?;

        // Update delegated amount if delegate was used
        if is_delegate {
            source.delegated_amount = source
                .delegated_amount
                .checked_sub(amount)
                .ok_or(RuntimeError::Overflow)?;
        }

        accounts[0].1.data = source.pack();
        accounts[1].1.data = mint.pack();

        Ok(())
    }

    fn process_close_account(
        &self,
        accounts: &mut [(Pubkey, Account)],
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let source_data = accounts[0].1.data.clone();
        let source = TokenAccount::unpack(&source_data)?;

        // Must have zero balance
        if source.amount != 0 {
            return Err(RuntimeError::NonZeroBalance);
        }

        // Verify close authority
        let close_authority = source.close_authority.unwrap_or(source.owner);
        if !signers.contains(&close_authority) {
            return Err(RuntimeError::InvalidSigner);
        }

        // Transfer lamports to destination
        let lamports = accounts[0].1.lamports;
        accounts[1].1.lamports = accounts[1]
            .1
            .lamports
            .checked_add(lamports)
            .ok_or(RuntimeError::Overflow)?;

        // Clear source account
        accounts[0].1.lamports = 0;
        accounts[0].1.data.clear();

        Ok(())
    }

    fn process_freeze_account(
        &self,
        accounts: &mut [(Pubkey, Account)],
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[1].1.data.clone();
        let token_data = accounts[0].1.data.clone();

        let mint = Mint::unpack(&mint_data)?;
        let mut token_account = TokenAccount::unpack(&token_data)?;

        // Verify token account belongs to mint
        if token_account.mint != accounts[1].0 {
            return Err(RuntimeError::InvalidMint);
        }

        // Verify freeze authority
        let freeze_authority = mint.freeze_authority.ok_or(RuntimeError::InvalidAuthority)?;
        if !signers.contains(&freeze_authority) {
            return Err(RuntimeError::InvalidSigner);
        }

        // Freeze the account
        token_account.state = AccountState::Frozen;

        accounts[0].1.data = token_account.pack();
        Ok(())
    }

    fn process_thaw_account(
        &self,
        accounts: &mut [(Pubkey, Account)],
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[1].1.data.clone();
        let token_data = accounts[0].1.data.clone();

        let mint = Mint::unpack(&mint_data)?;
        let mut token_account = TokenAccount::unpack(&token_data)?;

        // Verify token account belongs to mint
        if token_account.mint != accounts[1].0 {
            return Err(RuntimeError::InvalidMint);
        }

        // Verify freeze authority
        let freeze_authority = mint.freeze_authority.ok_or(RuntimeError::InvalidAuthority)?;
        if !signers.contains(&freeze_authority) {
            return Err(RuntimeError::InvalidSigner);
        }

        // Thaw the account
        token_account.state = AccountState::Initialized;

        accounts[0].1.data = token_account.pack();
        Ok(())
    }

    fn process_transfer_checked(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        decimals: u8,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        // TransferChecked includes mint account for decimal verification
        if accounts.len() < 4 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[2].1.data.clone();
        let mint = Mint::unpack(&mint_data)?;

        // Verify decimals match
        if mint.decimals != decimals {
            return Err(RuntimeError::DecimalsMismatch);
        }

        // Reorder accounts for standard transfer: [source, dest, authority]
        // Input is: [source, mint, dest, authority]
        let mut transfer_accounts = vec![
            accounts[0].clone(),
            accounts[2].clone(), // dest is at index 2 for checked
            accounts[3].clone(),
        ];

        self.process_transfer(&mut transfer_accounts, amount, signers)?;

        // Copy back modified accounts
        accounts[0] = transfer_accounts[0].clone();
        accounts[2] = transfer_accounts[1].clone();

        Ok(())
    }

    fn process_mint_to_checked(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        decimals: u8,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[0].1.data.clone();
        let mint = Mint::unpack(&mint_data)?;

        // Verify decimals match
        if mint.decimals != decimals {
            return Err(RuntimeError::DecimalsMismatch);
        }

        self.process_mint_to(accounts, amount, signers)
    }

    fn process_burn_checked(
        &self,
        accounts: &mut [(Pubkey, Account)],
        amount: u64,
        decimals: u8,
        signers: &[Pubkey],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let mint_data = accounts[1].1.data.clone();
        let mint = Mint::unpack(&mint_data)?;

        // Verify decimals match
        if mint.decimals != decimals {
            return Err(RuntimeError::DecimalsMismatch);
        }

        self.process_burn(accounts, amount, signers)
    }

    fn process_initialize_multisig(
        &self,
        accounts: &mut [(Pubkey, Account)],
        m: u8,
    ) -> RuntimeResult<()> {
        // Simplified multisig - just verify we have enough signers
        if accounts.len() < 2 {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let n = (accounts.len() - 1) as u8;
        if m > n || m == 0 {
            return Err(RuntimeError::InvalidMultisigConfig);
        }

        // Store multisig config in account data
        let config = MultisigConfig {
            m,
            n,
            is_initialized: true,
            signers: accounts[1..].iter().map(|(pk, _)| *pk).collect(),
        };

        accounts[0].1.data = borsh::to_vec(&config).unwrap_or_default();
        Ok(())
    }

    fn process_sync_native(&self, accounts: &mut [(Pubkey, Account)]) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::NotEnoughAccounts);
        }

        let token_data = accounts[0].1.data.clone();
        let mut token_account = TokenAccount::unpack(&token_data)?;

        // Sync amount with lamports (for wrapped SOL)
        // This is a simplified version - real implementation checks for native mint
        token_account.amount = accounts[0].1.lamports;

        accounts[0].1.data = token_account.pack();
        Ok(())
    }
}

/// Multisig configuration
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct MultisigConfig {
    pub m: u8,
    pub n: u8,
    pub is_initialized: bool,
    pub signers: Vec<Pubkey>,
}

impl Default for TokenProgram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_account(lamports: u64) -> Account {
        Account {
            lamports,
            data: vec![],
            owner: Pubkey::default(),
            executable: false,
            rent_epoch: 0,
        }
    }

    #[test]
    fn test_initialize_mint() {
        let program = TokenProgram::new();
        let mint_authority = Pubkey::new_unique();
        let mut accounts = vec![(Pubkey::new_unique(), create_test_account(1_000_000))];

        let result = program.process_initialize_mint(&mut accounts, 9, mint_authority, None);
        assert!(result.is_ok());

        let mint = Mint::unpack(&accounts[0].1.data).unwrap();
        assert!(mint.is_initialized);
        assert_eq!(mint.decimals, 9);
        assert_eq!(mint.supply, 0);
        assert_eq!(mint.mint_authority, Some(mint_authority));
    }

    #[test]
    fn test_initialize_account() {
        let program = TokenProgram::new();
        let mint_pubkey = Pubkey::new_unique();
        let owner_pubkey = Pubkey::new_unique();

        // Create initialized mint
        let mint = Mint {
            mint_authority: Some(Pubkey::new_unique()),
            supply: 0,
            decimals: 9,
            is_initialized: true,
            freeze_authority: None,
        };

        let mut mint_account = create_test_account(1_000_000);
        mint_account.data = mint.pack();

        let mut accounts = vec![
            (Pubkey::new_unique(), create_test_account(1_000_000)),
            (mint_pubkey, mint_account),
            (owner_pubkey, create_test_account(0)),
        ];

        let result = program.process_initialize_account(&mut accounts);
        assert!(result.is_ok());

        let token_account = TokenAccount::unpack(&accounts[0].1.data).unwrap();
        assert!(token_account.is_initialized());
        assert_eq!(token_account.mint, mint_pubkey);
        assert_eq!(token_account.owner, owner_pubkey);
        assert_eq!(token_account.amount, 0);
    }

    #[test]
    fn test_mint_to() {
        let program = TokenProgram::new();
        let mint_authority = Pubkey::new_unique();
        let mint_pubkey = Pubkey::new_unique();

        // Create mint
        let mint = Mint {
            mint_authority: Some(mint_authority),
            supply: 0,
            decimals: 9,
            is_initialized: true,
            freeze_authority: None,
        };

        let mut mint_account = create_test_account(1_000_000);
        mint_account.data = mint.pack();

        // Create token account
        let token_account = TokenAccount {
            mint: mint_pubkey,
            owner: Pubkey::new_unique(),
            amount: 0,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut dest_account = create_test_account(1_000_000);
        dest_account.data = token_account.pack();

        let mut accounts = vec![
            (mint_pubkey, mint_account),
            (Pubkey::new_unique(), dest_account),
            (mint_authority, create_test_account(0)),
        ];

        let signers = vec![mint_authority];
        let result = program.process_mint_to(&mut accounts, 1000, &signers);
        assert!(result.is_ok());

        let updated_mint = Mint::unpack(&accounts[0].1.data).unwrap();
        let updated_token = TokenAccount::unpack(&accounts[1].1.data).unwrap();

        assert_eq!(updated_mint.supply, 1000);
        assert_eq!(updated_token.amount, 1000);
    }

    #[test]
    fn test_transfer() {
        let program = TokenProgram::new();
        let mint_pubkey = Pubkey::new_unique();
        let owner = Pubkey::new_unique();

        // Create source token account with balance
        let source = TokenAccount {
            mint: mint_pubkey,
            owner,
            amount: 1000,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut source_account = create_test_account(1_000_000);
        source_account.data = source.pack();

        // Create destination token account
        let dest = TokenAccount {
            mint: mint_pubkey,
            owner: Pubkey::new_unique(),
            amount: 0,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut dest_account = create_test_account(1_000_000);
        dest_account.data = dest.pack();

        let mut accounts = vec![
            (Pubkey::new_unique(), source_account),
            (Pubkey::new_unique(), dest_account),
            (owner, create_test_account(0)),
        ];

        let signers = vec![owner];
        let result = program.process_transfer(&mut accounts, 500, &signers);
        assert!(result.is_ok());

        let updated_source = TokenAccount::unpack(&accounts[0].1.data).unwrap();
        let updated_dest = TokenAccount::unpack(&accounts[1].1.data).unwrap();

        assert_eq!(updated_source.amount, 500);
        assert_eq!(updated_dest.amount, 500);
    }

    #[test]
    fn test_transfer_insufficient_funds() {
        let program = TokenProgram::new();
        let mint_pubkey = Pubkey::new_unique();
        let owner = Pubkey::new_unique();

        let source = TokenAccount {
            mint: mint_pubkey,
            owner,
            amount: 100,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut source_account = create_test_account(1_000_000);
        source_account.data = source.pack();

        let dest = TokenAccount {
            mint: mint_pubkey,
            owner: Pubkey::new_unique(),
            amount: 0,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut dest_account = create_test_account(1_000_000);
        dest_account.data = dest.pack();

        let mut accounts = vec![
            (Pubkey::new_unique(), source_account),
            (Pubkey::new_unique(), dest_account),
            (owner, create_test_account(0)),
        ];

        let signers = vec![owner];
        let result = program.process_transfer(&mut accounts, 500, &signers);
        assert!(matches!(result, Err(RuntimeError::InsufficientFunds)));
    }

    #[test]
    fn test_freeze_thaw() {
        let program = TokenProgram::new();
        let freeze_authority = Pubkey::new_unique();
        let mint_pubkey = Pubkey::new_unique();

        let mint = Mint {
            mint_authority: Some(Pubkey::new_unique()),
            supply: 1000,
            decimals: 9,
            is_initialized: true,
            freeze_authority: Some(freeze_authority),
        };

        let mut mint_account = create_test_account(1_000_000);
        mint_account.data = mint.pack();

        let token = TokenAccount {
            mint: mint_pubkey,
            owner: Pubkey::new_unique(),
            amount: 500,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut token_account = create_test_account(1_000_000);
        token_account.data = token.pack();

        let mut accounts = vec![
            (Pubkey::new_unique(), token_account),
            (mint_pubkey, mint_account),
            (freeze_authority, create_test_account(0)),
        ];

        let signers = vec![freeze_authority];

        // Freeze
        let result = program.process_freeze_account(&mut accounts, &signers);
        assert!(result.is_ok());

        let frozen = TokenAccount::unpack(&accounts[0].1.data).unwrap();
        assert!(frozen.is_frozen());

        // Thaw
        let result = program.process_thaw_account(&mut accounts, &signers);
        assert!(result.is_ok());

        let thawed = TokenAccount::unpack(&accounts[0].1.data).unwrap();
        assert!(!thawed.is_frozen());
    }

    #[test]
    fn test_burn() {
        let program = TokenProgram::new();
        let owner = Pubkey::new_unique();
        let mint_pubkey = Pubkey::new_unique();

        let mint = Mint {
            mint_authority: Some(Pubkey::new_unique()),
            supply: 1000,
            decimals: 9,
            is_initialized: true,
            freeze_authority: None,
        };

        let mut mint_account = create_test_account(1_000_000);
        mint_account.data = mint.pack();

        let token = TokenAccount {
            mint: mint_pubkey,
            owner,
            amount: 500,
            delegate: None,
            state: AccountState::Initialized,
            delegated_amount: 0,
            close_authority: None,
        };

        let mut token_account = create_test_account(1_000_000);
        token_account.data = token.pack();

        let mut accounts = vec![
            (Pubkey::new_unique(), token_account),
            (mint_pubkey, mint_account),
            (owner, create_test_account(0)),
        ];

        let signers = vec![owner];
        let result = program.process_burn(&mut accounts, 200, &signers);
        assert!(result.is_ok());

        let updated_token = TokenAccount::unpack(&accounts[0].1.data).unwrap();
        let updated_mint = Mint::unpack(&accounts[1].1.data).unwrap();

        assert_eq!(updated_token.amount, 300);
        assert_eq!(updated_mint.supply, 800);
    }
}
