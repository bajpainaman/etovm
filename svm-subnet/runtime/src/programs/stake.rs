//! Stake Program
//!
//! Handles stake account operations for validator delegation:
//! - Initialize stake accounts
//! - Delegate stake to validators
//! - Deactivate stake
//! - Withdraw stake
//! - Split/merge stake accounts

use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, AccountMeta, Instruction, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};

/// Stake state stored in stake account data
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum StakeState {
    /// Uninitialized stake account
    Uninitialized,
    /// Initialized but not delegated
    Initialized(Meta),
    /// Active stake delegated to a validator
    Stake(Meta, Stake),
    /// Rewards pool (deprecated but kept for compatibility)
    RewardsPool,
}

impl Default for StakeState {
    fn default() -> Self {
        StakeState::Uninitialized
    }
}

/// Stake account metadata
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct Meta {
    /// Rent exempt reserve (minimum balance)
    pub rent_exempt_reserve: u64,
    /// Authority allowed to deactivate and withdraw
    pub authorized: Authorized,
    /// Lockup configuration
    pub lockup: Lockup,
}

/// Authorized stakers and withdrawers
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct Authorized {
    /// Pubkey authorized to stake
    pub staker: Pubkey,
    /// Pubkey authorized to withdraw
    pub withdrawer: Pubkey,
}

impl Authorized {
    pub fn new(staker: Pubkey, withdrawer: Pubkey) -> Self {
        Self { staker, withdrawer }
    }

    pub fn auto(authorized: &Pubkey) -> Self {
        Self {
            staker: *authorized,
            withdrawer: *authorized,
        }
    }
}

/// Lockup configuration for stake account
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct Lockup {
    /// Unix timestamp after which stake can be withdrawn
    pub unix_timestamp: i64,
    /// Epoch after which stake can be withdrawn
    pub epoch: u64,
    /// Custodian that can modify lockup
    pub custodian: Pubkey,
}

/// Active stake delegation info
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct Stake {
    /// Delegation details
    pub delegation: Delegation,
    /// Credits observed for reward calculation
    pub credits_observed: u64,
}

/// Delegation information
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct Delegation {
    /// Vote account being delegated to
    pub voter_pubkey: Pubkey,
    /// Amount of stake delegated (in lamports)
    pub stake: u64,
    /// Epoch at which delegation became active
    pub activation_epoch: u64,
    /// Epoch at which stake was deactivated (u64::MAX if still active)
    pub deactivation_epoch: u64,
    /// Warmup/cooldown rate
    pub warmup_cooldown_rate: f64,
}

impl Delegation {
    pub fn new(voter_pubkey: &Pubkey, stake: u64, activation_epoch: u64) -> Self {
        Self {
            voter_pubkey: *voter_pubkey,
            stake,
            activation_epoch,
            deactivation_epoch: u64::MAX, // Not deactivated
            warmup_cooldown_rate: 0.25,   // 25% warmup/cooldown per epoch
        }
    }

    pub fn is_deactivating(&self, epoch: u64) -> bool {
        self.deactivation_epoch != u64::MAX && epoch >= self.deactivation_epoch
    }
}

/// Stake program instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum StakeInstruction {
    /// Initialize a stake account with authorized staker/withdrawer
    /// Accounts:
    /// 0. [WRITE] Stake account to initialize
    /// 1. [] Rent sysvar
    Initialize(Authorized, Lockup),

    /// Authorize a new staker or withdrawer
    /// Accounts:
    /// 0. [WRITE] Stake account
    /// 1. [SIGNER] Current authority
    /// 2. [] Clock sysvar (optional)
    Authorize(Pubkey, StakeAuthorize),

    /// Delegate stake to a vote account
    /// Accounts:
    /// 0. [WRITE] Stake account
    /// 1. [] Vote account to delegate to
    /// 2. [] Clock sysvar
    /// 3. [] Stake history sysvar
    /// 4. [] Stake config account
    /// 5. [SIGNER] Stake authority
    DelegateStake,

    /// Split stake account
    /// Accounts:
    /// 0. [WRITE] Source stake account
    /// 1. [WRITE] Destination stake account (uninitialized)
    /// 2. [SIGNER] Stake authority
    Split(u64),

    /// Withdraw from stake account
    /// Accounts:
    /// 0. [WRITE] Stake account
    /// 1. [WRITE] Recipient account
    /// 2. [] Clock sysvar
    /// 3. [] Stake history sysvar
    /// 4. [SIGNER] Withdraw authority
    Withdraw(u64),

    /// Deactivate stake (begin cooldown)
    /// Accounts:
    /// 0. [WRITE] Stake account
    /// 1. [] Clock sysvar
    /// 2. [SIGNER] Stake authority
    Deactivate,

    /// Merge two stake accounts
    /// Accounts:
    /// 0. [WRITE] Destination stake account
    /// 1. [WRITE] Source stake account (will be closed)
    /// 2. [] Clock sysvar
    /// 3. [] Stake history sysvar
    /// 4. [SIGNER] Stake authority
    Merge,
}

/// Type of stake authorization to modify
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum StakeAuthorize {
    Staker,
    Withdrawer,
}

/// Stake program processor
pub struct StakeProgram;

impl StakeProgram {
    /// Minimum stake delegation amount (0.01 SOL equivalent)
    pub const MIN_DELEGATION: u64 = 10_000_000;
    /// Stake account size
    pub const STAKE_ACCOUNT_SIZE: usize = 200;

    pub fn new() -> Self {
        Self
    }

    pub fn process_instruction(
        &self,
        accounts: &mut [(Pubkey, Account)],
        instruction_data: &[u8],
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        let instruction: StakeInstruction = borsh::from_slice(instruction_data)
            .map_err(|e| RuntimeError::Program(format!("Failed to deserialize: {}", e)))?;

        match instruction {
            StakeInstruction::Initialize(authorized, lockup) => {
                self.initialize(accounts, authorized, lockup)
            }
            StakeInstruction::Authorize(pubkey, authorize_type) => {
                self.authorize(accounts, pubkey, authorize_type)
            }
            StakeInstruction::DelegateStake => self.delegate_stake(accounts, current_epoch),
            StakeInstruction::Split(lamports) => self.split(accounts, lamports),
            StakeInstruction::Withdraw(lamports) => {
                self.withdraw(accounts, lamports, current_epoch)
            }
            StakeInstruction::Deactivate => self.deactivate(accounts, current_epoch),
            StakeInstruction::Merge => self.merge(accounts, current_epoch),
        }
    }

    fn initialize(
        &self,
        accounts: &mut [(Pubkey, Account)],
        authorized: Authorized,
        lockup: Lockup,
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let stake_account = &mut accounts[0].1;

        // Check owner
        if stake_account.owner != Pubkey::stake_program() {
            return Err(RuntimeError::Program("Invalid stake account owner".into()));
        }

        // Deserialize current state
        let state: StakeState = if stake_account.data.is_empty() {
            StakeState::Uninitialized
        } else {
            borsh::from_slice(&stake_account.data)
                .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?
        };

        // Must be uninitialized
        if !matches!(state, StakeState::Uninitialized) {
            return Err(RuntimeError::Program("Already initialized".into()));
        }

        // Calculate rent-exempt reserve
        let rent_exempt_reserve = Self::calculate_rent_exempt_reserve();

        if stake_account.lamports < rent_exempt_reserve {
            return Err(RuntimeError::Program("Insufficient funds for rent".into()));
        }

        let meta = Meta {
            rent_exempt_reserve,
            authorized,
            lockup,
        };

        let new_state = StakeState::Initialized(meta);
        stake_account.data = borsh::to_vec(&new_state).unwrap();

        Ok(())
    }

    fn authorize(
        &self,
        accounts: &mut [(Pubkey, Account)],
        new_authority: Pubkey,
        authorize_type: StakeAuthorize,
    ) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let stake_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[1].0;

        let mut state: StakeState = borsh::from_slice(&stake_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        let meta = match &mut state {
            StakeState::Initialized(meta) => meta,
            StakeState::Stake(meta, _) => meta,
            _ => return Err(RuntimeError::Program("Invalid stake state".into())),
        };

        // Verify signer is current authority
        match authorize_type {
            StakeAuthorize::Staker => {
                if *signer_pubkey != meta.authorized.staker {
                    return Err(RuntimeError::Program("Invalid staker authority".into()));
                }
                meta.authorized.staker = new_authority;
            }
            StakeAuthorize::Withdrawer => {
                if *signer_pubkey != meta.authorized.withdrawer {
                    return Err(RuntimeError::Program("Invalid withdraw authority".into()));
                }
                meta.authorized.withdrawer = new_authority;
            }
        }

        stake_account.data = borsh::to_vec(&state).unwrap();
        Ok(())
    }

    fn delegate_stake(
        &self,
        accounts: &mut [(Pubkey, Account)],
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 6 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Extract data to avoid borrow conflicts
        let stake_data = accounts[0].1.data.clone();
        let stake_lamports = accounts[0].1.lamports;
        let vote_pubkey = accounts[1].0;
        let signer_pubkey = accounts[5].0;

        let state: StakeState = borsh::from_slice(&stake_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        let meta = match &state {
            StakeState::Initialized(meta) => meta.clone(),
            StakeState::Stake(meta, _) => meta.clone(), // Re-delegation
            _ => return Err(RuntimeError::Program("Invalid stake state".into())),
        };

        // Verify staker authority
        if signer_pubkey != meta.authorized.staker {
            return Err(RuntimeError::Program("Invalid staker authority".into()));
        }

        // Calculate stake amount
        let stake_amount = stake_lamports
            .checked_sub(meta.rent_exempt_reserve)
            .ok_or_else(|| RuntimeError::Program("Insufficient balance".into()))?;

        if stake_amount < Self::MIN_DELEGATION {
            return Err(RuntimeError::Program(format!(
                "Stake amount {} below minimum {}",
                stake_amount,
                Self::MIN_DELEGATION
            )));
        }

        let delegation = Delegation::new(&vote_pubkey, stake_amount, current_epoch);
        let stake = Stake {
            delegation,
            credits_observed: 0,
        };

        let new_state = StakeState::Stake(meta, stake);
        accounts[0].1.data = borsh::to_vec(&new_state).unwrap();

        Ok(())
    }

    fn split(
        &self,
        accounts: &mut [(Pubkey, Account)],
        lamports: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Extract data we need first
        let signer_pubkey = accounts[2].0;
        let source_data = accounts[0].1.data.clone();
        let source_balance = accounts[0].1.lamports;

        let state: StakeState = borsh::from_slice(&source_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        let (meta, stake_opt) = match &state {
            StakeState::Initialized(meta) => (meta.clone(), None),
            StakeState::Stake(meta, stake) => (meta.clone(), Some(stake.clone())),
            _ => return Err(RuntimeError::Program("Invalid stake state".into())),
        };

        // Verify authority
        if signer_pubkey != meta.authorized.staker {
            return Err(RuntimeError::Program("Invalid staker authority".into()));
        }

        // Check sufficient balance
        let min_split = meta.rent_exempt_reserve + Self::MIN_DELEGATION;

        if lamports < min_split {
            return Err(RuntimeError::Program("Split amount too small".into()));
        }

        if source_balance < lamports + min_split {
            return Err(RuntimeError::Program("Insufficient balance for split".into()));
        }

        // Initialize destination with same state
        let dest_meta = Meta {
            rent_exempt_reserve: meta.rent_exempt_reserve,
            authorized: meta.authorized.clone(),
            lockup: meta.lockup.clone(),
        };

        let dest_state = if let Some(mut stake) = stake_opt {
            let split_ratio = lamports as f64 / source_balance as f64;
            stake.delegation.stake = (stake.delegation.stake as f64 * split_ratio) as u64;
            StakeState::Stake(dest_meta, stake)
        } else {
            StakeState::Initialized(dest_meta)
        };

        // Perform split using indexing to avoid borrow conflicts
        accounts[0].1.lamports -= lamports;
        accounts[1].1.lamports += lamports;
        accounts[1].1.data = borsh::to_vec(&dest_state).unwrap();
        accounts[1].1.owner = Pubkey::stake_program();

        Ok(())
    }

    fn withdraw(
        &self,
        accounts: &mut [(Pubkey, Account)],
        lamports: u64,
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 5 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Extract data first to avoid borrow conflicts
        let stake_data = accounts[0].1.data.clone();
        let stake_balance = accounts[0].1.lamports;
        let signer_pubkey = accounts[4].0;

        let state: StakeState = borsh::from_slice(&stake_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        let (meta, withdrawable) = match &state {
            StakeState::Uninitialized => {
                // Fully withdrawable
                accounts[0].1.lamports -= lamports;
                accounts[1].1.lamports += lamports;
                return Ok(());
            }
            StakeState::Initialized(meta) => {
                // Everything above rent reserve is withdrawable
                let reserve = meta.rent_exempt_reserve;
                let withdrawable = stake_balance.saturating_sub(reserve);
                (meta.clone(), withdrawable)
            }
            StakeState::Stake(meta, stake) => {
                // Check if stake is fully deactivated
                if stake.delegation.deactivation_epoch == u64::MAX {
                    return Err(RuntimeError::Program("Stake is still active".into()));
                }

                // Simplified: require full cooldown (deactivation_epoch + 1)
                if current_epoch <= stake.delegation.deactivation_epoch {
                    return Err(RuntimeError::Program("Stake is cooling down".into()));
                }

                let reserve = meta.rent_exempt_reserve;
                let withdrawable = stake_balance.saturating_sub(reserve);
                (meta.clone(), withdrawable)
            }
            _ => return Err(RuntimeError::Program("Invalid stake state".into())),
        };

        // Verify withdraw authority
        if signer_pubkey != meta.authorized.withdrawer {
            return Err(RuntimeError::Program("Invalid withdraw authority".into()));
        }

        // Check lockup
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        if meta.lockup.unix_timestamp > now || meta.lockup.epoch > current_epoch {
            if signer_pubkey != meta.lockup.custodian {
                return Err(RuntimeError::Program("Stake is locked".into()));
            }
        }

        if lamports > withdrawable {
            return Err(RuntimeError::Program(format!(
                "Withdrawal amount {} exceeds withdrawable {}",
                lamports, withdrawable
            )));
        }

        // Perform withdrawal
        accounts[0].1.lamports -= lamports;
        accounts[1].1.lamports += lamports;

        Ok(())
    }

    fn deactivate(
        &self,
        accounts: &mut [(Pubkey, Account)],
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let stake_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[2].0;

        let mut state: StakeState = borsh::from_slice(&stake_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        let (meta, stake) = match &mut state {
            StakeState::Stake(meta, stake) => (meta, stake),
            _ => return Err(RuntimeError::Program("Stake is not delegated".into())),
        };

        // Verify staker authority
        if *signer_pubkey != meta.authorized.staker {
            return Err(RuntimeError::Program("Invalid staker authority".into()));
        }

        // Already deactivating?
        if stake.delegation.deactivation_epoch != u64::MAX {
            return Err(RuntimeError::Program("Already deactivating".into()));
        }

        // Set deactivation epoch
        stake.delegation.deactivation_epoch = current_epoch;
        stake_account.data = borsh::to_vec(&state).unwrap();

        Ok(())
    }

    fn merge(
        &self,
        accounts: &mut [(Pubkey, Account)],
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 5 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Extract data to avoid borrow conflicts
        let dest_data = accounts[0].1.data.clone();
        let source_data = accounts[1].1.data.clone();
        let source_lamports = accounts[1].1.lamports;
        let signer_pubkey = accounts[4].0;

        let dest_state: StakeState = borsh::from_slice(&dest_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid dest state: {}", e)))?;

        let source_state: StakeState = borsh::from_slice(&source_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid source state: {}", e)))?;

        // Both must be Stake state with same voter
        let (dest_meta, dest_stake) = match &dest_state {
            StakeState::Stake(meta, stake) => (meta.clone(), stake.clone()),
            _ => return Err(RuntimeError::Program("Dest must be delegated".into())),
        };

        let (source_meta, source_stake) = match &source_state {
            StakeState::Stake(meta, stake) => (meta.clone(), stake.clone()),
            _ => return Err(RuntimeError::Program("Source must be delegated".into())),
        };

        // Verify same voter
        if dest_stake.delegation.voter_pubkey != source_stake.delegation.voter_pubkey {
            return Err(RuntimeError::Program("Different validators".into()));
        }

        // Verify authority
        if signer_pubkey != dest_meta.authorized.staker
            || signer_pubkey != source_meta.authorized.staker
        {
            return Err(RuntimeError::Program("Invalid staker authority".into()));
        }

        // Neither can be deactivating
        if dest_stake.delegation.is_deactivating(current_epoch)
            || source_stake.delegation.is_deactivating(current_epoch)
        {
            return Err(RuntimeError::Program("Cannot merge deactivating stake".into()));
        }

        // Merge
        let merged_stake = dest_stake.delegation.stake + source_stake.delegation.stake;
        let mut new_stake = dest_stake.clone();
        new_stake.delegation.stake = merged_stake;

        let new_state = StakeState::Stake(dest_meta, new_stake);
        let uninitialized_state = StakeState::Uninitialized;

        // Apply changes using indexing
        accounts[0].1.lamports += source_lamports;
        accounts[1].1.lamports = 0;
        accounts[0].1.data = borsh::to_vec(&new_state).unwrap();
        accounts[1].1.data = borsh::to_vec(&uninitialized_state).unwrap();

        Ok(())
    }

    fn calculate_rent_exempt_reserve() -> u64 {
        // Based on Solana's rent calculation: 2 years of rent
        // bytes * lamports_per_byte_year * 2
        Self::STAKE_ACCOUNT_SIZE as u64 * 3480 * 2
    }
}

impl Default for StakeProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create stake instructions
pub mod instruction {
    use super::*;

    pub fn initialize(
        stake_pubkey: &Pubkey,
        authorized: &Authorized,
        lockup: &Lockup,
    ) -> Instruction {
        let data = borsh::to_vec(&StakeInstruction::Initialize(
            authorized.clone(),
            lockup.clone(),
        ))
        .unwrap();

        Instruction {
            program_id: Pubkey::stake_program(),
            accounts: vec![
                AccountMeta::new(*stake_pubkey, false),
                AccountMeta::new_readonly(Pubkey::sysvar_rent(), false),
            ],
            data,
        }
    }

    pub fn delegate_stake(
        stake_pubkey: &Pubkey,
        vote_pubkey: &Pubkey,
        authority: &Pubkey,
    ) -> Instruction {
        let data = borsh::to_vec(&StakeInstruction::DelegateStake).unwrap();

        Instruction {
            program_id: Pubkey::stake_program(),
            accounts: vec![
                AccountMeta::new(*stake_pubkey, false),
                AccountMeta::new_readonly(*vote_pubkey, false),
                AccountMeta::new_readonly(Pubkey::sysvar_clock(), false),
                AccountMeta::new_readonly(Pubkey::sysvar_stake_history(), false),
                AccountMeta::new_readonly(Pubkey::stake_config(), false),
                AccountMeta::new_readonly(*authority, true),
            ],
            data,
        }
    }

    pub fn deactivate(stake_pubkey: &Pubkey, authority: &Pubkey) -> Instruction {
        let data = borsh::to_vec(&StakeInstruction::Deactivate).unwrap();

        Instruction {
            program_id: Pubkey::stake_program(),
            accounts: vec![
                AccountMeta::new(*stake_pubkey, false),
                AccountMeta::new_readonly(Pubkey::sysvar_clock(), false),
                AccountMeta::new_readonly(*authority, true),
            ],
            data,
        }
    }

    pub fn withdraw(
        stake_pubkey: &Pubkey,
        recipient: &Pubkey,
        authority: &Pubkey,
        lamports: u64,
    ) -> Instruction {
        let data = borsh::to_vec(&StakeInstruction::Withdraw(lamports)).unwrap();

        Instruction {
            program_id: Pubkey::stake_program(),
            accounts: vec![
                AccountMeta::new(*stake_pubkey, false),
                AccountMeta::new(*recipient, false),
                AccountMeta::new_readonly(Pubkey::sysvar_clock(), false),
                AccountMeta::new_readonly(Pubkey::sysvar_stake_history(), false),
                AccountMeta::new_readonly(*authority, true),
            ],
            data,
        }
    }

    pub fn split(
        source: &Pubkey,
        dest: &Pubkey,
        authority: &Pubkey,
        lamports: u64,
    ) -> Instruction {
        let data = borsh::to_vec(&StakeInstruction::Split(lamports)).unwrap();

        Instruction {
            program_id: Pubkey::stake_program(),
            accounts: vec![
                AccountMeta::new(*source, false),
                AccountMeta::new(*dest, false),
                AccountMeta::new_readonly(*authority, true),
            ],
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stake_state_serialization() {
        let authorized = Authorized::new(
            Pubkey::new([1u8; 32]),
            Pubkey::new([2u8; 32]),
        );
        let lockup = Lockup::default();
        let meta = Meta {
            rent_exempt_reserve: 1_000_000,
            authorized,
            lockup,
        };

        let state = StakeState::Initialized(meta);
        let bytes = borsh::to_vec(&state).unwrap();
        let decoded: StakeState = borsh::from_slice(&bytes).unwrap();

        match decoded {
            StakeState::Initialized(m) => {
                assert_eq!(m.rent_exempt_reserve, 1_000_000);
            }
            _ => panic!("Wrong state type"),
        }
    }

    #[test]
    fn test_delegation_deactivating() {
        let mut delegation = Delegation::new(&Pubkey::default(), 1000, 10);
        assert!(!delegation.is_deactivating(15));

        delegation.deactivation_epoch = 20;
        assert!(!delegation.is_deactivating(15));
        assert!(delegation.is_deactivating(20));
        assert!(delegation.is_deactivating(25));
    }
}
