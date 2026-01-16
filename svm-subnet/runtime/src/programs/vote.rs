//! Vote Program
//!
//! Handles validator vote account operations:
//! - Initialize vote accounts
//! - Record votes for slots
//! - Update validator identity/commission
//! - Withdraw rewards

use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, AccountMeta, Instruction, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};
use std::collections::VecDeque;

/// Maximum number of votes to keep in history
pub const MAX_LOCKOUT_HISTORY: usize = 31;
/// Maximum number of epoch credits entries
pub const MAX_EPOCH_CREDITS_HISTORY: usize = 64;
/// Default commission rate (percentage)
pub const DEFAULT_COMMISSION: u8 = 100;

/// Vote state stored in vote account data
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct VoteState {
    /// Node identity (validator pubkey)
    pub node_pubkey: Pubkey,
    /// Authority to submit votes
    pub authorized_voter: Pubkey,
    /// Authority to withdraw rewards
    pub authorized_withdrawer: Pubkey,
    /// Commission percentage (0-100)
    pub commission: u8,
    /// History of votes
    pub votes: VecDeque<Lockout>,
    /// Most recent slot voted on
    pub root_slot: Option<u64>,
    /// Epoch credits: (epoch, credits, prev_credits)
    pub epoch_credits: Vec<(u64, u64, u64)>,
    /// Most recent timestamp from vote
    pub last_timestamp: BlockTimestamp,
}

impl Default for VoteState {
    fn default() -> Self {
        Self {
            node_pubkey: Pubkey::default(),
            authorized_voter: Pubkey::default(),
            authorized_withdrawer: Pubkey::default(),
            commission: DEFAULT_COMMISSION,
            votes: VecDeque::with_capacity(MAX_LOCKOUT_HISTORY),
            root_slot: None,
            epoch_credits: Vec::with_capacity(MAX_EPOCH_CREDITS_HISTORY),
            last_timestamp: BlockTimestamp::default(),
        }
    }
}

impl VoteState {
    pub fn new(init: &VoteInit) -> Self {
        Self {
            node_pubkey: init.node_pubkey,
            authorized_voter: init.authorized_voter,
            authorized_withdrawer: init.authorized_withdrawer,
            commission: init.commission,
            ..Default::default()
        }
    }

    /// Process votes and update state
    pub fn process_vote(&mut self, vote: Vote, current_epoch: u64) {
        let slots_len = vote.slots.len();
        let last_slot = vote.slots.last().copied();

        for slot in &vote.slots {
            self.process_slot_vote(*slot);
        }

        // Update timestamp if provided
        if let Some(timestamp) = vote.timestamp {
            self.last_timestamp = BlockTimestamp {
                slot: last_slot.unwrap_or(0),
                timestamp,
            };
        }

        // Record credits for epoch
        self.increment_credits(current_epoch, slots_len as u64);
    }

    fn process_slot_vote(&mut self, slot: u64) {
        // Check for duplicate votes
        if self.votes.iter().any(|v| v.slot == slot) {
            return;
        }

        // Check if slot is older than our most recent vote
        if let Some(top) = self.votes.back() {
            if slot <= top.slot {
                return; // Can't vote on older slots
            }
        }

        // Pop expired lockouts from the front
        while self.votes.len() >= MAX_LOCKOUT_HISTORY {
            self.root_slot = self.votes.pop_front().map(|v| v.slot);
        }

        // Increase confirmation counts for existing votes (Tower BFT)
        for lockout in self.votes.iter_mut() {
            lockout.confirmation_count = lockout.confirmation_count.saturating_add(1);
        }

        // Add new vote
        self.votes.push_back(Lockout {
            slot,
            confirmation_count: 1,
        });
    }

    fn increment_credits(&mut self, epoch: u64, credits: u64) {
        if self.epoch_credits.is_empty() {
            self.epoch_credits.push((epoch, credits, 0));
            return;
        }

        let last_idx = self.epoch_credits.len() - 1;
        let (last_epoch, _, _) = self.epoch_credits[last_idx];

        if last_epoch == epoch {
            self.epoch_credits[last_idx].1 += credits;
        } else {
            let prev_credits = self.epoch_credits[last_idx].1;
            self.epoch_credits.push((epoch, credits, prev_credits));

            // Trim history
            while self.epoch_credits.len() > MAX_EPOCH_CREDITS_HISTORY {
                self.epoch_credits.remove(0);
            }
        }
    }

    pub fn credits(&self) -> u64 {
        self.epoch_credits.last().map(|(_, c, _)| *c).unwrap_or(0)
    }
}

/// Vote lockout information
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct Lockout {
    pub slot: u64,
    pub confirmation_count: u32,
}

impl Lockout {
    /// Get the lockout period (exponential backoff)
    pub fn lockout(&self) -> u64 {
        2u64.pow(self.confirmation_count)
    }

    /// Check if still locked out at given slot
    pub fn is_locked_out_at_slot(&self, slot: u64) -> bool {
        self.slot + self.lockout() >= slot
    }

    /// Get the last slot this lockout applies to
    pub fn last_locked_out_slot(&self) -> u64 {
        self.slot + self.lockout()
    }
}

/// Block timestamp for vote timing
#[derive(Clone, Debug, Default, BorshSerialize, BorshDeserialize)]
pub struct BlockTimestamp {
    pub slot: u64,
    pub timestamp: i64,
}

/// Vote initialization data
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct VoteInit {
    pub node_pubkey: Pubkey,
    pub authorized_voter: Pubkey,
    pub authorized_withdrawer: Pubkey,
    pub commission: u8,
}

/// A vote transaction
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct Vote {
    /// Slots being voted on
    pub slots: Vec<u64>,
    /// Bank hash of the voted slot (optional for compact votes)
    pub hash: [u8; 32],
    /// Timestamp of vote (optional)
    pub timestamp: Option<i64>,
}

/// Vote program instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum VoteInstruction {
    /// Initialize a vote account
    /// Accounts:
    /// 0. [WRITE] Vote account to initialize
    /// 1. [] Rent sysvar
    /// 2. [] Clock sysvar
    /// 3. [SIGNER] Node identity
    Initialize(VoteInit),

    /// Authorize a new voter or withdrawer
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [] Clock sysvar
    /// 2. [SIGNER] Current authority
    Authorize(Pubkey, VoteAuthorize),

    /// Submit a vote
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [] Slot hashes sysvar
    /// 2. [] Clock sysvar
    /// 3. [SIGNER] Vote authority
    Vote(Vote),

    /// Withdraw from vote account
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [WRITE] Recipient
    /// 2. [SIGNER] Withdraw authority
    Withdraw(u64),

    /// Update validator identity
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [SIGNER] New identity
    /// 2. [SIGNER] Withdraw authority
    UpdateValidatorIdentity,

    /// Update commission
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [SIGNER] Withdraw authority
    UpdateCommission(u8),

    /// Compact vote with slots and hash (v0.23.5+)
    /// Accounts:
    /// 0. [WRITE] Vote account
    /// 1. [SIGNER] Vote authority
    CompactVoteStateUpdate(CompactVoteStateUpdate),
}

/// Type of vote authorization to modify
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum VoteAuthorize {
    Voter,
    Withdrawer,
}

/// Compact vote state update (for efficient vote transactions)
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct CompactVoteStateUpdate {
    pub root: u64,
    pub lockouts: Vec<Lockout>,
    pub hash: [u8; 32],
    pub timestamp: Option<i64>,
}

/// Vote program processor
pub struct VoteProgram;

impl VoteProgram {
    /// Vote account minimum size
    pub const VOTE_ACCOUNT_SIZE: usize = 3762;

    pub fn new() -> Self {
        Self
    }

    pub fn process_instruction(
        &self,
        accounts: &mut [(Pubkey, Account)],
        instruction_data: &[u8],
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        let instruction: VoteInstruction = borsh::from_slice(instruction_data)
            .map_err(|e| RuntimeError::Program(format!("Failed to deserialize: {}", e)))?;

        match instruction {
            VoteInstruction::Initialize(init) => self.initialize(accounts, init),
            VoteInstruction::Authorize(pubkey, auth_type) => {
                self.authorize(accounts, pubkey, auth_type)
            }
            VoteInstruction::Vote(vote) => self.vote(accounts, vote, current_epoch),
            VoteInstruction::Withdraw(lamports) => self.withdraw(accounts, lamports),
            VoteInstruction::UpdateValidatorIdentity => self.update_validator_identity(accounts),
            VoteInstruction::UpdateCommission(commission) => {
                self.update_commission(accounts, commission)
            }
            VoteInstruction::CompactVoteStateUpdate(update) => {
                self.compact_vote(accounts, update, current_epoch)
            }
        }
    }

    fn initialize(
        &self,
        accounts: &mut [(Pubkey, Account)],
        init: VoteInit,
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let vote_account = &mut accounts[0].1;

        // Check owner
        if vote_account.owner != Pubkey::vote_program() {
            return Err(RuntimeError::Program("Invalid vote account owner".into()));
        }

        // Validate commission
        if init.commission > 100 {
            return Err(RuntimeError::Program("Commission must be <= 100".into()));
        }

        // Initialize state
        let state = VoteState::new(&init);
        vote_account.data = borsh::to_vec(&state).unwrap();

        Ok(())
    }

    fn authorize(
        &self,
        accounts: &mut [(Pubkey, Account)],
        new_authority: Pubkey,
        auth_type: VoteAuthorize,
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let vote_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[2].0;

        let mut state: VoteState = borsh::from_slice(&vote_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        match auth_type {
            VoteAuthorize::Voter => {
                if *signer_pubkey != state.authorized_voter {
                    return Err(RuntimeError::Program("Invalid voter authority".into()));
                }
                state.authorized_voter = new_authority;
            }
            VoteAuthorize::Withdrawer => {
                if *signer_pubkey != state.authorized_withdrawer {
                    return Err(RuntimeError::Program("Invalid withdraw authority".into()));
                }
                state.authorized_withdrawer = new_authority;
            }
        }

        vote_account.data = borsh::to_vec(&state).unwrap();
        Ok(())
    }

    fn vote(
        &self,
        accounts: &mut [(Pubkey, Account)],
        vote: Vote,
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 4 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let vote_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[3].0;

        let mut state: VoteState = borsh::from_slice(&vote_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        // Verify vote authority
        if *signer_pubkey != state.authorized_voter {
            return Err(RuntimeError::Program("Invalid vote authority".into()));
        }

        // Validate vote slots
        if vote.slots.is_empty() {
            return Err(RuntimeError::Program("Empty vote".into()));
        }

        // Process the vote
        state.process_vote(vote, current_epoch);
        vote_account.data = borsh::to_vec(&state).unwrap();

        Ok(())
    }

    fn withdraw(
        &self,
        accounts: &mut [(Pubkey, Account)],
        lamports: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Extract data to avoid borrow conflicts
        let vote_data = accounts[0].1.data.clone();
        let vote_balance = accounts[0].1.lamports;
        let signer_pubkey = accounts[2].0;

        let state: VoteState = borsh::from_slice(&vote_data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        // Verify withdraw authority
        if signer_pubkey != state.authorized_withdrawer {
            return Err(RuntimeError::Program("Invalid withdraw authority".into()));
        }

        // Calculate rent-exempt reserve
        let rent_reserve = Self::calculate_rent_exempt_reserve();
        let withdrawable = vote_balance.saturating_sub(rent_reserve);

        if lamports > withdrawable {
            return Err(RuntimeError::Program(format!(
                "Withdrawal {} exceeds withdrawable {}",
                lamports, withdrawable
            )));
        }

        // Perform withdrawal using indexing
        accounts[0].1.lamports -= lamports;
        accounts[1].1.lamports += lamports;

        Ok(())
    }

    fn update_validator_identity(
        &self,
        accounts: &mut [(Pubkey, Account)],
    ) -> RuntimeResult<()> {
        if accounts.len() < 3 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let vote_account = &mut accounts[0].1;
        let new_identity = &accounts[1].0;
        let signer_pubkey = &accounts[2].0;

        let mut state: VoteState = borsh::from_slice(&vote_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        // Verify withdraw authority
        if *signer_pubkey != state.authorized_withdrawer {
            return Err(RuntimeError::Program("Invalid withdraw authority".into()));
        }

        state.node_pubkey = *new_identity;
        vote_account.data = borsh::to_vec(&state).unwrap();

        Ok(())
    }

    fn update_commission(
        &self,
        accounts: &mut [(Pubkey, Account)],
        commission: u8,
    ) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        if commission > 100 {
            return Err(RuntimeError::Program("Commission must be <= 100".into()));
        }

        let vote_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[1].0;

        let mut state: VoteState = borsh::from_slice(&vote_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        // Verify withdraw authority
        if *signer_pubkey != state.authorized_withdrawer {
            return Err(RuntimeError::Program("Invalid withdraw authority".into()));
        }

        state.commission = commission;
        vote_account.data = borsh::to_vec(&state).unwrap();

        Ok(())
    }

    fn compact_vote(
        &self,
        accounts: &mut [(Pubkey, Account)],
        update: CompactVoteStateUpdate,
        current_epoch: u64,
    ) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let vote_account = &mut accounts[0].1;
        let signer_pubkey = &accounts[1].0;

        let mut state: VoteState = borsh::from_slice(&vote_account.data)
            .map_err(|e| RuntimeError::Program(format!("Invalid state: {}", e)))?;

        // Verify vote authority
        if *signer_pubkey != state.authorized_voter {
            return Err(RuntimeError::Program("Invalid vote authority".into()));
        }

        // Apply compact update
        state.root_slot = Some(update.root);
        state.votes = VecDeque::from(update.lockouts);

        if let Some(timestamp) = update.timestamp {
            state.last_timestamp = BlockTimestamp {
                slot: state.votes.back().map(|l| l.slot).unwrap_or(update.root),
                timestamp,
            };
        }

        // Increment credits
        state.increment_credits(current_epoch, 1);

        vote_account.data = borsh::to_vec(&state).unwrap();
        Ok(())
    }

    fn calculate_rent_exempt_reserve() -> u64 {
        Self::VOTE_ACCOUNT_SIZE as u64 * 3480 * 2
    }
}

impl Default for VoteProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create vote instructions
pub mod instruction {
    use super::*;

    pub fn initialize(
        vote_pubkey: &Pubkey,
        init: &VoteInit,
    ) -> Instruction {
        let data = borsh::to_vec(&VoteInstruction::Initialize(init.clone())).unwrap();

        Instruction {
            program_id: Pubkey::vote_program(),
            accounts: vec![
                AccountMeta::new(*vote_pubkey, false),
                AccountMeta::new_readonly(Pubkey::sysvar_rent(), false),
                AccountMeta::new_readonly(Pubkey::sysvar_clock(), false),
                AccountMeta::new_readonly(init.node_pubkey, true),
            ],
            data,
        }
    }

    pub fn vote(
        vote_pubkey: &Pubkey,
        authorized_voter: &Pubkey,
        vote: Vote,
    ) -> Instruction {
        let data = borsh::to_vec(&VoteInstruction::Vote(vote)).unwrap();

        Instruction {
            program_id: Pubkey::vote_program(),
            accounts: vec![
                AccountMeta::new(*vote_pubkey, false),
                AccountMeta::new_readonly(Pubkey::sysvar_slot_hashes(), false),
                AccountMeta::new_readonly(Pubkey::sysvar_clock(), false),
                AccountMeta::new_readonly(*authorized_voter, true),
            ],
            data,
        }
    }

    pub fn withdraw(
        vote_pubkey: &Pubkey,
        recipient: &Pubkey,
        authorized_withdrawer: &Pubkey,
        lamports: u64,
    ) -> Instruction {
        let data = borsh::to_vec(&VoteInstruction::Withdraw(lamports)).unwrap();

        Instruction {
            program_id: Pubkey::vote_program(),
            accounts: vec![
                AccountMeta::new(*vote_pubkey, false),
                AccountMeta::new(*recipient, false),
                AccountMeta::new_readonly(*authorized_withdrawer, true),
            ],
            data,
        }
    }

    pub fn update_commission(
        vote_pubkey: &Pubkey,
        authorized_withdrawer: &Pubkey,
        commission: u8,
    ) -> Instruction {
        let data = borsh::to_vec(&VoteInstruction::UpdateCommission(commission)).unwrap();

        Instruction {
            program_id: Pubkey::vote_program(),
            accounts: vec![
                AccountMeta::new(*vote_pubkey, false),
                AccountMeta::new_readonly(*authorized_withdrawer, true),
            ],
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vote_state_init() {
        let init = VoteInit {
            node_pubkey: Pubkey::new([1u8; 32]),
            authorized_voter: Pubkey::new([2u8; 32]),
            authorized_withdrawer: Pubkey::new([3u8; 32]),
            commission: 10,
        };

        let state = VoteState::new(&init);
        assert_eq!(state.commission, 10);
        assert_eq!(state.node_pubkey, Pubkey::new([1u8; 32]));
    }

    #[test]
    fn test_lockout() {
        let lockout = Lockout {
            slot: 100,
            confirmation_count: 3,
        };

        // 2^3 = 8 slot lockout
        assert_eq!(lockout.lockout(), 8);
        assert!(lockout.is_locked_out_at_slot(107));
        assert!(!lockout.is_locked_out_at_slot(109));
    }

    #[test]
    fn test_vote_processing() {
        let init = VoteInit {
            node_pubkey: Pubkey::default(),
            authorized_voter: Pubkey::default(),
            authorized_withdrawer: Pubkey::default(),
            commission: 0,
        };

        let mut state = VoteState::new(&init);

        // Process some votes
        let vote = Vote {
            slots: vec![100, 101, 102],
            hash: [0u8; 32],
            timestamp: Some(1234567890),
        };

        state.process_vote(vote, 0);

        assert_eq!(state.votes.len(), 3);
        assert_eq!(state.credits(), 3);
    }
}
