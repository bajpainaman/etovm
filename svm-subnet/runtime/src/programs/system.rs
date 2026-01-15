use crate::error::{RuntimeError, RuntimeResult};
use crate::sysvars::Rent;
use crate::types::{Account, AccountMeta, Instruction, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};

/// System program instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum SystemInstruction {
    /// Create a new account
    CreateAccount {
        lamports: u64,
        space: u64,
        owner: Pubkey,
    },

    /// Assign account to a program
    Assign { owner: Pubkey },

    /// Transfer lamports
    Transfer { lamports: u64 },

    /// Create account with seed
    CreateAccountWithSeed {
        base: Pubkey,
        seed: String,
        lamports: u64,
        space: u64,
        owner: Pubkey,
    },

    /// Allocate space for account data
    Allocate { space: u64 },

    /// Allocate space with seed
    AllocateWithSeed {
        base: Pubkey,
        seed: String,
        space: u64,
        owner: Pubkey,
    },

    /// Assign with seed
    AssignWithSeed {
        base: Pubkey,
        seed: String,
        owner: Pubkey,
    },

    /// Transfer with seed
    TransferWithSeed {
        lamports: u64,
        from_seed: String,
        from_owner: Pubkey,
    },
}

/// System program processor
pub struct SystemProgram {
    rent: Rent,
}

impl SystemProgram {
    pub fn new(rent: Rent) -> Self {
        Self { rent }
    }

    pub fn process_instruction(
        &self,
        accounts: &mut [(Pubkey, Account)],
        instruction_data: &[u8],
    ) -> RuntimeResult<()> {
        let instruction: SystemInstruction = borsh::from_slice(instruction_data)
            .map_err(|e| RuntimeError::Program(format!("Failed to deserialize: {}", e)))?;

        match instruction {
            SystemInstruction::CreateAccount {
                lamports,
                space,
                owner,
            } => self.create_account(accounts, lamports, space, owner),
            SystemInstruction::Assign { owner } => self.assign(accounts, owner),
            SystemInstruction::Transfer { lamports } => self.transfer(accounts, lamports),
            SystemInstruction::CreateAccountWithSeed {
                base,
                seed,
                lamports,
                space,
                owner,
            } => self.create_account_with_seed(accounts, base, seed, lamports, space, owner),
            SystemInstruction::Allocate { space } => self.allocate(accounts, space),
            SystemInstruction::AllocateWithSeed {
                base,
                seed,
                space,
                owner,
            } => self.allocate_with_seed(accounts, base, seed, space, owner),
            SystemInstruction::AssignWithSeed { base, seed, owner } => {
                self.assign_with_seed(accounts, base, seed, owner)
            }
            SystemInstruction::TransferWithSeed {
                lamports,
                from_seed,
                from_owner,
            } => self.transfer_with_seed(accounts, lamports, from_seed, from_owner),
        }
    }

    fn create_account(
        &self,
        accounts: &mut [(Pubkey, Account)],
        lamports: u64,
        space: u64,
        owner: Pubkey,
    ) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        // Check new account state first
        if accounts[1].1.lamports != 0 || !accounts[1].1.data.is_empty() {
            return Err(RuntimeError::Program("Account already initialized".into()));
        }

        let min_balance = self.rent.minimum_balance(space as usize);
        if lamports < min_balance {
            return Err(RuntimeError::Program(format!(
                "Insufficient lamports for rent exemption: {} < {}",
                lamports, min_balance
            )));
        }

        if accounts[0].1.lamports < lamports {
            return Err(RuntimeError::Program("Insufficient funds".into()));
        }

        // Modify accounts using split_at_mut to satisfy borrow checker
        let (first, rest) = accounts.split_at_mut(1);
        let funding_account = &mut first[0].1;
        let new_account = &mut rest[0].1;

        funding_account.lamports -= lamports;
        new_account.lamports = lamports;
        new_account.data = vec![0; space as usize];
        new_account.owner = owner;

        Ok(())
    }

    fn assign(&self, accounts: &mut [(Pubkey, Account)], owner: Pubkey) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let account = &mut accounts[0].1;

        if account.owner != Pubkey::system_program() {
            return Err(RuntimeError::Program(
                "Account not owned by system program".into(),
            ));
        }

        account.owner = owner;
        Ok(())
    }

    fn transfer(&self, accounts: &mut [(Pubkey, Account)], lamports: u64) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        if accounts[0].1.lamports < lamports {
            return Err(RuntimeError::Program("Insufficient funds".into()));
        }

        // Use split_at_mut to satisfy borrow checker
        let (first, rest) = accounts.split_at_mut(1);
        first[0].1.lamports -= lamports;
        rest[0].1.lamports += lamports;

        Ok(())
    }

    fn create_account_with_seed(
        &self,
        accounts: &mut [(Pubkey, Account)],
        base: Pubkey,
        seed: String,
        lamports: u64,
        space: u64,
        owner: Pubkey,
    ) -> RuntimeResult<()> {
        if accounts.len() < 2 {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let expected_address = Pubkey::create_program_address(&[base.as_ref(), seed.as_bytes()], &owner)
            .map_err(|e| RuntimeError::Program(format!("Invalid seed: {:?}", e)))?;

        if accounts[1].0 != expected_address {
            return Err(RuntimeError::Program("Address mismatch".into()));
        }

        self.create_account(accounts, lamports, space, owner)
    }

    fn allocate(&self, accounts: &mut [(Pubkey, Account)], space: u64) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let account = &mut accounts[0].1;

        if account.owner != Pubkey::system_program() {
            return Err(RuntimeError::Program(
                "Account not owned by system program".into(),
            ));
        }

        if !account.data.is_empty() {
            return Err(RuntimeError::Program("Account already has data".into()));
        }

        account.data = vec![0; space as usize];
        Ok(())
    }

    fn allocate_with_seed(
        &self,
        accounts: &mut [(Pubkey, Account)],
        _base: Pubkey,
        _seed: String,
        space: u64,
        _owner: Pubkey,
    ) -> RuntimeResult<()> {
        self.allocate(accounts, space)
    }

    fn assign_with_seed(
        &self,
        accounts: &mut [(Pubkey, Account)],
        _base: Pubkey,
        _seed: String,
        owner: Pubkey,
    ) -> RuntimeResult<()> {
        self.assign(accounts, owner)
    }

    fn transfer_with_seed(
        &self,
        accounts: &mut [(Pubkey, Account)],
        lamports: u64,
        _from_seed: String,
        _from_owner: Pubkey,
    ) -> RuntimeResult<()> {
        self.transfer(accounts, lamports)
    }
}

/// Helper to create system program instructions
pub mod instruction {
    use super::*;

    pub fn create_account(
        from_pubkey: &Pubkey,
        to_pubkey: &Pubkey,
        lamports: u64,
        space: u64,
        owner: &Pubkey,
    ) -> Instruction {
        let data = borsh::to_vec(&SystemInstruction::CreateAccount {
            lamports,
            space,
            owner: *owner,
        })
        .unwrap();

        Instruction {
            program_id: Pubkey::system_program(),
            accounts: vec![
                AccountMeta::new(*from_pubkey, true),
                AccountMeta::new(*to_pubkey, true),
            ],
            data,
        }
    }

    pub fn transfer(from_pubkey: &Pubkey, to_pubkey: &Pubkey, lamports: u64) -> Instruction {
        let data = borsh::to_vec(&SystemInstruction::Transfer { lamports }).unwrap();

        Instruction {
            program_id: Pubkey::system_program(),
            accounts: vec![
                AccountMeta::new(*from_pubkey, true),
                AccountMeta::new(*to_pubkey, false),
            ],
            data,
        }
    }

    pub fn assign(pubkey: &Pubkey, owner: &Pubkey) -> Instruction {
        let data = borsh::to_vec(&SystemInstruction::Assign { owner: *owner }).unwrap();

        Instruction {
            program_id: Pubkey::system_program(),
            accounts: vec![AccountMeta::new(*pubkey, true)],
            data,
        }
    }

    pub fn allocate(pubkey: &Pubkey, space: u64) -> Instruction {
        let data = borsh::to_vec(&SystemInstruction::Allocate { space }).unwrap();

        Instruction {
            program_id: Pubkey::system_program(),
            accounts: vec![AccountMeta::new(*pubkey, true)],
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_accounts() -> Vec<(Pubkey, Account)> {
        let from_pubkey = Pubkey::new([1u8; 32]);
        let to_pubkey = Pubkey::new([2u8; 32]);

        let from_account = Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        };

        let to_account = Account::default();

        vec![(from_pubkey, from_account), (to_pubkey, to_account)]
    }

    #[test]
    fn test_transfer() {
        let mut accounts = setup_accounts();
        let system = SystemProgram::new(Rent::default());

        let ix_data = borsh::to_vec(&SystemInstruction::Transfer {
            lamports: 100_000_000,
        })
        .unwrap();

        system.process_instruction(&mut accounts, &ix_data).unwrap();

        assert_eq!(accounts[0].1.lamports, 900_000_000);
        assert_eq!(accounts[1].1.lamports, 100_000_000);
    }

    #[test]
    fn test_create_account() {
        let mut accounts = setup_accounts();
        let system = SystemProgram::new(Rent::default());

        let owner = Pubkey::new([3u8; 32]);
        let space = 100u64;
        let lamports = Rent::default().minimum_balance(space as usize);

        let ix_data = borsh::to_vec(&SystemInstruction::CreateAccount {
            lamports,
            space,
            owner,
        })
        .unwrap();

        system.process_instruction(&mut accounts, &ix_data).unwrap();

        assert_eq!(accounts[1].1.lamports, lamports);
        assert_eq!(accounts[1].1.data.len(), space as usize);
        assert_eq!(accounts[1].1.owner, owner);
    }
}
