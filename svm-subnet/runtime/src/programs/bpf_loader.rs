use crate::error::{RuntimeError, RuntimeResult};
use crate::types::{Account, AccountMeta, Instruction, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};

/// BPF Loader instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub enum BpfLoaderInstruction {
    /// Write to program account data
    Write { offset: u32, bytes: Vec<u8> },
    /// Finalize the account as executable
    Finalize,
}

/// BPF Loader program processor
pub struct BpfLoaderProgram;

impl BpfLoaderProgram {
    pub fn new() -> Self {
        Self
    }

    pub fn process_instruction(
        &self,
        accounts: &mut [(Pubkey, Account)],
        instruction_data: &[u8],
    ) -> RuntimeResult<()> {
        let instruction: BpfLoaderInstruction = borsh::from_slice(instruction_data)
            .map_err(|e| RuntimeError::Program(format!("Failed to deserialize: {}", e)))?;

        match instruction {
            BpfLoaderInstruction::Write { offset, bytes } => self.write(accounts, offset, bytes),
            BpfLoaderInstruction::Finalize => self.finalize(accounts),
        }
    }

    fn write(
        &self,
        accounts: &mut [(Pubkey, Account)],
        offset: u32,
        bytes: Vec<u8>,
    ) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let program_account = &mut accounts[0].1;

        if program_account.owner != Pubkey::bpf_loader() {
            return Err(RuntimeError::Program(
                "Account not owned by BPF loader".into(),
            ));
        }

        let end = offset as usize + bytes.len();
        if end > program_account.data.len() {
            return Err(RuntimeError::Program("Write exceeds account data".into()));
        }

        program_account.data[offset as usize..end].copy_from_slice(&bytes);
        Ok(())
    }

    fn finalize(&self, accounts: &mut [(Pubkey, Account)]) -> RuntimeResult<()> {
        if accounts.is_empty() {
            return Err(RuntimeError::Program("Not enough accounts".into()));
        }

        let program_account = &mut accounts[0].1;

        if program_account.owner != Pubkey::bpf_loader() {
            return Err(RuntimeError::Program(
                "Account not owned by BPF loader".into(),
            ));
        }

        // TODO: Verify ELF bytecode is valid
        program_account.executable = true;
        Ok(())
    }
}

impl Default for BpfLoaderProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create BPF loader instructions
pub mod instruction {
    use super::*;

    pub fn write(program_id: &Pubkey, offset: u32, bytes: Vec<u8>) -> Instruction {
        let data = borsh::to_vec(&BpfLoaderInstruction::Write { offset, bytes }).unwrap();

        Instruction {
            program_id: Pubkey::bpf_loader(),
            accounts: vec![AccountMeta::new(*program_id, true)],
            data,
        }
    }

    pub fn finalize(program_id: &Pubkey) -> Instruction {
        let data = borsh::to_vec(&BpfLoaderInstruction::Finalize).unwrap();

        Instruction {
            program_id: Pubkey::bpf_loader(),
            accounts: vec![AccountMeta::new(*program_id, true)],
            data,
        }
    }
}
