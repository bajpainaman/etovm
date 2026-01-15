use super::{AccountMeta, Pubkey};
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

/// A single instruction to be executed by a program
#[derive(
    Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize,
)]
pub struct Instruction {
    /// Program ID that will process this instruction
    pub program_id: Pubkey,
    /// Accounts required by the instruction
    pub accounts: Vec<AccountMeta>,
    /// Instruction data (program-specific)
    pub data: Vec<u8>,
}

impl Instruction {
    pub fn new(program_id: Pubkey, data: Vec<u8>, accounts: Vec<AccountMeta>) -> Self {
        Self {
            program_id,
            accounts,
            data,
        }
    }

    pub fn new_with_borsh<T: BorshSerialize>(
        program_id: Pubkey,
        data: &T,
        accounts: Vec<AccountMeta>,
    ) -> Result<Self, std::io::Error> {
        Ok(Self {
            program_id,
            accounts,
            data: borsh::to_vec(data)?,
        })
    }
}

/// Compiled instruction - indexes into the transaction's account list
#[derive(Clone, Debug, PartialEq, Eq, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct CompiledInstruction {
    /// Index into the transaction's account keys array
    pub program_id_index: u8,
    /// Indices into the transaction's account keys array
    pub accounts: Vec<u8>,
    /// Instruction data
    pub data: Vec<u8>,
}
