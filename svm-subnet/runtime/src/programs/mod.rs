//! Native Programs
//!
//! Built-in programs that run at native speed:
//! - System Program: Account creation, transfers, allocation
//! - BPF Loader: Program deployment and execution
//! - Stake Program: Validator staking and delegation
//! - Vote Program: Validator voting for consensus

pub mod bpf_loader;
pub mod stake;
pub mod system;
pub mod vote;

pub use bpf_loader::*;
pub use stake::*;
pub use system::*;
pub use vote::*;

use crate::types::Pubkey;
use crate::sysvars::{Rent, is_sysvar as sysvar_is_sysvar};

/// Check if a pubkey is a native program
pub fn is_native_program(pubkey: &Pubkey) -> bool {
    *pubkey == Pubkey::system_program()
        || *pubkey == Pubkey::bpf_loader()
        || *pubkey == Pubkey::bpf_loader_upgradeable()
        || *pubkey == Pubkey::stake_program()
        || *pubkey == Pubkey::vote_program()
}

/// Native program IDs with their processors
pub struct NativePrograms {
    pub system: SystemProgram,
    pub bpf_loader: BpfLoaderProgram,
    pub stake: StakeProgram,
    pub vote: VoteProgram,
}

impl Default for NativePrograms {
    fn default() -> Self {
        Self::new()
    }
}

impl NativePrograms {
    pub fn new() -> Self {
        Self {
            system: SystemProgram::new(Rent::default()),
            bpf_loader: BpfLoaderProgram::new(),
            stake: StakeProgram::new(),
            vote: VoteProgram::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_native_program() {
        assert!(is_native_program(&Pubkey::system_program()));
        assert!(is_native_program(&Pubkey::bpf_loader()));
        assert!(is_native_program(&Pubkey::stake_program()));
        assert!(is_native_program(&Pubkey::vote_program()));
        assert!(!is_native_program(&Pubkey::new([42u8; 32])));
    }

    #[test]
    fn test_is_sysvar() {
        use crate::sysvars::ids;
        assert!(sysvar_is_sysvar(&*ids::CLOCK));
        assert!(sysvar_is_sysvar(&*ids::RENT));
        assert!(!sysvar_is_sysvar(&Pubkey::system_program()));
    }
}
