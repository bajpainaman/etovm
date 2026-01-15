pub mod bpf_loader;
pub mod system;

pub use bpf_loader::*;
pub use system::*;

use crate::types::Pubkey;

/// Check if a pubkey is a native program
pub fn is_native_program(pubkey: &Pubkey) -> bool {
    *pubkey == Pubkey::system_program()
        || *pubkey == Pubkey::bpf_loader()
        || *pubkey == Pubkey::bpf_loader_upgradeable()
}
