pub mod clock;
pub mod epoch_schedule;
pub mod fees;
pub mod rent;

pub use clock::*;
pub use epoch_schedule::*;
pub use fees::*;
pub use rent::*;

use crate::types::Pubkey;

/// Sysvar account addresses
pub mod ids {
    use super::*;
    use lazy_static::lazy_static;

    lazy_static! {
        pub static ref CLOCK: Pubkey = Pubkey::sysvar_clock();
        pub static ref RENT: Pubkey = Pubkey::sysvar_rent();
        pub static ref EPOCH_SCHEDULE: Pubkey = Pubkey::sysvar_epoch_schedule();
        pub static ref FEES: Pubkey = pubkey_from_hash(b"sysvar:fees");
        pub static ref RECENT_BLOCKHASHES: Pubkey = Pubkey::sysvar_recent_blockhashes();
        pub static ref STAKE_HISTORY: Pubkey = Pubkey::sysvar_stake_history();
        pub static ref INSTRUCTIONS: Pubkey = pubkey_from_hash(b"sysvar:instructions");
        pub static ref SLOT_HASHES: Pubkey = Pubkey::sysvar_slot_hashes();
    }

    fn pubkey_from_hash(seed: &[u8]) -> Pubkey {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(seed);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&hash);
        Pubkey::from(arr)
    }
}

pub fn is_sysvar(pubkey: &Pubkey) -> bool {
    *pubkey == *ids::CLOCK
        || *pubkey == *ids::RENT
        || *pubkey == *ids::EPOCH_SCHEDULE
        || *pubkey == *ids::FEES
        || *pubkey == *ids::RECENT_BLOCKHASHES
        || *pubkey == *ids::STAKE_HISTORY
        || *pubkey == *ids::INSTRUCTIONS
        || *pubkey == *ids::SLOT_HASHES
}
