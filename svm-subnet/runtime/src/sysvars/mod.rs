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
        pub static ref CLOCK: Pubkey = pubkey_from_str("SysvarC1ock11111111111111111111111111111111");
        pub static ref RENT: Pubkey = pubkey_from_str("SysvarRent111111111111111111111111111111111");
        pub static ref EPOCH_SCHEDULE: Pubkey =
            pubkey_from_str("SysvarEpochScheworLd1111111111111111111111111");
        pub static ref FEES: Pubkey = pubkey_from_str("SysvarFees111111111111111111111111111111111");
        pub static ref RECENT_BLOCKHASHES: Pubkey =
            pubkey_from_str("SysvarRecentB1ockHashes11111111111111111111");
        pub static ref STAKE_HISTORY: Pubkey =
            pubkey_from_str("SysvarStakeHistory1111111111111111111111111");
        pub static ref INSTRUCTIONS: Pubkey =
            pubkey_from_str("Sysvar1nstructions1111111111111111111111111");
        pub static ref SLOT_HASHES: Pubkey =
            pubkey_from_str("SysvarS1otHashes111111111111111111111111111");
    }

    fn pubkey_from_str(s: &str) -> Pubkey {
        let bytes = bs58::decode(s).into_vec().expect("Invalid base58");
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
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
