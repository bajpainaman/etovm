use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// 32-byte public key, compatible with Solana's Pubkey
#[derive(
    Clone,
    Copy,
    Default,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    BorshSerialize,
    BorshDeserialize,
    Serialize,
    Deserialize,
)]
#[repr(transparent)]
pub struct Pubkey(pub [u8; 32]);

impl Pubkey {
    pub const LEN: usize = 32;

    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn new_from_slice(slice: &[u8]) -> Result<Self, PubkeyError> {
        if slice.len() != 32 {
            return Err(PubkeyError::InvalidLength);
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(slice);
        Ok(Self(bytes))
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    pub fn as_ref(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create a program-derived address (PDA)
    pub fn find_program_address(seeds: &[&[u8]], program_id: &Pubkey) -> (Pubkey, u8) {
        for bump in (0..=255).rev() {
            let mut seeds_with_bump = seeds.to_vec();
            let bump_slice = &[bump];
            seeds_with_bump.push(bump_slice);

            if let Ok(address) = Self::create_program_address(&seeds_with_bump, program_id) {
                return (address, bump);
            }
        }
        panic!("Unable to find a viable program address bump seed");
    }

    pub fn create_program_address(
        seeds: &[&[u8]],
        program_id: &Pubkey,
    ) -> Result<Pubkey, PubkeyError> {
        let mut hasher = Sha256::new();
        for seed in seeds {
            hasher.update(seed);
        }
        hasher.update(program_id.as_ref());
        hasher.update(b"ProgramDerivedAddress");

        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);

        Ok(Pubkey(bytes))
    }

    // Well-known program IDs
    pub const fn system_program() -> Self {
        Self([0u8; 32])
    }

    pub fn bpf_loader() -> Self {
        // BPFLoader2111111111111111111111111111111111
        Self::from_base58("BPFLoader2111111111111111111111111111111111")
    }

    pub fn bpf_loader_upgradeable() -> Self {
        // BPFLoaderUpgradeab1e11111111111111111111111
        Self::from_base58("BPFLoaderUpgradeab1e11111111111111111111111")
    }

    pub fn stake_program() -> Self {
        // Stake11111111111111111111111111111111111111
        Self::from_base58("Stake11111111111111111111111111111111111111")
    }

    pub fn vote_program() -> Self {
        // Vote111111111111111111111111111111111111111
        Self::from_base58("Vote111111111111111111111111111111111111111")
    }

    pub fn stake_config() -> Self {
        // StakeConfig11111111111111111111111111111111
        Self::from_base58("StakeConfig11111111111111111111111111111111")
    }

    // Sysvar addresses - deterministic pubkeys from seed hashes
    pub fn sysvar_clock() -> Self {
        Self::from_seed(b"sysvar:clock")
    }

    pub fn sysvar_rent() -> Self {
        Self::from_seed(b"sysvar:rent")
    }

    pub fn sysvar_epoch_schedule() -> Self {
        Self::from_seed(b"sysvar:epoch_schedule")
    }

    pub fn sysvar_slot_hashes() -> Self {
        Self::from_seed(b"sysvar:slot_hashes")
    }

    pub fn sysvar_recent_blockhashes() -> Self {
        Self::from_seed(b"sysvar:recent_blockhashes")
    }

    pub fn sysvar_stake_history() -> Self {
        Self::from_seed(b"sysvar:stake_history")
    }

    fn from_seed(seed: &[u8]) -> Self {
        let hash = Sha256::digest(seed);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&hash);
        Self(arr)
    }

    fn from_base58(s: &str) -> Self {
        let bytes = bs58::decode(s).into_vec().expect("Invalid base58");
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Self(arr)
    }

    /// Create a unique pubkey for testing (uses incrementing counter)
    #[cfg(test)]
    pub fn new_unique() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&count.to_le_bytes());
        Self(bytes)
    }
}

impl fmt::Debug for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", bs58::encode(&self.0).into_string())
    }
}

impl fmt::Display for Pubkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", bs58::encode(&self.0).into_string())
    }
}

impl From<[u8; 32]> for Pubkey {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for Pubkey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PubkeyError {
    #[error("Invalid pubkey length")]
    InvalidLength,
    #[error("Invalid base58 string")]
    InvalidBase58,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pubkey_display() {
        let pk = Pubkey::system_program();
        assert_eq!(
            pk.to_string(),
            "11111111111111111111111111111111"
        );
    }

    #[test]
    fn test_pubkey_from_bytes() {
        let bytes = [1u8; 32];
        let pk = Pubkey::new(bytes);
        assert_eq!(pk.to_bytes(), bytes);
    }
}
