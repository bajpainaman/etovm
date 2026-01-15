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

    fn from_base58(s: &str) -> Self {
        let bytes = bs58::decode(s).into_vec().expect("Invalid base58");
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Self(arr)
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
