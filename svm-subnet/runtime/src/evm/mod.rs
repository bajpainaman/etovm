mod executor;
mod bridge;
mod state;

pub use executor::EvmExecutor;
pub use bridge::{EvmBridge, AddressMapper};
pub use state::EvmStateAdapter;

use crate::{RuntimeError, RuntimeResult};

/// EVM address (20 bytes)
pub type EvmAddress = [u8; 20];

/// EVM execution result
#[derive(Debug, Clone)]
pub struct EvmExecutionResult {
    pub success: bool,
    pub output: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<EvmLog>,
}

/// EVM log entry
#[derive(Debug, Clone)]
pub struct EvmLog {
    pub address: EvmAddress,
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

/// Convert Solana pubkey to EVM address (truncate to 20 bytes)
pub fn pubkey_to_evm_address(pubkey: &[u8; 32]) -> EvmAddress {
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&pubkey[12..32]);
    addr
}

/// Convert EVM address to Solana pubkey (sha256 hash with prefix)
pub fn evm_address_to_pubkey(addr: &EvmAddress) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(b"evm:");
    hasher.update(addr);
    hasher.finalize().into()
}
