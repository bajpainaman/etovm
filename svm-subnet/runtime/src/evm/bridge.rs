use crate::{Account, Pubkey, RuntimeError, RuntimeResult};
use super::{EvmAddress, EvmExecutor, EvmStateAdapter, evm_address_to_pubkey, pubkey_to_evm_address};
use std::collections::HashMap;

/// Address mapper for bidirectional SVM ↔ EVM address translation
pub struct AddressMapper {
    /// EVM address -> Solana pubkey
    evm_to_svm: HashMap<EvmAddress, Pubkey>,
    /// Solana pubkey -> EVM address
    svm_to_evm: HashMap<Pubkey, EvmAddress>,
}

impl AddressMapper {
    pub fn new() -> Self {
        Self {
            evm_to_svm: HashMap::new(),
            svm_to_evm: HashMap::new(),
        }
    }

    /// Register a bidirectional mapping
    pub fn register(&mut self, evm_addr: EvmAddress, svm_pubkey: Pubkey) {
        self.evm_to_svm.insert(evm_addr, svm_pubkey);
        self.svm_to_evm.insert(svm_pubkey, evm_addr);
    }

    /// Get SVM pubkey for EVM address
    pub fn evm_to_svm(&self, evm_addr: &EvmAddress) -> Option<Pubkey> {
        self.evm_to_svm.get(evm_addr).copied()
    }

    /// Get EVM address for SVM pubkey
    pub fn svm_to_evm(&self, svm_pubkey: &Pubkey) -> Option<EvmAddress> {
        self.svm_to_evm.get(svm_pubkey).copied()
    }

    /// Derive EVM address from SVM pubkey (deterministic)
    pub fn derive_evm_address(pubkey: &Pubkey) -> EvmAddress {
        pubkey_to_evm_address(&pubkey.0)
    }

    /// Derive SVM pubkey from EVM address (deterministic)
    pub fn derive_svm_pubkey(evm_addr: &EvmAddress) -> Pubkey {
        Pubkey(evm_address_to_pubkey(evm_addr))
    }
}

impl Default for AddressMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-VM bridge for SVM ↔ EVM interoperability
pub struct EvmBridge {
    /// EVM executor
    executor: EvmExecutor,
    /// Address mapper
    mapper: AddressMapper,
    /// EVM state
    state: EvmStateAdapter,
}

impl EvmBridge {
    pub fn new(chain_id: u64) -> Self {
        Self {
            executor: EvmExecutor::new(chain_id),
            mapper: AddressMapper::new(),
            state: EvmStateAdapter::new(),
        }
    }

    /// Set block context for EVM execution
    pub fn set_block(&mut self, number: u64, timestamp: u64) {
        self.executor.set_block(number, timestamp);
    }

    /// Load SVM accounts into EVM state
    pub fn load_accounts(&mut self, accounts: Vec<(Pubkey, Account)>) {
        let converted: Vec<([u8; 32], Account)> = accounts
            .into_iter()
            .map(|(pk, acc)| (pk.0, acc))
            .collect();
        self.state.load_accounts(converted);
    }

    /// Call EVM contract from SVM
    ///
    /// This is the main entry point for SVM → EVM calls.
    /// Called via the sol_call_evm syscall.
    pub fn call_evm(
        &mut self,
        caller_pubkey: &Pubkey,
        evm_contract: &EvmAddress,
        calldata: &[u8],
        value: u128,
    ) -> RuntimeResult<Vec<u8>> {
        // Convert SVM caller to EVM address
        let caller_evm = self.mapper.svm_to_evm(caller_pubkey)
            .unwrap_or_else(|| AddressMapper::derive_evm_address(caller_pubkey));

        // Execute EVM call
        let result = self.executor.execute(
            &mut self.state,
            caller_evm,
            Some(*evm_contract),
            value,
            calldata.to_vec(),
            1_000_000, // 1M gas for cross-VM calls
        )?;

        if result.success {
            Ok(result.output)
        } else {
            Err(RuntimeError::CrossVmCallFailed(
                format!("EVM call reverted: {}", hex::encode(&result.output))
            ))
        }
    }

    /// Static call to EVM contract (read-only)
    pub fn static_call_evm(
        &mut self,
        evm_contract: &EvmAddress,
        calldata: &[u8],
    ) -> RuntimeResult<Vec<u8>> {
        let zero_caller = [0u8; 20];
        self.executor.static_call(&mut self.state, zero_caller, *evm_contract, calldata.to_vec())
    }

    /// Get balance of an EVM address (returns wei)
    pub fn get_evm_balance(&self, evm_addr: &EvmAddress) -> RuntimeResult<u128> {
        let pubkey = AddressMapper::derive_svm_pubkey(evm_addr);
        // Look up in SVM accounts
        // For now return 0 - would need state access
        Ok(0)
    }

    /// Deploy EVM contract from SVM
    pub fn deploy_evm_contract(
        &mut self,
        deployer_pubkey: &Pubkey,
        bytecode: &[u8],
        value: u128,
    ) -> RuntimeResult<EvmAddress> {
        let deployer_evm = self.mapper.svm_to_evm(deployer_pubkey)
            .unwrap_or_else(|| AddressMapper::derive_evm_address(deployer_pubkey));

        self.executor.deploy(&mut self.state, deployer_evm, bytecode.to_vec(), value)
    }

    /// Get modified accounts after EVM execution (for state sync)
    pub fn get_modified_accounts(&self) -> Vec<(Pubkey, Account)> {
        self.state.get_modified_accounts()
            .into_iter()
            .map(|(pk, acc)| (Pubkey(pk), acc))
            .collect()
    }

    /// Encode a Solidity function call
    ///
    /// Helper for building calldata for EVM calls.
    pub fn encode_call(selector: [u8; 4], args: &[u8]) -> Vec<u8> {
        let mut calldata = Vec::with_capacity(4 + args.len());
        calldata.extend_from_slice(&selector);
        calldata.extend_from_slice(args);
        calldata
    }

    /// Compute function selector (keccak256(signature)[0:4])
    pub fn function_selector(signature: &str) -> [u8; 4] {
        use sha3::{Keccak256, Digest};
        let hash = Keccak256::digest(signature.as_bytes());
        let mut selector = [0u8; 4];
        selector.copy_from_slice(&hash[..4]);
        selector
    }
}

/// Precompile addresses for cross-VM operations
pub mod precompiles {
    use super::EvmAddress;

    /// SVM Bridge precompile - allows EVM contracts to call SVM programs
    /// Address: 0x0000000000000000000000000000000000000100
    pub const SVM_BRIDGE: EvmAddress = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    ];

    /// System Program precompile
    /// Address: 0x0000000000000000000000000000000000000101
    pub const SYSTEM_PROGRAM: EvmAddress = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    ];

    /// SPL Token precompile
    /// Address: 0x0000000000000000000000000000000000000102
    pub const SPL_TOKEN: EvmAddress = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_mapper() {
        let mut mapper = AddressMapper::new();

        let evm_addr = [1u8; 20];
        let svm_pubkey = Pubkey([2u8; 32]);

        mapper.register(evm_addr, svm_pubkey);

        assert_eq!(mapper.evm_to_svm(&evm_addr), Some(svm_pubkey));
        assert_eq!(mapper.svm_to_evm(&svm_pubkey), Some(evm_addr));
    }

    #[test]
    fn test_function_selector() {
        // transfer(address,uint256) -> 0xa9059cbb
        let selector = EvmBridge::function_selector("transfer(address,uint256)");
        assert_eq!(selector, [0xa9, 0x05, 0x9c, 0xbb]);
    }

    #[test]
    fn test_derive_addresses() {
        let pubkey = Pubkey([42u8; 32]);
        let evm_addr = AddressMapper::derive_evm_address(&pubkey);

        // Should be last 20 bytes
        assert_eq!(evm_addr, [42u8; 20]);

        // Deriving back should give a different pubkey (hashed)
        let derived_pubkey = AddressMapper::derive_svm_pubkey(&evm_addr);
        assert_ne!(derived_pubkey, pubkey);
    }
}
