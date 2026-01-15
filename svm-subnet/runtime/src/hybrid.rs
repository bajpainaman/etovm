use crate::{
    Account, Pubkey, RuntimeError, RuntimeResult, Transaction,
    bpf::{BpfVm, DEFAULT_COMPUTE_UNITS},
    evm::{EvmBridge, EvmAddress, AddressMapper},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Hybrid executor that runs both SVM (BPF) and EVM code
///
/// This is the "morphed twins" architecture - SVM and EVM coexist
/// on the same chain with bidirectional interoperability.
pub struct HybridExecutor {
    /// Chain ID
    chain_id: u64,
    /// Current slot/block number
    slot: u64,
    /// Current timestamp
    timestamp: u64,
    /// Account state (shared between VMs)
    accounts: Arc<RwLock<HashMap<Pubkey, Account>>>,
    /// EVM bridge for cross-VM calls
    evm_bridge: EvmBridge,
    /// Compute budget per transaction
    compute_budget: u64,
}

impl HybridExecutor {
    pub fn new(chain_id: u64) -> Self {
        Self {
            chain_id,
            slot: 0,
            timestamp: 0,
            accounts: Arc::new(RwLock::new(HashMap::new())),
            evm_bridge: EvmBridge::new(chain_id),
            compute_budget: DEFAULT_COMPUTE_UNITS,
        }
    }

    /// Set block context for both VMs
    pub fn set_block(&mut self, slot: u64, timestamp: u64) {
        self.slot = slot;
        self.timestamp = timestamp;
        self.evm_bridge.set_block(slot, timestamp);
    }

    /// Load an account into the shared state
    pub fn load_account(&mut self, pubkey: Pubkey, account: Account) {
        let mut accounts = self.accounts.write().unwrap();
        accounts.insert(pubkey, account);
    }

    /// Get an account from the shared state
    pub fn get_account(&self, pubkey: &Pubkey) -> Option<Account> {
        let accounts = self.accounts.read().unwrap();
        accounts.get(pubkey).cloned()
    }

    /// Execute a Solana transaction (may invoke EVM via bridge)
    pub fn execute_svm_transaction(&mut self, tx: &Transaction) -> RuntimeResult<HybridExecutionResult> {
        let mut result = HybridExecutionResult::default();
        let mut compute_units = self.compute_budget;

        // Get program account
        let program_id = tx.message.account_keys
            .get(tx.message.header.num_required_signatures as usize)
            .ok_or_else(|| RuntimeError::Program("missing program account".into()))?;

        let program_account = self.get_account(program_id)
            .ok_or_else(|| RuntimeError::AccountNotFound(*program_id))?;

        // Check if this is a native program or BPF program
        if self.is_native_program(program_id) {
            // Execute native program
            self.execute_native(program_id, tx, &mut result)?;
        } else if program_account.executable {
            // Execute BPF program
            self.execute_bpf(&program_account.data, tx, &mut result, &mut compute_units)?;
        } else {
            return Err(RuntimeError::ProgramNotExecutable);
        }

        result.compute_units_used = self.compute_budget - compute_units;
        result.success = true;

        Ok(result)
    }

    /// Execute an EVM transaction
    pub fn execute_evm_transaction(
        &mut self,
        caller: EvmAddress,
        to: Option<EvmAddress>,
        value: u128,
        data: Vec<u8>,
        gas_limit: u64,
    ) -> RuntimeResult<HybridExecutionResult> {
        let mut result = HybridExecutionResult::default();

        // Sync SVM accounts to EVM state
        let accounts: Vec<(Pubkey, Account)> = {
            let acc = self.accounts.read().unwrap();
            acc.iter().map(|(k, v)| (*k, v.clone())).collect()
        };
        self.evm_bridge.load_accounts(accounts);

        // Execute EVM call
        let caller_pubkey = AddressMapper::derive_svm_pubkey(&caller);
        let evm_result = match to {
            Some(to_addr) => {
                self.evm_bridge.call_evm(&caller_pubkey, &to_addr, &data, value)?
            }
            None => {
                // Contract deployment
                let contract_addr = self.evm_bridge.deploy_evm_contract(&caller_pubkey, &data, value)?;
                contract_addr.to_vec()
            }
        };

        // Sync EVM state changes back to SVM
        for (pubkey, account) in self.evm_bridge.get_modified_accounts() {
            self.load_account(pubkey, account);
        }

        result.output = evm_result;
        result.success = true;
        result.compute_units_used = gas_limit; // TODO: Get actual gas used

        Ok(result)
    }

    /// Check if pubkey is a native program
    fn is_native_program(&self, pubkey: &Pubkey) -> bool {
        // System program
        if pubkey.0 == [0u8; 32] {
            return true;
        }
        // Check for known native program IDs
        let native_programs = [
            "11111111111111111111111111111111", // System
            "BPFLoader2111111111111111111111111111111111",
            "NativeLoader1111111111111111111111111111111",
        ];
        let pubkey_str = bs58::encode(&pubkey.0).into_string();
        native_programs.contains(&pubkey_str.as_str())
    }

    /// Execute a native program
    fn execute_native(
        &mut self,
        _program_id: &Pubkey,
        _tx: &Transaction,
        _result: &mut HybridExecutionResult,
    ) -> RuntimeResult<()> {
        // Native program execution is handled separately
        // This is a simplified stub
        Ok(())
    }

    /// Execute BPF bytecode
    fn execute_bpf(
        &mut self,
        bytecode: &[u8],
        tx: &Transaction,
        result: &mut HybridExecutionResult,
        compute_units: &mut u64,
    ) -> RuntimeResult<()> {
        // Serialize input data for BPF program
        let input = self.serialize_bpf_input(tx)?;

        // Create BPF VM
        let mut vm = BpfVm::new(bytecode, input)?;
        vm.set_compute_units(*compute_units);

        // Get mutable account references
        let program_id = tx.message.account_keys
            .get(tx.message.header.num_required_signatures as usize)
            .ok_or_else(|| RuntimeError::Program("missing program".into()))?;

        let mut accounts: Vec<Account> = tx.message.account_keys
            .iter()
            .filter_map(|pk| self.get_account(pk))
            .collect();

        // Execute
        let return_value = vm.execute(&mut accounts, *program_id)?;

        // Update compute units
        *compute_units = vm.compute_units();

        // Collect logs
        result.logs = vm.logs().to_vec();
        result.output = return_value.to_le_bytes().to_vec();

        // Write back modified accounts
        for (i, pk) in tx.message.account_keys.iter().enumerate() {
            if i < accounts.len() {
                self.load_account(*pk, accounts[i].clone());
            }
        }

        Ok(())
    }

    /// Serialize transaction data for BPF input
    fn serialize_bpf_input(&self, tx: &Transaction) -> RuntimeResult<Vec<u8>> {
        // Simplified serialization - real implementation would follow
        // Solana's BPF input format
        let mut input = Vec::new();

        // Number of accounts
        input.extend_from_slice(&(tx.message.account_keys.len() as u64).to_le_bytes());

        // Account data
        for pubkey in &tx.message.account_keys {
            // Pubkey
            input.extend_from_slice(&pubkey.0);

            // Account info (if exists)
            if let Some(account) = self.get_account(pubkey) {
                // Lamports
                input.extend_from_slice(&account.lamports.to_le_bytes());
                // Data length
                input.extend_from_slice(&(account.data.len() as u64).to_le_bytes());
                // Data
                input.extend_from_slice(&account.data);
                // Owner
                input.extend_from_slice(&account.owner.0);
                // Executable
                input.push(if account.executable { 1 } else { 0 });
                // Rent epoch
                input.extend_from_slice(&account.rent_epoch.to_le_bytes());
            } else {
                // Zero account
                input.extend_from_slice(&0u64.to_le_bytes()); // lamports
                input.extend_from_slice(&0u64.to_le_bytes()); // data_len
                input.extend_from_slice(&[0u8; 32]); // owner
                input.push(0); // executable
                input.extend_from_slice(&0u64.to_le_bytes()); // rent_epoch
            }
        }

        // Instruction data
        if let Some(ix) = tx.message.instructions.first() {
            input.extend_from_slice(&(ix.data.len() as u64).to_le_bytes());
            input.extend_from_slice(&ix.data);
        }

        Ok(input)
    }

    /// Cross-VM call: SVM -> EVM
    pub fn svm_call_evm(
        &mut self,
        caller_pubkey: &Pubkey,
        evm_contract: &EvmAddress,
        calldata: &[u8],
        value: u128,
    ) -> RuntimeResult<Vec<u8>> {
        // Sync accounts
        let accounts: Vec<(Pubkey, Account)> = {
            let acc = self.accounts.read().unwrap();
            acc.iter().map(|(k, v)| (*k, v.clone())).collect()
        };
        self.evm_bridge.load_accounts(accounts);

        // Make EVM call
        let result = self.evm_bridge.call_evm(caller_pubkey, evm_contract, calldata, value)?;

        // Sync back
        for (pubkey, account) in self.evm_bridge.get_modified_accounts() {
            self.load_account(pubkey, account);
        }

        Ok(result)
    }

    /// Cross-VM call: EVM -> SVM
    ///
    /// This would be called from an EVM precompile
    pub fn evm_call_svm(
        &mut self,
        _evm_caller: &EvmAddress,
        _svm_program: &Pubkey,
        _instruction_data: &[u8],
    ) -> RuntimeResult<Vec<u8>> {
        // TODO: Build Solana transaction and execute
        // This is the inverse direction - EVM calling SVM
        Err(RuntimeError::CrossVmCallFailed("EVM->SVM not yet implemented".into()))
    }
}

/// Result of hybrid execution
#[derive(Debug, Clone, Default)]
pub struct HybridExecutionResult {
    pub success: bool,
    pub output: Vec<u8>,
    pub compute_units_used: u64,
    pub logs: Vec<String>,
    pub state_changes: Vec<(Pubkey, Account)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_executor_creation() {
        let executor = HybridExecutor::new(43114);
        assert_eq!(executor.chain_id, 43114);
    }

    #[test]
    fn test_account_operations() {
        let mut executor = HybridExecutor::new(1);

        let pubkey = Pubkey([1u8; 32]);
        let account = Account {
            lamports: 1000,
            data: vec![],
            owner: Pubkey([0u8; 32]),
            executable: false,
            rent_epoch: 0,
        };

        executor.load_account(pubkey, account.clone());
        let loaded = executor.get_account(&pubkey).unwrap();
        assert_eq!(loaded.lamports, 1000);
    }

    #[test]
    fn test_is_native_program() {
        let executor = HybridExecutor::new(1);

        // System program (all zeros)
        let system = Pubkey([0u8; 32]);
        assert!(executor.is_native_program(&system));

        // Random program
        let random = Pubkey([42u8; 32]);
        assert!(!executor.is_native_program(&random));
    }
}
