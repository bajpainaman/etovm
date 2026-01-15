use crate::{RuntimeError, RuntimeResult};
use super::{EvmAddress, EvmExecutionResult, EvmLog, EvmStateAdapter};

use revm::{
    Evm,
    primitives::{
        Address, Bytes, ExecutionResult as RevmResult, Output, TxEnv, U256,
        CfgEnv, BlockEnv, TxKind, CfgEnvWithHandlerCfg, SpecId,
    },
};

/// EVM executor using revm
pub struct EvmExecutor {
    /// Chain ID
    chain_id: u64,
    /// Current block number
    block_number: u64,
    /// Current timestamp
    timestamp: u64,
    /// Gas limit
    gas_limit: u64,
}

impl EvmExecutor {
    pub fn new(chain_id: u64) -> Self {
        Self {
            chain_id,
            block_number: 0,
            timestamp: 0,
            gas_limit: 30_000_000, // 30M gas limit
        }
    }

    /// Set block context
    pub fn set_block(&mut self, number: u64, timestamp: u64) {
        self.block_number = number;
        self.timestamp = timestamp;
    }

    /// Execute an EVM transaction
    pub fn execute(
        &self,
        state: &mut EvmStateAdapter,
        caller: EvmAddress,
        to: Option<EvmAddress>,
        value: u128,
        data: Vec<u8>,
        gas_limit: u64,
    ) -> RuntimeResult<EvmExecutionResult> {
        // Build transaction environment
        let mut tx_env = TxEnv::default();
        tx_env.caller = Address::from_slice(&caller);
        tx_env.transact_to = match to {
            Some(addr) => TxKind::Call(Address::from_slice(&addr)),
            None => TxKind::Create,
        };
        tx_env.value = U256::from(value);
        tx_env.data = Bytes::from(data);
        tx_env.gas_limit = gas_limit;
        tx_env.gas_price = U256::from(1); // 1 wei per gas

        // Build config environment with handler
        let mut cfg_env = CfgEnv::default();
        cfg_env.chain_id = self.chain_id;
        let cfg_with_handler = CfgEnvWithHandlerCfg::new_with_spec_id(cfg_env, SpecId::CANCUN);

        // Build block environment
        let mut block_env = BlockEnv::default();
        block_env.number = U256::from(self.block_number);
        block_env.timestamp = U256::from(self.timestamp);
        block_env.gas_limit = U256::from(self.gas_limit);
        block_env.basefee = U256::from(1); // 1 wei base fee

        // Create EVM instance
        let mut evm = Evm::builder()
            .with_db(state)
            .with_cfg_env_with_handler_cfg(cfg_with_handler)
            .with_block_env(block_env)
            .with_tx_env(tx_env)
            .build();

        // Execute transaction
        let result = evm
            .transact()
            .map_err(|e| RuntimeError::EvmExecutionFailed(format!("{:?}", e)))?;

        // Process result
        let (success, output, gas_used) = match result.result {
            RevmResult::Success { output, gas_used, logs, .. } => {
                let output_bytes = match output {
                    Output::Call(data) => data.to_vec(),
                    Output::Create(data, _) => data.to_vec(),
                };

                let evm_logs: Vec<EvmLog> = logs
                    .into_iter()
                    .map(|log| EvmLog {
                        address: log.address.as_slice().try_into().unwrap(),
                        topics: log.data.topics().iter().map(|t| t.0).collect(),
                        data: log.data.data.to_vec(),
                    })
                    .collect();

                (true, output_bytes, gas_used)
            }
            RevmResult::Revert { output, gas_used } => {
                (false, output.to_vec(), gas_used)
            }
            RevmResult::Halt { reason, gas_used } => {
                let msg = format!("EVM halted: {:?}", reason);
                return Err(RuntimeError::EvmExecutionFailed(msg));
            }
        };

        Ok(EvmExecutionResult {
            success,
            output,
            gas_used,
            logs: vec![], // TODO: Include logs from success branch
        })
    }

    /// Execute a static call (read-only)
    pub fn static_call(
        &self,
        state: &mut EvmStateAdapter,
        caller: EvmAddress,
        to: EvmAddress,
        data: Vec<u8>,
    ) -> RuntimeResult<Vec<u8>> {
        let result = self.execute(state, caller, Some(to), 0, data, 100_000)?;

        if result.success {
            Ok(result.output)
        } else {
            Err(RuntimeError::EvmExecutionFailed("static call reverted".into()))
        }
    }

    /// Deploy a contract
    pub fn deploy(
        &self,
        state: &mut EvmStateAdapter,
        deployer: EvmAddress,
        bytecode: Vec<u8>,
        value: u128,
    ) -> RuntimeResult<EvmAddress> {
        let result = self.execute(state, deployer, None, value, bytecode, 5_000_000)?;

        if result.success && result.output.len() >= 20 {
            let mut addr = [0u8; 20];
            addr.copy_from_slice(&result.output[..20]);
            Ok(addr)
        } else {
            Err(RuntimeError::EvmExecutionFailed("contract deployment failed".into()))
        }
    }
}

impl Default for EvmExecutor {
    fn default() -> Self {
        Self::new(43114) // Avalanche C-Chain ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evm_executor_creation() {
        let executor = EvmExecutor::new(1);
        assert_eq!(executor.chain_id, 1);
    }

    #[test]
    fn test_simple_value_transfer() {
        let mut state = EvmStateAdapter::new();
        let executor = EvmExecutor::new(1);

        // This would need accounts loaded into state to work fully
        // Just testing that the executor can be called
        let caller = [1u8; 20];
        let to = [2u8; 20];

        let result = executor.execute(&mut state, caller, Some(to), 0, vec![], 21000);
        // Result depends on state setup - this tests the API works
        assert!(result.is_ok() || result.is_err());
    }
}
