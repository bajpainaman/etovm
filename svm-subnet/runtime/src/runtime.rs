use crate::accounts::{AccountsDB, AccountsManager, InMemoryAccountsDB};
use crate::error::RuntimeResult;
use crate::executor::{ExecutionContext, ExecutionResult, Executor, ExecutorConfig};
use crate::sysvars::Clock;
use crate::types::{Account, Pubkey, Transaction};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Main SVM runtime
pub struct SvmRuntime<DB: AccountsDB> {
    executor: Executor<DB>,
    context: Arc<RwLock<ExecutionContext>>,
    config: RuntimeConfig,
}

/// Runtime configuration
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub executor_config: ExecutorConfig,
    pub slots_per_epoch: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            executor_config: ExecutorConfig::default(),
            slots_per_epoch: 432000,
        }
    }
}

impl SvmRuntime<InMemoryAccountsDB> {
    /// Create a new runtime with in-memory storage (for testing)
    pub fn new_in_memory(config: RuntimeConfig) -> Self {
        let accounts = AccountsManager::new(InMemoryAccountsDB::new());
        let executor = Executor::new(accounts, config.executor_config.clone());
        let context = Arc::new(RwLock::new(ExecutionContext::new(
            0,
            0,
            config.executor_config.clone(),
        )));

        Self {
            executor,
            context,
            config,
        }
    }
}

impl<DB: AccountsDB> SvmRuntime<DB> {
    /// Create runtime with custom accounts database
    pub fn new(accounts_db: DB, config: RuntimeConfig) -> Self {
        let accounts = AccountsManager::new(accounts_db);
        let executor = Executor::new(accounts, config.executor_config.clone());
        let context = Arc::new(RwLock::new(ExecutionContext::new(
            0,
            0,
            config.executor_config.clone(),
        )));

        Self {
            executor,
            context,
            config,
        }
    }

    /// Update block context (called when new block is being built)
    pub fn set_block_context(
        &self,
        slot: u64,
        timestamp: i64,
        blockhash: [u8; 32],
    ) -> RuntimeResult<()> {
        let mut ctx = self
            .context
            .write()
            .map_err(|e| crate::error::RuntimeError::State(e.to_string()))?;

        ctx.slot = slot;
        ctx.timestamp = timestamp;
        ctx.compute_units_remaining = self.config.executor_config.max_compute_units;
        ctx.add_blockhash(blockhash);

        Ok(())
    }

    /// Execute a single transaction
    pub fn execute_transaction(&self, tx: &Transaction) -> RuntimeResult<ExecutionResult> {
        let mut ctx = self
            .context
            .write()
            .map_err(|e| crate::error::RuntimeError::State(e.to_string()))?;

        Ok(self.executor.execute_transaction(tx, &mut ctx))
    }

    /// Execute a batch of transactions
    pub fn execute_transactions(&self, txs: &[Transaction]) -> Vec<ExecutionResult> {
        let mut results = Vec::with_capacity(txs.len());

        for tx in txs {
            match self.execute_transaction(tx) {
                Ok(result) => results.push(result),
                Err(e) => {
                    results.push(ExecutionResult {
                        success: false,
                        compute_units_used: 0,
                        fee: 0,
                        state_changes: HashMap::new(),
                        logs: vec![format!("Execution error: {:?}", e)],
                        error: Some(e),
                    });
                }
            }
        }

        results
    }

    /// Commit execution results to state
    pub fn commit(&self, result: &ExecutionResult) -> RuntimeResult<()> {
        self.executor.commit(result)
    }

    /// Get account
    pub fn get_account(&self, _pubkey: &Pubkey) -> RuntimeResult<Option<Account>> {
        // TODO: Implement proper account access
        Ok(None)
    }

    /// Get current slot
    pub fn current_slot(&self) -> RuntimeResult<u64> {
        let ctx = self
            .context
            .read()
            .map_err(|e| crate::error::RuntimeError::State(e.to_string()))?;
        Ok(ctx.slot)
    }

    /// Get current blockhash
    pub fn current_blockhash(&self) -> RuntimeResult<[u8; 32]> {
        let ctx = self
            .context
            .read()
            .map_err(|e| crate::error::RuntimeError::State(e.to_string()))?;
        Ok(ctx.recent_blockhashes.last().copied().unwrap_or([0u8; 32]))
    }

    /// Get clock sysvar
    pub fn get_clock(&self) -> RuntimeResult<Clock> {
        let ctx = self
            .context
            .read()
            .map_err(|e| crate::error::RuntimeError::State(e.to_string()))?;
        Ok(Clock::from_avalanche_block(
            ctx.slot,
            ctx.timestamp,
            self.config.slots_per_epoch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = SvmRuntime::new_in_memory(RuntimeConfig::default());
        assert_eq!(runtime.current_slot().unwrap(), 0);
    }

    #[test]
    fn test_set_block_context() {
        let runtime = SvmRuntime::new_in_memory(RuntimeConfig::default());
        runtime
            .set_block_context(100, 1700000000, [1u8; 32])
            .unwrap();
        assert_eq!(runtime.current_slot().unwrap(), 100);
    }
}
