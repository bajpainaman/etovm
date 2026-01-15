use crate::accounts::{AccountsDB, AccountsManager, InMemoryAccountsDB};
use crate::error::{RuntimeError, RuntimeResult};
use crate::programs::{BpfLoaderProgram, SystemProgram};
use crate::sysvars::{ids as sysvar_ids, Clock, EpochSchedule, Fees, Rent};
use crate::types::{Account, CompiledInstruction, Pubkey, Transaction, TransactionError};
use std::collections::HashMap;

/// Configuration for transaction execution
#[derive(Clone, Debug)]
pub struct ExecutorConfig {
    /// Maximum compute units per transaction
    pub max_compute_units: u64,
    /// Lamports per compute unit (for priority fees)
    pub lamports_per_compute_unit: u64,
    /// Base fee per signature
    pub lamports_per_signature: u64,
    /// Maximum CPI depth
    pub max_cpi_depth: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_compute_units: 1_400_000,
            lamports_per_compute_unit: 0,
            lamports_per_signature: 5000,
            max_cpi_depth: 4,
        }
    }
}

/// Transaction execution context
pub struct ExecutionContext {
    /// Current slot (block height)
    pub slot: u64,
    /// Block timestamp
    pub timestamp: i64,
    /// Recent blockhashes for validation
    pub recent_blockhashes: Vec<[u8; 32]>,
    /// Compute units remaining
    pub compute_units_remaining: u64,
    /// CPI depth
    pub cpi_depth: usize,
    /// Configuration
    pub config: ExecutorConfig,
}

impl ExecutionContext {
    pub fn new(slot: u64, timestamp: i64, config: ExecutorConfig) -> Self {
        Self {
            slot,
            timestamp,
            recent_blockhashes: vec![],
            compute_units_remaining: config.max_compute_units,
            cpi_depth: 0,
            config,
        }
    }

    pub fn add_blockhash(&mut self, blockhash: [u8; 32]) {
        self.recent_blockhashes.push(blockhash);
        if self.recent_blockhashes.len() > 300 {
            self.recent_blockhashes.remove(0);
        }
    }

    pub fn is_valid_blockhash(&self, blockhash: &[u8; 32]) -> bool {
        self.recent_blockhashes.contains(blockhash)
    }

    pub fn consume_compute_units(&mut self, units: u64) -> RuntimeResult<()> {
        if units > self.compute_units_remaining {
            return Err(RuntimeError::ComputeBudgetExceeded);
        }
        self.compute_units_remaining -= units;
        Ok(())
    }
}

/// Transaction execution result
#[derive(Debug)]
pub struct ExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Compute units used
    pub compute_units_used: u64,
    /// Fee charged
    pub fee: u64,
    /// State changes (pubkey -> new account state)
    pub state_changes: HashMap<Pubkey, Account>,
    /// Logs generated during execution
    pub logs: Vec<String>,
    /// Error if failed
    pub error: Option<RuntimeError>,
}

/// Transaction executor
pub struct Executor<DB: AccountsDB> {
    accounts: AccountsManager<DB>,
    system_program: SystemProgram,
    bpf_loader: BpfLoaderProgram,
    config: ExecutorConfig,
}

impl<DB: AccountsDB> Executor<DB> {
    pub fn new(accounts: AccountsManager<DB>, config: ExecutorConfig) -> Self {
        Self {
            accounts,
            system_program: SystemProgram::new(Rent::default()),
            bpf_loader: BpfLoaderProgram::new(),
            config,
        }
    }

    /// Execute a transaction
    pub fn execute_transaction(
        &self,
        tx: &Transaction,
        ctx: &mut ExecutionContext,
    ) -> ExecutionResult {
        let mut result = ExecutionResult {
            success: false,
            compute_units_used: 0,
            fee: 0,
            state_changes: HashMap::new(),
            logs: vec![],
            error: None,
        };

        // 1. Verify signatures
        if let Err(e) = tx.verify() {
            result.error = Some(RuntimeError::Transaction(e));
            return result;
        }

        // 2. Validate blockhash
        if !ctx.is_valid_blockhash(&tx.message.recent_blockhash) {
            result.error = Some(RuntimeError::Transaction(TransactionError::BlockhashNotFound));
            return result;
        }

        // 3. Load accounts
        let mut loaded_accounts: Vec<(Pubkey, Account)> = match self.load_accounts(tx) {
            Ok(accounts) => accounts,
            Err(e) => {
                result.error = Some(e);
                return result;
            }
        };

        // 4. Calculate and charge fee
        let fee = self.calculate_fee(tx);
        result.fee = fee;

        if let Some(fee_payer) = loaded_accounts.first_mut() {
            if fee_payer.1.lamports < fee {
                result.error = Some(RuntimeError::Transaction(
                    TransactionError::InsufficientFundsForFee,
                ));
                return result;
            }
            fee_payer.1.lamports -= fee;
        }

        // 5. Execute instructions
        let initial_compute_units = ctx.compute_units_remaining;

        for (ix_index, instruction) in tx.message.instructions.iter().enumerate() {
            match self.execute_instruction(
                instruction,
                &tx.message.account_keys,
                &mut loaded_accounts,
                ctx,
            ) {
                Ok(()) => {
                    result
                        .logs
                        .push(format!("Instruction {} succeeded", ix_index));
                }
                Err(e) => {
                    result
                        .logs
                        .push(format!("Instruction {} failed: {:?}", ix_index, e));
                    result.error = Some(e);
                    result.compute_units_used = initial_compute_units - ctx.compute_units_remaining;
                    return result;
                }
            }
        }

        // 6. Success - record state changes
        result.success = true;
        result.compute_units_used = initial_compute_units - ctx.compute_units_remaining;

        for (pubkey, account) in loaded_accounts {
            result.state_changes.insert(pubkey, account);
        }

        result
    }

    fn load_accounts(&self, tx: &Transaction) -> RuntimeResult<Vec<(Pubkey, Account)>> {
        let mut accounts = Vec::with_capacity(tx.message.account_keys.len());

        for pubkey in &tx.message.account_keys {
            let account = self.accounts.get_account_or_default(pubkey)?;
            accounts.push((*pubkey, account));
        }

        self.inject_sysvars(&mut accounts);

        Ok(accounts)
    }

    fn inject_sysvars(&self, accounts: &mut Vec<(Pubkey, Account)>) {
        let clock = Clock::default();
        let rent = Rent::default();
        let epoch_schedule = EpochSchedule::default();
        let fees = Fees::default();

        let sysvars = [
            (*sysvar_ids::CLOCK, clock.to_account_data()),
            (*sysvar_ids::RENT, rent.to_account_data()),
            (*sysvar_ids::EPOCH_SCHEDULE, epoch_schedule.to_account_data()),
            (*sysvar_ids::FEES, fees.to_account_data()),
        ];

        for (pubkey, data) in sysvars {
            if !accounts.iter().any(|(pk, _)| pk == &pubkey) {
                let account = Account {
                    lamports: 1,
                    data,
                    owner: Pubkey::system_program(),
                    executable: false,
                    rent_epoch: 0,
                };
                accounts.push((pubkey, account));
            }
        }
    }

    fn calculate_fee(&self, tx: &Transaction) -> u64 {
        let num_signatures = tx.signatures.len() as u64;
        num_signatures * self.config.lamports_per_signature
    }

    fn execute_instruction(
        &self,
        instruction: &CompiledInstruction,
        account_keys: &[Pubkey],
        accounts: &mut Vec<(Pubkey, Account)>,
        ctx: &mut ExecutionContext,
    ) -> RuntimeResult<()> {
        // Consume base compute units
        ctx.consume_compute_units(200)?;

        let program_id = &account_keys[instruction.program_id_index as usize];

        // Get accounts referenced by this instruction
        let instruction_accounts: Vec<usize> =
            instruction.accounts.iter().map(|&idx| idx as usize).collect();

        // Build mutable account slice for processing
        let mut instruction_account_data: Vec<(Pubkey, Account)> = instruction_accounts
            .iter()
            .map(|&idx| accounts[idx].clone())
            .collect();

        // Execute based on program type
        let result = if *program_id == Pubkey::system_program() {
            self.system_program
                .process_instruction(&mut instruction_account_data, &instruction.data)
        } else if *program_id == Pubkey::bpf_loader() {
            self.bpf_loader
                .process_instruction(&mut instruction_account_data, &instruction.data)
        } else {
            // Check if it's an executable account
            let program_account = accounts
                .iter()
                .find(|(pk, _)| pk == program_id)
                .map(|(_, acc)| acc);

            match program_account {
                Some(acc) if acc.executable => {
                    // TODO: Execute BPF bytecode
                    Err(RuntimeError::Program(
                        "BPF execution not yet implemented".into(),
                    ))
                }
                Some(_) => Err(RuntimeError::ProgramNotExecutable),
                None => Err(RuntimeError::AccountNotFound(*program_id)),
            }
        };

        // Write back account changes
        if result.is_ok() {
            for (i, &account_idx) in instruction_accounts.iter().enumerate() {
                accounts[account_idx] = instruction_account_data[i].clone();
            }
        }

        result
    }

    /// Commit execution results to account database
    pub fn commit(&self, result: &ExecutionResult) -> RuntimeResult<()> {
        if !result.success {
            return Ok(());
        }

        let accounts: Vec<(Pubkey, Account)> = result
            .state_changes
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        self.accounts.commit_accounts(&accounts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programs::system::instruction as system_ix;
    use crate::types::Message;

    fn create_test_executor() -> Executor<InMemoryAccountsDB> {
        let accounts = AccountsManager::new(InMemoryAccountsDB::new());
        Executor::new(accounts, ExecutorConfig::default())
    }

    #[test]
    fn test_executor_creation() {
        let executor = create_test_executor();
        assert_eq!(executor.config.max_compute_units, 1_400_000);
    }
}
