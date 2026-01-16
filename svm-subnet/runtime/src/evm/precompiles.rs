//! EVM Precompiles for SVM Interoperability
//!
//! This module provides seamless EVM ↔ SVM bridging through precompile contracts.
//! ERC20 calls to the Token precompile are routed to SPL Token with zero overhead.
//!
//! Precompile Addresses:
//! - 0x100: SVM Bridge (arbitrary program calls)
//! - 0x101: System Program
//! - 0x102: SPL Token (ERC20 compatible)
//! - 0x103: Associated Token Account

use crate::error::{RuntimeError, RuntimeResult};
use crate::programs::token::{TokenProgram, TokenAccount, Mint, TokenInstruction, AccountState};
use crate::types::{Account, Pubkey};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Precompile address range (0x100 - 0x1FF)
pub const PRECOMPILE_START: u8 = 0x01;
pub const PRECOMPILE_END: u8 = 0xFF;

/// EVM address type
pub type EvmAddress = [u8; 20];

/// Precompile addresses
pub mod addresses {
    use super::EvmAddress;

    /// Generic SVM Bridge - call any program
    pub const SVM_BRIDGE: EvmAddress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x00];

    /// System Program precompile
    pub const SYSTEM_PROGRAM: EvmAddress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x01];

    /// SPL Token precompile (ERC20 compatible)
    pub const SPL_TOKEN: EvmAddress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x02];

    /// Associated Token Account precompile
    pub const ATA: EvmAddress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x03];

    /// Check if address is a precompile
    pub fn is_precompile(addr: &EvmAddress) -> bool {
        // Check if address is in range 0x100-0x1FF
        addr[0..17] == [0u8; 17] && addr[17] == 0 && addr[18] == 0x01
    }
}

/// ERC20 function selectors (keccak256 first 4 bytes)
pub mod erc20_selectors {
    /// name() -> string
    pub const NAME: [u8; 4] = [0x06, 0xfd, 0xde, 0x03];
    /// symbol() -> string
    pub const SYMBOL: [u8; 4] = [0x95, 0xd8, 0x9b, 0x41];
    /// decimals() -> uint8
    pub const DECIMALS: [u8; 4] = [0x31, 0x3c, 0xe5, 0x67];
    /// totalSupply() -> uint256
    pub const TOTAL_SUPPLY: [u8; 4] = [0x18, 0x16, 0x0d, 0xdd];
    /// balanceOf(address) -> uint256
    pub const BALANCE_OF: [u8; 4] = [0x70, 0xa0, 0x82, 0x31];
    /// transfer(address,uint256) -> bool
    pub const TRANSFER: [u8; 4] = [0xa9, 0x05, 0x9c, 0xbb];
    /// approve(address,uint256) -> bool
    pub const APPROVE: [u8; 4] = [0x09, 0x5e, 0xa7, 0xb3];
    /// allowance(address,address) -> uint256
    pub const ALLOWANCE: [u8; 4] = [0xdd, 0x62, 0xed, 0x3e];
    /// transferFrom(address,address,uint256) -> bool
    pub const TRANSFER_FROM: [u8; 4] = [0x23, 0xb8, 0x72, 0xdd];

    // Extended functions
    /// mint(address,uint256) -> bool (requires mint authority)
    pub const MINT: [u8; 4] = [0x40, 0xc1, 0x0f, 0x19];
    /// burn(uint256) -> bool
    pub const BURN: [u8; 4] = [0x42, 0x96, 0x6c, 0x68];
}

/// SVM Bridge function selectors
pub mod bridge_selectors {
    /// callSVM(bytes32 program, bytes data, bytes32[] accounts) -> bytes
    pub const CALL_SVM: [u8; 4] = [0xca, 0x11, 0x5f, 0x4d];
    /// getSVMAccount(bytes32 pubkey) -> (uint64 lamports, bytes data)
    pub const GET_ACCOUNT: [u8; 4] = [0x67, 0x65, 0x74, 0x41];
}

/// Precompile execution context
pub struct PrecompileContext<'a> {
    /// Caller's EVM address
    pub caller: EvmAddress,
    /// Value sent (in wei, converted to lamports)
    pub value: u128,
    /// SVM accounts state (shared with SVM runtime)
    pub accounts: &'a mut HashMap<Pubkey, Account>,
    /// Token program instance
    pub token_program: &'a TokenProgram,
    /// Current mint address (for token precompile)
    pub mint: Option<Pubkey>,
}

/// Precompile result
pub struct PrecompileResult {
    /// Success flag
    pub success: bool,
    /// Output data (ABI encoded)
    pub output: Vec<u8>,
    /// Gas used
    pub gas_used: u64,
}

/// Precompile registry and handler
pub struct PrecompileRegistry {
    /// Token program
    token_program: TokenProgram,
    /// Default mint for ERC20 operations (can be overridden per-call)
    default_mint: Option<Pubkey>,
    /// Token metadata cache
    token_metadata: HashMap<Pubkey, TokenMetadata>,
}

/// Cached token metadata for ERC20 responses
#[derive(Clone)]
pub struct TokenMetadata {
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
}

impl PrecompileRegistry {
    pub fn new() -> Self {
        Self {
            token_program: TokenProgram::new(),
            default_mint: None,
            token_metadata: HashMap::new(),
        }
    }

    /// Set default mint for ERC20 operations
    pub fn set_default_mint(&mut self, mint: Pubkey, metadata: TokenMetadata) {
        self.default_mint = Some(mint);
        self.token_metadata.insert(mint, metadata);
    }

    /// Check if address is handled by precompiles
    pub fn is_precompile(&self, addr: &EvmAddress) -> bool {
        addresses::is_precompile(addr)
    }

    /// Execute precompile call
    pub fn execute(
        &self,
        addr: &EvmAddress,
        caller: EvmAddress,
        value: u128,
        input: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if *addr == addresses::SPL_TOKEN {
            self.execute_token_precompile(caller, value, input, accounts)
        } else if *addr == addresses::SVM_BRIDGE {
            self.execute_bridge_precompile(caller, value, input, accounts)
        } else if *addr == addresses::SYSTEM_PROGRAM {
            self.execute_system_precompile(caller, value, input, accounts)
        } else {
            Err(RuntimeError::InvalidProgram)
        }
    }

    /// Execute SPL Token precompile (ERC20 interface)
    fn execute_token_precompile(
        &self,
        caller: EvmAddress,
        value: u128,
        input: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if input.len() < 4 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let selector: [u8; 4] = input[0..4].try_into().unwrap();
        let data = &input[4..];

        match selector {
            erc20_selectors::NAME => self.erc20_name(),
            erc20_selectors::SYMBOL => self.erc20_symbol(),
            erc20_selectors::DECIMALS => self.erc20_decimals(accounts),
            erc20_selectors::TOTAL_SUPPLY => self.erc20_total_supply(accounts),
            erc20_selectors::BALANCE_OF => self.erc20_balance_of(data, accounts),
            erc20_selectors::TRANSFER => self.erc20_transfer(caller, data, accounts),
            erc20_selectors::APPROVE => self.erc20_approve(caller, data, accounts),
            erc20_selectors::ALLOWANCE => self.erc20_allowance(data, accounts),
            erc20_selectors::TRANSFER_FROM => self.erc20_transfer_from(caller, data, accounts),
            erc20_selectors::MINT => self.erc20_mint(caller, data, accounts),
            erc20_selectors::BURN => self.erc20_burn(caller, data, accounts),
            _ => Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            }),
        }
    }

    // ==================== ERC20 Implementation ====================

    fn erc20_name(&self) -> RuntimeResult<PrecompileResult> {
        let name = self.default_mint
            .as_ref()
            .and_then(|m| self.token_metadata.get(m))
            .map(|m| m.name.clone())
            .unwrap_or_else(|| "SVM Token".to_string());

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_string(&name),
            gas_used: 200,
        })
    }

    fn erc20_symbol(&self) -> RuntimeResult<PrecompileResult> {
        let symbol = self.default_mint
            .as_ref()
            .and_then(|m| self.token_metadata.get(m))
            .map(|m| m.symbol.clone())
            .unwrap_or_else(|| "SVM".to_string());

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_string(&symbol),
            gas_used: 200,
        })
    }

    fn erc20_decimals(&self, accounts: &HashMap<Pubkey, Account>) -> RuntimeResult<PrecompileResult> {
        let decimals = if let Some(mint_pk) = &self.default_mint {
            if let Some(account) = accounts.get(mint_pk) {
                Mint::unpack(&account.data)
                    .map(|m| m.decimals)
                    .unwrap_or(9)
            } else {
                9
            }
        } else {
            9
        };

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_uint8(decimals),
            gas_used: 200,
        })
    }

    fn erc20_total_supply(&self, accounts: &HashMap<Pubkey, Account>) -> RuntimeResult<PrecompileResult> {
        let supply = if let Some(mint_pk) = &self.default_mint {
            if let Some(account) = accounts.get(mint_pk) {
                Mint::unpack(&account.data)
                    .map(|m| m.supply)
                    .unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_uint256(supply as u128),
            gas_used: 500,
        })
    }

    fn erc20_balance_of(
        &self,
        data: &[u8],
        accounts: &HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 32 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        // Decode address (last 20 bytes of 32-byte word)
        let evm_addr: EvmAddress = data[12..32].try_into().unwrap();
        let owner_pubkey = evm_to_svm_pubkey(&evm_addr);

        // Find token account for this owner
        let balance = self.find_token_account_balance(&owner_pubkey, accounts);

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_uint256(balance as u128),
            gas_used: 700,
        })
    }

    fn erc20_transfer(
        &self,
        caller: EvmAddress,
        data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 64 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        // Decode: to (address), amount (uint256)
        let to_addr: EvmAddress = data[12..32].try_into().unwrap();
        let amount = abi_decode_uint256(&data[32..64]);

        let from_pubkey = evm_to_svm_pubkey(&caller);
        let to_pubkey = evm_to_svm_pubkey(&to_addr);

        // Execute SPL Token transfer
        let result = self.spl_transfer(&from_pubkey, &to_pubkey, amount as u64, accounts);

        match result {
            Ok(()) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(true),
                gas_used: 5000,
            }),
            Err(_) => Ok(PrecompileResult {
                success: true, // ERC20 returns false, doesn't revert
                output: abi_encode_bool(false),
                gas_used: 3000,
            }),
        }
    }

    fn erc20_approve(
        &self,
        caller: EvmAddress,
        data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 64 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let spender_addr: EvmAddress = data[12..32].try_into().unwrap();
        let amount = abi_decode_uint256(&data[32..64]);

        let owner_pubkey = evm_to_svm_pubkey(&caller);
        let delegate_pubkey = evm_to_svm_pubkey(&spender_addr);

        let result = self.spl_approve(&owner_pubkey, &delegate_pubkey, amount as u64, accounts);

        match result {
            Ok(()) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(true),
                gas_used: 4000,
            }),
            Err(_) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(false),
                gas_used: 2000,
            }),
        }
    }

    fn erc20_allowance(
        &self,
        data: &[u8],
        accounts: &HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 64 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let owner_addr: EvmAddress = data[12..32].try_into().unwrap();
        let spender_addr: EvmAddress = data[44..64].try_into().unwrap();

        let owner_pubkey = evm_to_svm_pubkey(&owner_addr);
        let spender_pubkey = evm_to_svm_pubkey(&spender_addr);

        // Find token account and check delegated amount
        let allowance = self.find_allowance(&owner_pubkey, &spender_pubkey, accounts);

        Ok(PrecompileResult {
            success: true,
            output: abi_encode_uint256(allowance as u128),
            gas_used: 700,
        })
    }

    fn erc20_transfer_from(
        &self,
        caller: EvmAddress,
        data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 96 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let from_addr: EvmAddress = data[12..32].try_into().unwrap();
        let to_addr: EvmAddress = data[44..64].try_into().unwrap();
        let amount = abi_decode_uint256(&data[64..96]);

        let from_pubkey = evm_to_svm_pubkey(&from_addr);
        let to_pubkey = evm_to_svm_pubkey(&to_addr);
        let delegate_pubkey = evm_to_svm_pubkey(&caller);

        // Execute delegated transfer
        let result = self.spl_transfer_from(
            &from_pubkey,
            &to_pubkey,
            &delegate_pubkey,
            amount as u64,
            accounts,
        );

        match result {
            Ok(()) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(true),
                gas_used: 6000,
            }),
            Err(_) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(false),
                gas_used: 3000,
            }),
        }
    }

    fn erc20_mint(
        &self,
        caller: EvmAddress,
        data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 64 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let to_addr: EvmAddress = data[12..32].try_into().unwrap();
        let amount = abi_decode_uint256(&data[32..64]);

        let mint_authority = evm_to_svm_pubkey(&caller);
        let dest_pubkey = evm_to_svm_pubkey(&to_addr);

        let result = self.spl_mint_to(&mint_authority, &dest_pubkey, amount as u64, accounts);

        match result {
            Ok(()) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(true),
                gas_used: 5000,
            }),
            Err(_) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(false),
                gas_used: 2000,
            }),
        }
    }

    fn erc20_burn(
        &self,
        caller: EvmAddress,
        data: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 32 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let amount = abi_decode_uint256(&data[0..32]);
        let owner_pubkey = evm_to_svm_pubkey(&caller);

        let result = self.spl_burn(&owner_pubkey, amount as u64, accounts);

        match result {
            Ok(()) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(true),
                gas_used: 4000,
            }),
            Err(_) => Ok(PrecompileResult {
                success: true,
                output: abi_encode_bool(false),
                gas_used: 2000,
            }),
        }
    }

    // ==================== SPL Token Operations ====================

    fn find_token_account_balance(
        &self,
        owner: &Pubkey,
        accounts: &HashMap<Pubkey, Account>,
    ) -> u64 {
        let mint = match &self.default_mint {
            Some(m) => m,
            None => return 0,
        };

        // Derive token account address (simplified ATA derivation)
        let token_account_pk = derive_token_account(owner, mint);

        if let Some(account) = accounts.get(&token_account_pk) {
            TokenAccount::unpack(&account.data)
                .map(|ta| ta.amount)
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn find_allowance(
        &self,
        owner: &Pubkey,
        spender: &Pubkey,
        accounts: &HashMap<Pubkey, Account>,
    ) -> u64 {
        let mint = match &self.default_mint {
            Some(m) => m,
            None => return 0,
        };

        let token_account_pk = derive_token_account(owner, mint);

        if let Some(account) = accounts.get(&token_account_pk) {
            if let Ok(ta) = TokenAccount::unpack(&account.data) {
                if ta.delegate == Some(*spender) {
                    return ta.delegated_amount;
                }
            }
        }
        0
    }

    fn spl_transfer(
        &self,
        from: &Pubkey,
        to: &Pubkey,
        amount: u64,
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<()> {
        let mint = self.default_mint.ok_or(RuntimeError::InvalidMint)?;

        let from_token = derive_token_account(from, &mint);
        let to_token = derive_token_account(to, &mint);

        // Get or create accounts
        let from_account = accounts.get(&from_token)
            .ok_or(RuntimeError::AccountNotFound(from_token))?
            .clone();
        let to_account = accounts.get(&to_token)
            .ok_or(RuntimeError::AccountNotFound(to_token))?
            .clone();

        let mut svm_accounts = vec![
            (from_token, from_account),
            (to_token, to_account),
            (*from, Account::default()),
        ];

        self.token_program.process_transfer(&mut svm_accounts, amount, &[*from])?;

        // Write back modified accounts
        accounts.insert(svm_accounts[0].0, svm_accounts[0].1.clone());
        accounts.insert(svm_accounts[1].0, svm_accounts[1].1.clone());

        Ok(())
    }

    fn spl_approve(
        &self,
        owner: &Pubkey,
        delegate: &Pubkey,
        amount: u64,
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<()> {
        let mint = self.default_mint.ok_or(RuntimeError::InvalidMint)?;
        let token_account_pk = derive_token_account(owner, &mint);

        let token_account = accounts.get(&token_account_pk)
            .ok_or(RuntimeError::AccountNotFound(token_account_pk))?
            .clone();

        let mut svm_accounts = vec![
            (token_account_pk, token_account),
            (*delegate, Account::default()),
            (*owner, Account::default()),
        ];

        self.token_program.process_approve(&mut svm_accounts, amount, &[*owner])?;

        accounts.insert(svm_accounts[0].0, svm_accounts[0].1.clone());

        Ok(())
    }

    fn spl_transfer_from(
        &self,
        from: &Pubkey,
        to: &Pubkey,
        delegate: &Pubkey,
        amount: u64,
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<()> {
        let mint = self.default_mint.ok_or(RuntimeError::InvalidMint)?;

        let from_token = derive_token_account(from, &mint);
        let to_token = derive_token_account(to, &mint);

        let from_account = accounts.get(&from_token)
            .ok_or(RuntimeError::AccountNotFound(from_token))?
            .clone();
        let to_account = accounts.get(&to_token)
            .ok_or(RuntimeError::AccountNotFound(to_token))?
            .clone();

        let mut svm_accounts = vec![
            (from_token, from_account),
            (to_token, to_account),
            (*delegate, Account::default()),
        ];

        // Transfer using delegate authority
        self.token_program.process_transfer(&mut svm_accounts, amount, &[*delegate])?;

        accounts.insert(svm_accounts[0].0, svm_accounts[0].1.clone());
        accounts.insert(svm_accounts[1].0, svm_accounts[1].1.clone());

        Ok(())
    }

    fn spl_mint_to(
        &self,
        mint_authority: &Pubkey,
        dest: &Pubkey,
        amount: u64,
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<()> {
        let mint_pk = self.default_mint.ok_or(RuntimeError::InvalidMint)?;
        let dest_token = derive_token_account(dest, &mint_pk);

        let mint_account = accounts.get(&mint_pk)
            .ok_or(RuntimeError::AccountNotFound(mint_pk))?
            .clone();
        let dest_account = accounts.get(&dest_token)
            .ok_or(RuntimeError::AccountNotFound(dest_token))?
            .clone();

        let mut svm_accounts = vec![
            (mint_pk, mint_account),
            (dest_token, dest_account),
            (*mint_authority, Account::default()),
        ];

        self.token_program.process_mint_to(&mut svm_accounts, amount, &[*mint_authority])?;

        accounts.insert(svm_accounts[0].0, svm_accounts[0].1.clone());
        accounts.insert(svm_accounts[1].0, svm_accounts[1].1.clone());

        Ok(())
    }

    fn spl_burn(
        &self,
        owner: &Pubkey,
        amount: u64,
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<()> {
        let mint_pk = self.default_mint.ok_or(RuntimeError::InvalidMint)?;
        let token_account_pk = derive_token_account(owner, &mint_pk);

        let token_account = accounts.get(&token_account_pk)
            .ok_or(RuntimeError::AccountNotFound(token_account_pk))?
            .clone();
        let mint_account = accounts.get(&mint_pk)
            .ok_or(RuntimeError::AccountNotFound(mint_pk))?
            .clone();

        let mut svm_accounts = vec![
            (token_account_pk, token_account),
            (mint_pk, mint_account),
            (*owner, Account::default()),
        ];

        self.token_program.process_burn(&mut svm_accounts, amount, &[*owner])?;

        accounts.insert(svm_accounts[0].0, svm_accounts[0].1.clone());
        accounts.insert(svm_accounts[1].0, svm_accounts[1].1.clone());

        Ok(())
    }

    // ==================== Bridge Precompile ====================

    fn execute_bridge_precompile(
        &self,
        caller: EvmAddress,
        _value: u128,
        input: &[u8],
        accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if input.len() < 4 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let selector: [u8; 4] = input[0..4].try_into().unwrap();
        let data = &input[4..];

        match selector {
            bridge_selectors::CALL_SVM => self.bridge_call_svm(caller, data, accounts),
            bridge_selectors::GET_ACCOUNT => self.bridge_get_account(data, accounts),
            _ => Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            }),
        }
    }

    fn bridge_call_svm(
        &self,
        _caller: EvmAddress,
        data: &[u8],
        _accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        // callSVM(bytes32 program, bytes data, bytes32[] accounts)
        // This would parse the ABI-encoded call and route to the appropriate program
        // For now, return success with empty data

        if data.len() < 32 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        // Parse program ID
        let _program_id = Pubkey::new(data[0..32].try_into().unwrap());

        // TODO: Parse instruction data and accounts, execute program

        Ok(PrecompileResult {
            success: true,
            output: vec![],
            gas_used: 10000,
        })
    }

    fn bridge_get_account(
        &self,
        data: &[u8],
        accounts: &HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        if data.len() < 32 {
            return Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 200,
            });
        }

        let pubkey = Pubkey::new(data[0..32].try_into().unwrap());

        if let Some(account) = accounts.get(&pubkey) {
            // Encode: (uint64 lamports, bytes data)
            let mut output = Vec::with_capacity(64 + account.data.len());
            output.extend_from_slice(&[0u8; 24]);
            output.extend_from_slice(&account.lamports.to_be_bytes());
            output.extend_from_slice(&[0u8; 31]);
            output.push(64); // offset to data
            output.extend_from_slice(&[0u8; 24]);
            output.extend_from_slice(&(account.data.len() as u64).to_be_bytes());
            output.extend_from_slice(&account.data);

            Ok(PrecompileResult {
                success: true,
                output,
                gas_used: 500 + (account.data.len() as u64 / 32) * 100,
            })
        } else {
            Ok(PrecompileResult {
                success: false,
                output: vec![],
                gas_used: 500,
            })
        }
    }

    // ==================== System Precompile ====================

    fn execute_system_precompile(
        &self,
        _caller: EvmAddress,
        _value: u128,
        _input: &[u8],
        _accounts: &mut HashMap<Pubkey, Account>,
    ) -> RuntimeResult<PrecompileResult> {
        // System program precompile for account creation, transfers, etc.
        // TODO: Implement system program routing
        Ok(PrecompileResult {
            success: true,
            output: vec![],
            gas_used: 1000,
        })
    }
}

impl Default for PrecompileRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Helper Functions ====================

/// Convert EVM address to SVM pubkey (pad to 32 bytes)
pub fn evm_to_svm_pubkey(evm_addr: &EvmAddress) -> Pubkey {
    let mut bytes = [0u8; 32];
    bytes[12..32].copy_from_slice(evm_addr);
    Pubkey::new(bytes)
}

/// Convert SVM pubkey to EVM address (take last 20 bytes)
pub fn svm_to_evm_address(pubkey: &Pubkey) -> EvmAddress {
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&pubkey.0[12..32]);
    addr
}

/// Derive token account address (simplified PDA derivation)
pub fn derive_token_account(owner: &Pubkey, mint: &Pubkey) -> Pubkey {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(owner.0);
    hasher.update(mint.0);
    hasher.update(b"token_account");
    let hash = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    Pubkey::new(bytes)
}

// ==================== ABI Encoding Helpers ====================

fn abi_encode_bool(value: bool) -> Vec<u8> {
    let mut output = vec![0u8; 32];
    if value {
        output[31] = 1;
    }
    output
}

fn abi_encode_uint8(value: u8) -> Vec<u8> {
    let mut output = vec![0u8; 32];
    output[31] = value;
    output
}

fn abi_encode_uint256(value: u128) -> Vec<u8> {
    let mut output = vec![0u8; 32];
    output[16..32].copy_from_slice(&value.to_be_bytes());
    output
}

fn abi_encode_string(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let len = bytes.len();

    // Offset to string data (32 bytes)
    let mut output = vec![0u8; 32];
    output[31] = 32;

    // String length
    let mut len_word = vec![0u8; 32];
    len_word[28..32].copy_from_slice(&(len as u32).to_be_bytes());
    output.extend(len_word);

    // String data (padded to 32 bytes)
    output.extend_from_slice(bytes);
    let padding = (32 - (len % 32)) % 32;
    output.extend(vec![0u8; padding]);

    output
}

fn abi_decode_uint256(data: &[u8]) -> u128 {
    if data.len() < 32 {
        return 0;
    }
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&data[16..32]);
    u128::from_be_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_precompile() {
        assert!(addresses::is_precompile(&addresses::SPL_TOKEN));
        assert!(addresses::is_precompile(&addresses::SVM_BRIDGE));
        assert!(!addresses::is_precompile(&[0u8; 20]));
        assert!(!addresses::is_precompile(&[1u8; 20]));
    }

    #[test]
    fn test_evm_to_svm_pubkey() {
        let evm_addr = [0x42u8; 20];
        let pubkey = evm_to_svm_pubkey(&evm_addr);
        assert_eq!(&pubkey.0[12..32], &evm_addr);
        assert_eq!(&pubkey.0[0..12], &[0u8; 12]);
    }

    #[test]
    fn test_svm_to_evm_address() {
        let pubkey = Pubkey::new([0x42u8; 32]);
        let evm_addr = svm_to_evm_address(&pubkey);
        assert_eq!(evm_addr, [0x42u8; 20]);
    }

    #[test]
    fn test_abi_encode_bool() {
        let true_encoded = abi_encode_bool(true);
        assert_eq!(true_encoded.len(), 32);
        assert_eq!(true_encoded[31], 1);

        let false_encoded = abi_encode_bool(false);
        assert_eq!(false_encoded[31], 0);
    }

    #[test]
    fn test_abi_encode_uint256() {
        let encoded = abi_encode_uint256(1000);
        assert_eq!(encoded.len(), 32);
        assert_eq!(abi_decode_uint256(&encoded), 1000);
    }

    #[test]
    fn test_erc20_selectors() {
        // Verify known selectors
        use sha3::{Keccak256, Digest};

        let transfer_hash = Keccak256::digest(b"transfer(address,uint256)");
        assert_eq!(&transfer_hash[0..4], &erc20_selectors::TRANSFER);

        let balance_hash = Keccak256::digest(b"balanceOf(address)");
        assert_eq!(&balance_hash[0..4], &erc20_selectors::BALANCE_OF);
    }

    #[test]
    fn test_precompile_registry() {
        let registry = PrecompileRegistry::new();

        // Should recognize precompile addresses
        assert!(registry.is_precompile(&addresses::SPL_TOKEN));
        assert!(registry.is_precompile(&addresses::SVM_BRIDGE));
        assert!(!registry.is_precompile(&[0u8; 20]));
    }

    #[test]
    fn test_erc20_name_symbol() {
        let mut registry = PrecompileRegistry::new();
        let mint = Pubkey::new([1u8; 32]);
        registry.set_default_mint(mint, TokenMetadata {
            name: "Test Token".to_string(),
            symbol: "TEST".to_string(),
            decimals: 9,
        });

        let result = registry.erc20_name().unwrap();
        assert!(result.success);
        // Output should contain "Test Token" encoded as string

        let result = registry.erc20_symbol().unwrap();
        assert!(result.success);
    }
}
