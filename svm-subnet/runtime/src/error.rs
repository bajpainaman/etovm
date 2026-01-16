use crate::types::{Pubkey, TransactionError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    #[error("Account error: {0}")]
    Account(String),

    #[error("Program error: {0}")]
    Program(String),

    #[error("State error: {0}")]
    State(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("BPF execution error: {0}")]
    BpfExecution(String),

    #[error("Compute budget exceeded")]
    ComputeBudgetExceeded,

    #[error("Stack height exceeded")]
    StackHeightExceeded,

    #[error("Call depth exceeded")]
    CallDepthExceeded,

    #[error("Invalid program")]
    InvalidProgram,

    #[error("Program not executable")]
    ProgramNotExecutable,

    #[error("Account not found: {0}")]
    AccountNotFound(Pubkey),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // BPF interpreter errors
    #[error("Memory access violation at address 0x{0:x}")]
    MemoryAccessViolation(u64),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),

    #[error("Compute units exhausted")]
    ComputeExhausted,

    #[error("Invalid syscall: 0x{0:x}")]
    InvalidSyscall(u32),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Program aborted")]
    ProgramAborted,

    #[error("Program panic: {0}")]
    ProgramPanic(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    // EVM bridge errors
    #[error("EVM execution failed: {0}")]
    EvmExecutionFailed(String),

    #[error("Invalid EVM address")]
    InvalidEvmAddress,

    #[error("Cross-VM call failed: {0}")]
    CrossVmCallFailed(String),

    #[error("Mempool error: {0}")]
    Mempool(String),

    // Native program errors
    #[error("Not enough accounts provided")]
    NotEnoughAccounts,

    #[error("Account already initialized")]
    AccountAlreadyInitialized,

    #[error("Uninitialized account")]
    UninitializedAccount,

    #[error("Invalid mint")]
    InvalidMint,

    #[error("Account is frozen")]
    AccountFrozen,

    #[error("Invalid signer")]
    InvalidSigner,

    #[error("Insufficient funds")]
    InsufficientFunds,

    #[error("Arithmetic overflow")]
    Overflow,

    #[error("Invalid authority")]
    InvalidAuthority,

    #[error("Account has non-zero balance")]
    NonZeroBalance,

    #[error("Decimals mismatch")]
    DecimalsMismatch,

    #[error("Invalid multisig configuration")]
    InvalidMultisigConfig,

    #[error("Invalid stake state")]
    InvalidStakeState,

    #[error("Invalid vote state")]
    InvalidVoteState,

    #[error("Invalid account data")]
    InvalidAccountData,
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;
