pub mod types;
pub mod error;
pub mod accounts;
pub mod qmdb_state;
pub mod real_qmdb_state;
pub mod programs;
pub mod sysvars;
pub mod executor;
pub mod runtime;
pub mod bpf;
pub mod evm;
pub mod hybrid;
pub mod sealevel;
pub mod hiperf;
pub mod gpu;

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uchar, c_ulong};
use std::ptr;
use std::slice;
use std::sync::Arc;

pub use accounts::*;
pub use error::*;
pub use executor::*;
pub use programs::*;
pub use runtime::*;
pub use sysvars::*;
pub use types::*;
pub use sealevel::{
    ParallelExecutor, ParallelExecutorConfig, BatchExecutionResult,
    AccessSet, BatchScheduler, ConflictGraph, AccountLocks,
    QMDBParallelExecutor, QMDBExecutorConfig, BlockExecutionResult, QMDBExecutorStats,
};
pub use qmdb_state::{
    InMemoryQMDBState, QMDBState, QMDBStateConfig, StateChangeSet, BlockStateBatch,
};
pub use real_qmdb_state::{RealQmdbState, QmdbParallelExecutorV2};
pub use hiperf::{
    TurboExecutor, TurboConfig, TurboBlockResult, TurboTiming, TurboStats,
    BatchVerifier, VerificationResult,
    ArenaPool, ExecutionArena,
    ExecutionPipeline, PipelineConfig, PipelineStats,
};

// ============================================================================
// FFI INTERFACE FOR GO
// ============================================================================

/// Opaque handle to the runtime
pub struct RuntimeHandle {
    runtime: SvmRuntime<InMemoryAccountsDB>,
}

/// FFI-safe execution result
#[repr(C)]
pub struct ExecutionResultFFI {
    pub success: c_int,
    pub compute_units_used: c_ulong,
    pub fee: c_ulong,
    pub error_msg: *mut c_char,
    pub logs: *mut *mut c_char,
    pub logs_len: usize,
    pub state_changes: *mut StateChangeFFI,
    pub state_changes_len: usize,
}

#[repr(C)]
pub struct StateChangeFFI {
    pub pubkey: [c_uchar; 32],
    pub account_data: *mut c_uchar,
    pub account_data_len: usize,
    pub lamports: c_ulong,
    pub owner: [c_uchar; 32],
    pub executable: c_int,
    pub rent_epoch: c_ulong,
}

/// Create a new runtime instance
#[no_mangle]
pub extern "C" fn svm_runtime_new() -> *mut RuntimeHandle {
    let config = RuntimeConfig::default();
    let runtime = SvmRuntime::new_in_memory(config);

    let handle = Box::new(RuntimeHandle { runtime });
    Box::into_raw(handle)
}

/// Destroy a runtime instance
#[no_mangle]
pub extern "C" fn svm_runtime_free(handle: *mut RuntimeHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Set block context
#[no_mangle]
pub extern "C" fn svm_set_block_context(
    handle: *mut RuntimeHandle,
    slot: c_ulong,
    timestamp: i64,
    blockhash: *const c_uchar,
) -> c_int {
    if handle.is_null() || blockhash.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };
    let blockhash_slice = unsafe { slice::from_raw_parts(blockhash, 32) };

    let mut hash = [0u8; 32];
    hash.copy_from_slice(blockhash_slice);

    match runtime.set_block_context(slot as u64, timestamp, hash) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Execute a transaction
#[no_mangle]
pub extern "C" fn svm_execute_transaction(
    handle: *mut RuntimeHandle,
    tx_data: *const c_uchar,
    tx_len: usize,
    result_out: *mut *mut ExecutionResultFFI,
) -> c_int {
    if handle.is_null() || tx_data.is_null() || result_out.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };
    let tx_bytes = unsafe { slice::from_raw_parts(tx_data, tx_len) };

    // Deserialize transaction
    let tx: Transaction = match borsh::from_slice(tx_bytes) {
        Ok(tx) => tx,
        Err(e) => {
            let result = create_error_result(&format!("Failed to deserialize tx: {}", e));
            unsafe { *result_out = result };
            return 0;
        }
    };

    // Execute
    let result = match runtime.execute_transaction(&tx) {
        Ok(r) => r,
        Err(e) => {
            let result = create_error_result(&format!("Execution error: {:?}", e));
            unsafe { *result_out = result };
            return 0;
        }
    };

    // Convert to FFI result
    let ffi_result = execution_result_to_ffi(&result);
    unsafe { *result_out = ffi_result };

    0
}

fn create_error_result(msg: &str) -> *mut ExecutionResultFFI {
    let error_msg = CString::new(msg).unwrap().into_raw();

    let result = Box::new(ExecutionResultFFI {
        success: 0,
        compute_units_used: 0,
        fee: 0,
        error_msg,
        logs: ptr::null_mut(),
        logs_len: 0,
        state_changes: ptr::null_mut(),
        state_changes_len: 0,
    });

    Box::into_raw(result)
}

fn execution_result_to_ffi(result: &ExecutionResult) -> *mut ExecutionResultFFI {
    // Convert logs
    let logs: Vec<*mut c_char> = result
        .logs
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap().into_raw())
        .collect();

    let (logs_ptr, logs_len) = if logs.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        let len = logs.len();
        let mut logs_vec = logs.into_boxed_slice();
        let ptr = logs_vec.as_mut_ptr();
        std::mem::forget(logs_vec);
        (ptr, len)
    };

    // Convert state changes
    let state_changes: Vec<StateChangeFFI> = result
        .state_changes
        .iter()
        .map(|(pubkey, account)| {
            let mut data = account.data.clone().into_boxed_slice();
            let data_ptr = data.as_mut_ptr();
            let data_len = data.len();
            std::mem::forget(data);

            StateChangeFFI {
                pubkey: pubkey.to_bytes(),
                account_data: data_ptr,
                account_data_len: data_len,
                lamports: account.lamports as c_ulong,
                owner: account.owner.to_bytes(),
                executable: if account.executable { 1 } else { 0 },
                rent_epoch: account.rent_epoch as c_ulong,
            }
        })
        .collect();

    let (state_changes_ptr, state_changes_len) = if state_changes.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        let len = state_changes.len();
        let mut changes_vec = state_changes.into_boxed_slice();
        let ptr = changes_vec.as_mut_ptr();
        std::mem::forget(changes_vec);
        (ptr, len)
    };

    // Convert error message
    let error_msg = result
        .error
        .as_ref()
        .map(|e| CString::new(format!("{:?}", e)).unwrap().into_raw())
        .unwrap_or(ptr::null_mut());

    let ffi_result = Box::new(ExecutionResultFFI {
        success: if result.success { 1 } else { 0 },
        compute_units_used: result.compute_units_used as c_ulong,
        fee: result.fee as c_ulong,
        error_msg,
        logs: logs_ptr,
        logs_len,
        state_changes: state_changes_ptr,
        state_changes_len,
    });

    Box::into_raw(ffi_result)
}

/// Free execution result
#[no_mangle]
pub extern "C" fn svm_free_result(result: *mut ExecutionResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result = Box::from_raw(result);

        if !result.error_msg.is_null() {
            drop(CString::from_raw(result.error_msg));
        }

        if !result.logs.is_null() {
            let logs = slice::from_raw_parts_mut(result.logs, result.logs_len);
            for log in logs {
                if !log.is_null() {
                    drop(CString::from_raw(*log));
                }
            }
            let _ = Vec::from_raw_parts(result.logs, result.logs_len, result.logs_len);
        }

        if !result.state_changes.is_null() {
            let changes =
                slice::from_raw_parts_mut(result.state_changes, result.state_changes_len);
            for change in changes {
                if !change.account_data.is_null() {
                    let _ = Vec::from_raw_parts(
                        change.account_data,
                        change.account_data_len,
                        change.account_data_len,
                    );
                }
            }
            let _ = Vec::from_raw_parts(
                result.state_changes,
                result.state_changes_len,
                result.state_changes_len,
            );
        }
    }
}

/// Get current slot
#[no_mangle]
pub extern "C" fn svm_get_slot(handle: *mut RuntimeHandle) -> c_ulong {
    if handle.is_null() {
        return 0;
    }

    let runtime = unsafe { &(*handle).runtime };
    runtime.current_slot().unwrap_or(0) as c_ulong
}

/// Get current blockhash
#[no_mangle]
pub extern "C" fn svm_get_blockhash(handle: *mut RuntimeHandle, blockhash_out: *mut c_uchar) -> c_int {
    if handle.is_null() || blockhash_out.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };

    match runtime.current_blockhash() {
        Ok(hash) => {
            unsafe {
                ptr::copy_nonoverlapping(hash.as_ptr(), blockhash_out, 32);
            }
            0
        }
        Err(_) => -1,
    }
}

// ============================================================================
// HYBRID EXECUTOR FFI (SVM + EVM)
// ============================================================================

use hybrid::HybridExecutor;

/// Opaque handle to hybrid executor
pub struct HybridHandle {
    executor: HybridExecutor,
}

/// Create new hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_new(chain_id: c_ulong) -> *mut HybridHandle {
    let executor = HybridExecutor::new(chain_id as u64);
    Box::into_raw(Box::new(HybridHandle { executor }))
}

/// Free hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_free(handle: *mut HybridHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Set block context for hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_set_block(handle: *mut HybridHandle, slot: c_ulong, timestamp: i64) -> c_int {
    if handle.is_null() {
        return -1;
    }
    let executor = unsafe { &mut (*handle).executor };
    executor.set_block(slot as u64, timestamp as u64);
    0
}

/// Load account into hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_load_account(
    handle: *mut HybridHandle,
    pubkey: *const c_uchar,
    lamports: c_ulong,
    data: *const c_uchar,
    data_len: usize,
    owner: *const c_uchar,
    executable: c_int,
) -> c_int {
    if handle.is_null() || pubkey.is_null() || owner.is_null() {
        return -1;
    }

    let executor = unsafe { &mut (*handle).executor };

    let mut pk = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(pubkey, pk.as_mut_ptr(), 32) };

    let mut own = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(owner, own.as_mut_ptr(), 32) };

    let account_data = if data.is_null() || data_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(data, data_len).to_vec() }
    };

    let account = Account {
        lamports: lamports as u64,
        data: account_data,
        owner: Pubkey(own),
        executable: executable != 0,
        rent_epoch: 0,
    };

    executor.load_account(Pubkey(pk), account);
    0
}

/// Get account from hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_get_account(
    handle: *mut HybridHandle,
    pubkey: *const c_uchar,
    lamports_out: *mut c_ulong,
    data_out: *mut *mut c_uchar,
    data_len_out: *mut usize,
    owner_out: *mut c_uchar,
    executable_out: *mut c_int,
) -> c_int {
    if handle.is_null() || pubkey.is_null() {
        return -1;
    }

    let executor = unsafe { &(*handle).executor };

    let mut pk = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(pubkey, pk.as_mut_ptr(), 32) };

    match executor.get_account(&Pubkey(pk)) {
        Some(account) => {
            unsafe {
                if !lamports_out.is_null() {
                    *lamports_out = account.lamports as c_ulong;
                }
                if !owner_out.is_null() {
                    ptr::copy_nonoverlapping(account.owner.0.as_ptr(), owner_out, 32);
                }
                if !executable_out.is_null() {
                    *executable_out = if account.executable { 1 } else { 0 };
                }
                if !data_out.is_null() && !data_len_out.is_null() {
                    let mut data = account.data.clone().into_boxed_slice();
                    *data_len_out = data.len();
                    *data_out = data.as_mut_ptr();
                    std::mem::forget(data);
                }
            }
            0
        }
        None => -1,
    }
}

/// Execute EVM transaction via hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_execute_evm(
    handle: *mut HybridHandle,
    caller: *const c_uchar,      // 20 bytes EVM address
    to: *const c_uchar,          // 20 bytes EVM address (null for deploy)
    value_lo: c_ulong,           // Value lower 64 bits
    value_hi: c_ulong,           // Value upper 64 bits
    data: *const c_uchar,
    data_len: usize,
    gas_limit: c_ulong,
    result_out: *mut *mut ExecutionResultFFI,
) -> c_int {
    if handle.is_null() || caller.is_null() || result_out.is_null() {
        return -1;
    }

    let executor = unsafe { &mut (*handle).executor };

    let mut caller_addr = [0u8; 20];
    unsafe { ptr::copy_nonoverlapping(caller, caller_addr.as_mut_ptr(), 20) };

    let to_addr = if to.is_null() {
        None
    } else {
        let mut addr = [0u8; 20];
        unsafe { ptr::copy_nonoverlapping(to, addr.as_mut_ptr(), 20) };
        Some(addr)
    };

    let value = (value_hi as u128) << 64 | (value_lo as u128);

    let calldata = if data.is_null() || data_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(data, data_len).to_vec() }
    };

    match executor.execute_evm_transaction(caller_addr, to_addr, value, calldata, gas_limit as u64) {
        Ok(result) => {
            let ffi_result = Box::new(ExecutionResultFFI {
                success: if result.success { 1 } else { 0 },
                compute_units_used: result.compute_units_used as c_ulong,
                fee: 0,
                error_msg: ptr::null_mut(),
                logs: ptr::null_mut(),
                logs_len: 0,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            });
            unsafe { *result_out = Box::into_raw(ffi_result) };
            0
        }
        Err(e) => {
            let error_msg = CString::new(format!("{:?}", e)).unwrap().into_raw();
            let ffi_result = Box::new(ExecutionResultFFI {
                success: 0,
                compute_units_used: 0,
                fee: 0,
                error_msg,
                logs: ptr::null_mut(),
                logs_len: 0,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            });
            unsafe { *result_out = Box::into_raw(ffi_result) };
            -1
        }
    }
}

/// Execute BPF program via hybrid executor
#[no_mangle]
pub extern "C" fn hybrid_execute_bpf(
    handle: *mut HybridHandle,
    program_id: *const c_uchar,
    bytecode: *const c_uchar,
    bytecode_len: usize,
    input_data: *const c_uchar,
    input_len: usize,
    result_out: *mut *mut ExecutionResultFFI,
) -> c_int {
    if handle.is_null() || program_id.is_null() || bytecode.is_null() || result_out.is_null() {
        return -1;
    }

    let mut prog_id = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(program_id, prog_id.as_mut_ptr(), 32) };

    let code = unsafe { slice::from_raw_parts(bytecode, bytecode_len) };
    let input = if input_data.is_null() || input_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(input_data, input_len).to_vec() }
    };

    // Create BPF VM and execute
    match bpf::BpfVm::new(code, input) {
        Ok(mut vm) => {
            let mut accounts = vec![];
            match vm.execute(&mut accounts, Pubkey(prog_id)) {
                Ok(return_value) => {
                    let logs: Vec<*mut c_char> = vm.logs()
                        .iter()
                        .map(|s| CString::new(s.as_str()).unwrap().into_raw())
                        .collect();

                    let (logs_ptr, logs_len) = if logs.is_empty() {
                        (ptr::null_mut(), 0)
                    } else {
                        let len = logs.len();
                        let mut logs_vec = logs.into_boxed_slice();
                        let ptr = logs_vec.as_mut_ptr();
                        std::mem::forget(logs_vec);
                        (ptr, len)
                    };

                    let ffi_result = Box::new(ExecutionResultFFI {
                        success: 1,
                        compute_units_used: (bpf::DEFAULT_COMPUTE_UNITS - vm.compute_units()) as c_ulong,
                        fee: 0,
                        error_msg: ptr::null_mut(),
                        logs: logs_ptr,
                        logs_len,
                        state_changes: ptr::null_mut(),
                        state_changes_len: 0,
                    });
                    unsafe { *result_out = Box::into_raw(ffi_result) };
                    return_value as c_int
                }
                Err(e) => {
                    let error_msg = CString::new(format!("{:?}", e)).unwrap().into_raw();
                    let ffi_result = Box::new(ExecutionResultFFI {
                        success: 0,
                        compute_units_used: 0,
                        fee: 0,
                        error_msg,
                        logs: ptr::null_mut(),
                        logs_len: 0,
                        state_changes: ptr::null_mut(),
                        state_changes_len: 0,
                    });
                    unsafe { *result_out = Box::into_raw(ffi_result) };
                    -1
                }
            }
        }
        Err(e) => {
            let error_msg = CString::new(format!("{:?}", e)).unwrap().into_raw();
            let ffi_result = Box::new(ExecutionResultFFI {
                success: 0,
                compute_units_used: 0,
                fee: 0,
                error_msg,
                logs: ptr::null_mut(),
                logs_len: 0,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            });
            unsafe { *result_out = Box::into_raw(ffi_result) };
            -1
        }
    }
}

/// Cross-VM call: SVM program calling EVM contract
#[no_mangle]
pub extern "C" fn hybrid_svm_call_evm(
    handle: *mut HybridHandle,
    caller_pubkey: *const c_uchar,
    evm_contract: *const c_uchar,
    calldata: *const c_uchar,
    calldata_len: usize,
    value_lo: c_ulong,
    value_hi: c_ulong,
    result_out: *mut *mut c_uchar,
    result_len_out: *mut usize,
) -> c_int {
    if handle.is_null() || caller_pubkey.is_null() || evm_contract.is_null() || result_out.is_null() {
        return -1;
    }

    let executor = unsafe { &mut (*handle).executor };

    let mut caller = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(caller_pubkey, caller.as_mut_ptr(), 32) };

    let mut contract = [0u8; 20];
    unsafe { ptr::copy_nonoverlapping(evm_contract, contract.as_mut_ptr(), 20) };

    let data = if calldata.is_null() || calldata_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(calldata, calldata_len).to_vec() }
    };

    let value = (value_hi as u128) << 64 | (value_lo as u128);

    match executor.svm_call_evm(&Pubkey(caller), &contract, &data, value) {
        Ok(result) => {
            let mut result_data = result.into_boxed_slice();
            unsafe {
                *result_len_out = result_data.len();
                *result_out = result_data.as_mut_ptr();
            }
            std::mem::forget(result_data);
            0
        }
        Err(_) => -1,
    }
}

// ============================================================================
// SEALEVEL PARALLEL EXECUTOR FFI
// ============================================================================

use sealevel::{ParallelExecutor as SealevelExecutor, ParallelExecutorConfig as SealevelConfig};
use std::sync::Mutex;

/// Opaque handle to the parallel executor
pub struct ParallelExecutorHandle {
    executor: SealevelExecutor<InMemoryAccountsDB>,
    // Current execution context
    ctx: Mutex<ExecutionContext>,
}

/// Create a new parallel executor
#[no_mangle]
pub extern "C" fn sealevel_new(num_threads: c_ulong) -> *mut ParallelExecutorHandle {
    let accounts = AccountsManager::new(InMemoryAccountsDB::new());

    let mut config = SealevelConfig::default();
    if num_threads > 0 {
        config.num_threads = num_threads as usize;
    }

    let executor = SealevelExecutor::new(accounts, config.clone());
    let ctx = ExecutionContext::new(0, 0, config.executor_config);

    Box::into_raw(Box::new(ParallelExecutorHandle {
        executor,
        ctx: Mutex::new(ctx),
    }))
}

/// Free parallel executor
#[no_mangle]
pub extern "C" fn sealevel_free(handle: *mut ParallelExecutorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Set block context for parallel executor
#[no_mangle]
pub extern "C" fn sealevel_set_block(
    handle: *mut ParallelExecutorHandle,
    slot: c_ulong,
    timestamp: i64,
    blockhash: *const c_uchar,
) -> c_int {
    if handle.is_null() || blockhash.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let mut ctx = match handle.ctx.lock() {
        Ok(ctx) => ctx,
        Err(_) => return -1,
    };

    ctx.slot = slot as u64;
    ctx.timestamp = timestamp;

    let mut hash = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(blockhash, hash.as_mut_ptr(), 32) };
    ctx.add_blockhash(hash);

    0
}

/// FFI result for batch execution
#[repr(C)]
pub struct BatchResultFFI {
    pub successful: c_ulong,
    pub failed: c_ulong,
    pub total_compute_units: c_ulong,
    pub total_fees: c_ulong,
    pub execution_time_us: c_ulong,
    pub results: *mut ExecutionResultFFI,
    pub results_len: usize,
}

/// Execute transactions in parallel
///
/// Takes an array of serialized transactions and returns batch results.
#[no_mangle]
pub extern "C" fn sealevel_execute_batch(
    handle: *mut ParallelExecutorHandle,
    tx_data: *const *const c_uchar,
    tx_lens: *const usize,
    tx_count: usize,
    result_out: *mut *mut BatchResultFFI,
) -> c_int {
    if handle.is_null() || tx_data.is_null() || tx_lens.is_null() || result_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let ctx = match handle.ctx.lock() {
        Ok(ctx) => ctx,
        Err(_) => return -1,
    };

    // Deserialize transactions
    let mut transactions = Vec::with_capacity(tx_count);
    for i in 0..tx_count {
        let tx_ptr = unsafe { *tx_data.add(i) };
        let tx_len = unsafe { *tx_lens.add(i) };

        if tx_ptr.is_null() {
            continue;
        }

        let tx_bytes = unsafe { slice::from_raw_parts(tx_ptr, tx_len) };
        match borsh::from_slice::<Transaction>(tx_bytes) {
            Ok(tx) => transactions.push(tx),
            Err(_) => continue,
        }
    }

    // Execute in parallel
    let result = handle.executor.execute_transactions(&transactions, &ctx);

    // Convert to FFI
    let results: Vec<ExecutionResultFFI> = result.results
        .iter()
        .map(|r| {
            let error_msg = r.error
                .as_ref()
                .map(|e| CString::new(format!("{:?}", e)).unwrap().into_raw())
                .unwrap_or(ptr::null_mut());

            let logs: Vec<*mut c_char> = r.logs
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap().into_raw())
                .collect();

            let (logs_ptr, logs_len) = if logs.is_empty() {
                (ptr::null_mut(), 0)
            } else {
                let len = logs.len();
                let mut logs_vec = logs.into_boxed_slice();
                let ptr = logs_vec.as_mut_ptr();
                std::mem::forget(logs_vec);
                (ptr, len)
            };

            ExecutionResultFFI {
                success: if r.success { 1 } else { 0 },
                compute_units_used: r.compute_units_used as c_ulong,
                fee: r.fee as c_ulong,
                error_msg,
                logs: logs_ptr,
                logs_len,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            }
        })
        .collect();

    let (results_ptr, results_len) = if results.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        let len = results.len();
        let mut results_vec = results.into_boxed_slice();
        let ptr = results_vec.as_mut_ptr();
        std::mem::forget(results_vec);
        (ptr, len)
    };

    let batch_result = Box::new(BatchResultFFI {
        successful: result.successful as c_ulong,
        failed: result.failed as c_ulong,
        total_compute_units: result.total_compute_units as c_ulong,
        total_fees: result.total_fees as c_ulong,
        execution_time_us: result.execution_time_us as c_ulong,
        results: results_ptr,
        results_len,
    });

    unsafe { *result_out = Box::into_raw(batch_result) };

    0
}

/// Free batch result
#[no_mangle]
pub extern "C" fn sealevel_free_batch_result(result: *mut BatchResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result = Box::from_raw(result);

        if !result.results.is_null() {
            let results = slice::from_raw_parts_mut(result.results, result.results_len);
            for r in results {
                if !r.error_msg.is_null() {
                    drop(CString::from_raw(r.error_msg));
                }
                if !r.logs.is_null() {
                    let logs = slice::from_raw_parts_mut(r.logs, r.logs_len);
                    for log in logs {
                        if !log.is_null() {
                            drop(CString::from_raw(*log));
                        }
                    }
                    let _ = Vec::from_raw_parts(r.logs, r.logs_len, r.logs_len);
                }
            }
            let _ = Vec::from_raw_parts(result.results, result.results_len, result.results_len);
        }
    }
}

/// Get executor statistics
#[no_mangle]
pub extern "C" fn sealevel_get_stats(
    handle: *mut ParallelExecutorHandle,
    total_txs: *mut c_ulong,
    total_batches: *mut c_ulong,
    num_threads: *mut c_ulong,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let stats = handle.executor.get_stats();

    unsafe {
        if !total_txs.is_null() {
            *total_txs = stats.total_txs_processed as c_ulong;
        }
        if !total_batches.is_null() {
            *total_batches = stats.total_batches_processed as c_ulong;
        }
        if !num_threads.is_null() {
            *num_threads = stats.num_threads as c_ulong;
        }
    }

    0
}

// ============================================================================
// QMDB PARALLEL EXECUTOR FFI (Block-level execution with Merkle roots)
// ============================================================================

// QMDBParallelExecutor, QMDBExecutorConfig, and InMemoryQMDBState are already imported above

/// Opaque handle to the QMDB executor
pub struct QMDBExecutorHandle {
    executor: QMDBParallelExecutor,
    ctx: Mutex<ExecutionContext>,
}

/// FFI result for block execution
#[repr(C)]
pub struct BlockResultFFI {
    pub successful: c_ulong,
    pub failed: c_ulong,
    pub total_compute_units: c_ulong,
    pub total_fees: c_ulong,
    pub execution_time_us: c_ulong,
    pub merkle_root: [c_uchar; 32],
    pub block_height: c_ulong,
    pub results: *mut ExecutionResultFFI,
    pub results_len: usize,
}

/// Create a new QMDB parallel executor
#[no_mangle]
pub extern "C" fn qmdb_executor_new(num_threads: c_ulong) -> *mut QMDBExecutorHandle {
    let state = Arc::new(InMemoryQMDBState::new());

    let mut config = QMDBExecutorConfig::default();
    if num_threads > 0 {
        config.num_threads = num_threads as usize;
    }

    let executor = QMDBParallelExecutor::new(state, config.clone());
    let ctx = ExecutionContext::new(0, 0, config.executor_config);

    Box::into_raw(Box::new(QMDBExecutorHandle {
        executor,
        ctx: Mutex::new(ctx),
    }))
}

/// Free QMDB executor
#[no_mangle]
pub extern "C" fn qmdb_executor_free(handle: *mut QMDBExecutorHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Set block context for QMDB executor
#[no_mangle]
pub extern "C" fn qmdb_set_block_context(
    handle: *mut QMDBExecutorHandle,
    slot: c_ulong,
    timestamp: i64,
    blockhash: *const c_uchar,
) -> c_int {
    if handle.is_null() || blockhash.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let mut ctx = match handle.ctx.lock() {
        Ok(ctx) => ctx,
        Err(_) => return -1,
    };

    ctx.slot = slot as u64;
    ctx.timestamp = timestamp;

    let mut hash = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(blockhash, hash.as_mut_ptr(), 32) };
    ctx.add_blockhash(hash);

    0
}

/// Execute a full block of transactions with atomic commit
/// Returns the merkle root of the state after execution
#[no_mangle]
pub extern "C" fn qmdb_execute_block(
    handle: *mut QMDBExecutorHandle,
    block_height: c_ulong,
    tx_data: *const *const c_uchar,
    tx_lens: *const usize,
    tx_count: usize,
    result_out: *mut *mut BlockResultFFI,
) -> c_int {
    if handle.is_null() || tx_data.is_null() || tx_lens.is_null() || result_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let ctx = match handle.ctx.lock() {
        Ok(ctx) => ctx,
        Err(_) => return -1,
    };

    // Deserialize transactions
    let mut transactions = Vec::with_capacity(tx_count);
    for i in 0..tx_count {
        let tx_ptr = unsafe { *tx_data.add(i) };
        let tx_len = unsafe { *tx_lens.add(i) };

        if tx_ptr.is_null() {
            continue;
        }

        let tx_bytes = unsafe { slice::from_raw_parts(tx_ptr, tx_len) };
        match borsh::from_slice::<Transaction>(tx_bytes) {
            Ok(tx) => transactions.push(tx),
            Err(_) => continue,
        }
    }

    // Execute block with QMDB (atomic commit)
    let result = match handle.executor.execute_block(block_height as u64, &transactions, &ctx) {
        Ok(r) => r,
        Err(e) => {
            // Create error result
            let error_msg = CString::new(format!("{:?}", e)).unwrap().into_raw();
            let error_result = Box::new(ExecutionResultFFI {
                success: 0,
                compute_units_used: 0,
                fee: 0,
                error_msg,
                logs: ptr::null_mut(),
                logs_len: 0,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            });

            let block_result = Box::new(BlockResultFFI {
                successful: 0,
                failed: tx_count as c_ulong,
                total_compute_units: 0,
                total_fees: 0,
                execution_time_us: 0,
                merkle_root: [0u8; 32],
                block_height: block_height as c_ulong,
                results: Box::into_raw(error_result),
                results_len: 1,
            });

            unsafe { *result_out = Box::into_raw(block_result) };
            return -1;
        }
    };

    // Convert individual results to FFI
    let results: Vec<ExecutionResultFFI> = result.results
        .iter()
        .map(|r| {
            let error_msg = r.error
                .as_ref()
                .map(|e| CString::new(format!("{:?}", e)).unwrap().into_raw())
                .unwrap_or(ptr::null_mut());

            let logs: Vec<*mut c_char> = r.logs
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap().into_raw())
                .collect();

            let (logs_ptr, logs_len) = if logs.is_empty() {
                (ptr::null_mut(), 0)
            } else {
                let len = logs.len();
                let mut logs_vec = logs.into_boxed_slice();
                let ptr = logs_vec.as_mut_ptr();
                std::mem::forget(logs_vec);
                (ptr, len)
            };

            ExecutionResultFFI {
                success: if r.success { 1 } else { 0 },
                compute_units_used: r.compute_units_used as c_ulong,
                fee: r.fee as c_ulong,
                error_msg,
                logs: logs_ptr,
                logs_len,
                state_changes: ptr::null_mut(),
                state_changes_len: 0,
            }
        })
        .collect();

    let (results_ptr, results_len) = if results.is_empty() {
        (ptr::null_mut(), 0)
    } else {
        let len = results.len();
        let mut results_vec = results.into_boxed_slice();
        let ptr = results_vec.as_mut_ptr();
        std::mem::forget(results_vec);
        (ptr, len)
    };

    let block_result = Box::new(BlockResultFFI {
        successful: result.successful as c_ulong,
        failed: result.failed as c_ulong,
        total_compute_units: result.total_compute_units as c_ulong,
        total_fees: result.total_fees as c_ulong,
        execution_time_us: result.execution_time_us as c_ulong,
        merkle_root: result.merkle_root,
        block_height: result.block_height as c_ulong,
        results: results_ptr,
        results_len,
    });

    unsafe { *result_out = Box::into_raw(block_result) };

    0
}

/// Free block result
#[no_mangle]
pub extern "C" fn qmdb_free_block_result(result: *mut BlockResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result = Box::from_raw(result);

        if !result.results.is_null() {
            let results = slice::from_raw_parts_mut(result.results, result.results_len);
            for r in results {
                if !r.error_msg.is_null() {
                    drop(CString::from_raw(r.error_msg));
                }
                if !r.logs.is_null() {
                    let logs = slice::from_raw_parts_mut(r.logs, r.logs_len);
                    for log in logs {
                        if !log.is_null() {
                            drop(CString::from_raw(*log));
                        }
                    }
                    let _ = Vec::from_raw_parts(r.logs, r.logs_len, r.logs_len);
                }
            }
            let _ = Vec::from_raw_parts(result.results, result.results_len, result.results_len);
        }
    }
}

/// Get current merkle root from QMDB state
#[no_mangle]
pub extern "C" fn qmdb_get_merkle_root(
    handle: *mut QMDBExecutorHandle,
    root_out: *mut c_uchar,
) -> c_int {
    if handle.is_null() || root_out.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };

    match handle.executor.get_merkle_root() {
        Ok(root) => {
            unsafe { ptr::copy_nonoverlapping(root.as_ptr(), root_out, 32) };
            0
        }
        Err(_) => -1,
    }
}

/// Get QMDB executor statistics
#[no_mangle]
pub extern "C" fn qmdb_get_stats(
    handle: *mut QMDBExecutorHandle,
    total_txs: *mut c_ulong,
    total_blocks: *mut c_ulong,
    num_threads: *mut c_ulong,
) -> c_int {
    if handle.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };
    let stats = handle.executor.get_stats();

    unsafe {
        if !total_txs.is_null() {
            *total_txs = stats.total_txs_processed as c_ulong;
        }
        if !total_blocks.is_null() {
            *total_blocks = stats.total_blocks_processed as c_ulong;
        }
        if !num_threads.is_null() {
            *num_threads = stats.num_threads as c_ulong;
        }
    }

    0
}

/// Load account into QMDB state (for testing/initialization)
#[no_mangle]
pub extern "C" fn qmdb_load_account(
    handle: *mut QMDBExecutorHandle,
    pubkey: *const c_uchar,
    lamports: c_ulong,
    data: *const c_uchar,
    data_len: usize,
    owner: *const c_uchar,
    executable: c_int,
) -> c_int {
    if handle.is_null() || pubkey.is_null() || owner.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };

    let mut pk = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(pubkey, pk.as_mut_ptr(), 32) };

    let mut own = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(owner, own.as_mut_ptr(), 32) };

    let account_data = if data.is_null() || data_len == 0 {
        vec![]
    } else {
        unsafe { slice::from_raw_parts(data, data_len).to_vec() }
    };

    let account = Account {
        lamports: lamports as u64,
        data: account_data,
        owner: Pubkey(own),
        executable: executable != 0,
        rent_epoch: 0,
    };

    handle.executor.load_account(Pubkey(pk), account);
    0
}

/// Get account from QMDB state
#[no_mangle]
pub extern "C" fn qmdb_get_account(
    handle: *mut QMDBExecutorHandle,
    pubkey: *const c_uchar,
    lamports_out: *mut c_ulong,
    data_out: *mut *mut c_uchar,
    data_len_out: *mut usize,
    owner_out: *mut c_uchar,
    executable_out: *mut c_int,
) -> c_int {
    if handle.is_null() || pubkey.is_null() {
        return -1;
    }

    let handle = unsafe { &*handle };

    let mut pk = [0u8; 32];
    unsafe { ptr::copy_nonoverlapping(pubkey, pk.as_mut_ptr(), 32) };

    match handle.executor.get_account(&Pubkey(pk)) {
        Some(account) => {
            unsafe {
                if !lamports_out.is_null() {
                    *lamports_out = account.lamports as c_ulong;
                }
                if !owner_out.is_null() {
                    ptr::copy_nonoverlapping(account.owner.0.as_ptr(), owner_out, 32);
                }
                if !executable_out.is_null() {
                    *executable_out = if account.executable { 1 } else { 0 };
                }
                if !data_out.is_null() && !data_len_out.is_null() {
                    let mut data = account.data.clone().into_boxed_slice();
                    *data_len_out = data.len();
                    *data_out = data.as_mut_ptr();
                    std::mem::forget(data);
                }
            }
            0
        }
        None => -1,
    }
}
