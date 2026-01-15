use crate::{RuntimeError, RuntimeResult, Account, Pubkey};
use super::memory::BpfMemory;
use std::collections::HashMap;
use sha2::{Sha256, Digest};

/// Syscall function signature
pub type SyscallFn = fn(&mut SyscallContext, u64, u64, u64, u64, u64) -> RuntimeResult<u64>;

/// Syscall context passed to syscall handlers
pub struct SyscallContext<'a> {
    pub memory: &'a mut BpfMemory,
    pub accounts: &'a mut [Account],
    pub program_id: Pubkey,
    pub compute_units: &'a mut u64,
    pub return_data: &'a mut Option<(Pubkey, Vec<u8>)>,
    pub logs: &'a mut Vec<String>,
}

/// Syscall registry for BPF programs
pub struct SyscallRegistry {
    syscalls: HashMap<u32, (&'static str, SyscallFn)>,
}

impl SyscallRegistry {
    pub fn new() -> Self {
        let mut registry = SyscallRegistry {
            syscalls: HashMap::new(),
        };
        registry.register_defaults();
        registry
    }

    fn register_defaults(&mut self) {
        // Logging syscalls
        self.register(0x0, "abort", syscall_abort);
        self.register(0x1, "sol_panic_", syscall_panic);
        self.register(0x2, "sol_log_", syscall_log);
        self.register(0x3, "sol_log_64_", syscall_log_u64);
        self.register(0x4, "sol_log_compute_units_", syscall_log_compute_units);
        self.register(0x5, "sol_log_pubkey", syscall_log_pubkey);

        // Memory syscalls
        self.register(0x6, "sol_memcpy_", syscall_memcpy);
        self.register(0x7, "sol_memset_", syscall_memset);
        self.register(0x8, "sol_memmove_", syscall_memcpy); // Same as memcpy for now
        self.register(0x9, "sol_memcmp_", syscall_memcmp);

        // Crypto syscalls
        self.register(0xa, "sol_sha256", syscall_sha256);
        self.register(0xb, "sol_keccak256", syscall_keccak256);

        // Program syscalls
        self.register(0xc, "sol_invoke_signed_c", syscall_invoke_signed);
        self.register(0xd, "sol_alloc_free_", syscall_alloc_free);
        self.register(0xe, "sol_set_return_data", syscall_set_return_data);
        self.register(0xf, "sol_get_return_data", syscall_get_return_data);

        // Account info syscalls
        self.register(0x10, "sol_get_clock_sysvar", syscall_get_clock_sysvar);
        self.register(0x11, "sol_get_rent_sysvar", syscall_get_rent_sysvar);

        // Cross-VM bridge syscalls (custom for hybrid SVM+EVM)
        self.register(0x100, "sol_call_evm", syscall_call_evm);
        self.register(0x101, "sol_evm_balance", syscall_evm_balance);
    }

    pub fn register(&mut self, id: u32, name: &'static str, func: SyscallFn) {
        self.syscalls.insert(id, (name, func));
    }

    pub fn invoke(
        &self,
        ctx: &mut SyscallContext,
        syscall_id: u32,
        r1: u64, r2: u64, r3: u64, r4: u64, r5: u64,
    ) -> RuntimeResult<u64> {
        match self.syscalls.get(&syscall_id) {
            Some((name, func)) => {
                // Deduct compute units for syscall
                let cost = self.syscall_cost(syscall_id);
                if *ctx.compute_units < cost {
                    return Err(RuntimeError::ComputeExhausted);
                }
                *ctx.compute_units -= cost;

                log::trace!("syscall: {} (0x{:x})", name, syscall_id);
                func(ctx, r1, r2, r3, r4, r5)
            }
            None => Err(RuntimeError::InvalidSyscall(syscall_id)),
        }
    }

    fn syscall_cost(&self, syscall_id: u32) -> u64 {
        match syscall_id {
            0x0 | 0x1 => 0, // abort/panic - no cost
            0x2..=0x5 => 100, // logging
            0x6..=0x9 => 10,  // memory operations (per call, not per byte)
            0xa | 0xb => 1000, // crypto hashing
            0xc => 5000, // CPI invoke
            0xd => 50,   // alloc
            0xe | 0xf => 20, // return data
            0x10 | 0x11 => 50, // sysvar reads
            0x100 | 0x101 => 10000, // EVM bridge calls
            _ => 100,
        }
    }
}

impl Default for SyscallRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Syscall implementations
// ============================================================================

fn syscall_abort(_ctx: &mut SyscallContext, _: u64, _: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    Err(RuntimeError::ProgramAborted)
}

fn syscall_panic(ctx: &mut SyscallContext, file_ptr: u64, len: u64, line: u64, column: u64, _: u64) -> RuntimeResult<u64> {
    let file = if file_ptr != 0 && len > 0 {
        let bytes = ctx.memory.read_slice(file_ptr, len as usize)?;
        String::from_utf8_lossy(&bytes).to_string()
    } else {
        "unknown".to_string()
    };
    Err(RuntimeError::ProgramPanic(format!("{}:{}:{}", file, line, column)))
}

fn syscall_log(ctx: &mut SyscallContext, msg_ptr: u64, msg_len: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    if msg_len > 10000 {
        return Err(RuntimeError::InvalidArgument("log message too long".into()));
    }
    let bytes = ctx.memory.read_slice(msg_ptr, msg_len as usize)?;
    let msg = String::from_utf8_lossy(&bytes).to_string();
    log::info!("Program log: {}", msg);
    ctx.logs.push(msg);
    Ok(0)
}

fn syscall_log_u64(ctx: &mut SyscallContext, v1: u64, v2: u64, v3: u64, v4: u64, v5: u64) -> RuntimeResult<u64> {
    let msg = format!("{} {} {} {} {}", v1, v2, v3, v4, v5);
    log::info!("Program log: {}", msg);
    ctx.logs.push(msg);
    Ok(0)
}

fn syscall_log_compute_units(ctx: &mut SyscallContext, _: u64, _: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    let msg = format!("Compute units remaining: {}", ctx.compute_units);
    log::info!("Program log: {}", msg);
    ctx.logs.push(msg);
    Ok(0)
}

fn syscall_log_pubkey(ctx: &mut SyscallContext, pubkey_ptr: u64, _: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    let bytes = ctx.memory.read_slice(pubkey_ptr, 32)?;
    let pubkey = bs58::encode(&bytes).into_string();
    log::info!("Program log: {}", pubkey);
    ctx.logs.push(pubkey);
    Ok(0)
}

fn syscall_memcpy(ctx: &mut SyscallContext, dst: u64, src: u64, len: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    if len > 1_000_000 {
        return Err(RuntimeError::InvalidArgument("memcpy too large".into()));
    }
    let data = ctx.memory.read_slice(src, len as usize)?;
    ctx.memory.write_slice(dst, &data)?;
    Ok(0)
}

fn syscall_memset(ctx: &mut SyscallContext, dst: u64, val: u64, len: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    if len > 1_000_000 {
        return Err(RuntimeError::InvalidArgument("memset too large".into()));
    }
    let byte = val as u8;
    for i in 0..len {
        ctx.memory.write_u8(dst + i, byte)?;
    }
    Ok(0)
}

fn syscall_memcmp(ctx: &mut SyscallContext, s1: u64, s2: u64, len: u64, result_ptr: u64, _: u64) -> RuntimeResult<u64> {
    if len > 1_000_000 {
        return Err(RuntimeError::InvalidArgument("memcmp too large".into()));
    }
    let d1 = ctx.memory.read_slice(s1, len as usize)?;
    let d2 = ctx.memory.read_slice(s2, len as usize)?;

    let result: i32 = match d1.cmp(&d2) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    };

    ctx.memory.write_u32(result_ptr, result as u32)?;
    Ok(0)
}

fn syscall_sha256(ctx: &mut SyscallContext, vals_ptr: u64, vals_len: u64, result_ptr: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    let mut hasher = Sha256::new();

    // Read array of (ptr, len) pairs
    for i in 0..vals_len {
        let entry_ptr = vals_ptr + i * 16;
        let data_ptr = ctx.memory.read_u64(entry_ptr)?;
        let data_len = ctx.memory.read_u64(entry_ptr + 8)?;

        if data_len > 1_000_000 {
            return Err(RuntimeError::InvalidArgument("sha256 input too large".into()));
        }

        let data = ctx.memory.read_slice(data_ptr, data_len as usize)?;
        hasher.update(&data);
    }

    let hash = hasher.finalize();
    ctx.memory.write_slice(result_ptr, &hash)?;
    Ok(0)
}

fn syscall_keccak256(ctx: &mut SyscallContext, vals_ptr: u64, vals_len: u64, result_ptr: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    use sha3::{Keccak256, Digest as Sha3Digest};
    let mut hasher = Keccak256::new();

    for i in 0..vals_len {
        let entry_ptr = vals_ptr + i * 16;
        let data_ptr = ctx.memory.read_u64(entry_ptr)?;
        let data_len = ctx.memory.read_u64(entry_ptr + 8)?;

        if data_len > 1_000_000 {
            return Err(RuntimeError::InvalidArgument("keccak256 input too large".into()));
        }

        let data = ctx.memory.read_slice(data_ptr, data_len as usize)?;
        hasher.update(&data);
    }

    let hash = hasher.finalize();
    ctx.memory.write_slice(result_ptr, &hash)?;
    Ok(0)
}

fn syscall_invoke_signed(
    _ctx: &mut SyscallContext,
    _instruction_ptr: u64,
    _account_infos_ptr: u64,
    _account_infos_len: u64,
    _signers_seeds_ptr: u64,
    _signers_seeds_len: u64,
) -> RuntimeResult<u64> {
    // CPI (Cross-Program Invocation) - needs full executor context
    // This will be wired up to call back into the executor
    Err(RuntimeError::InvalidSyscall(0xc)) // TODO: Implement CPI
}

fn syscall_alloc_free(ctx: &mut SyscallContext, size: u64, free_ptr: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    if free_ptr != 0 {
        // Free operation - bump allocator doesn't actually free
        Ok(0)
    } else {
        // Allocate operation
        ctx.memory.alloc(size as usize)
    }
}

fn syscall_set_return_data(ctx: &mut SyscallContext, data_ptr: u64, data_len: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    if data_len > 1024 {
        return Err(RuntimeError::InvalidArgument("return data too large".into()));
    }
    let data = ctx.memory.read_slice(data_ptr, data_len as usize)?;
    *ctx.return_data = Some((ctx.program_id, data));
    Ok(0)
}

fn syscall_get_return_data(ctx: &mut SyscallContext, data_ptr: u64, data_len: u64, program_id_ptr: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    match ctx.return_data {
        Some((program_id, data)) => {
            let copy_len = std::cmp::min(data_len as usize, data.len());
            ctx.memory.write_slice(data_ptr, &data[..copy_len])?;
            ctx.memory.write_slice(program_id_ptr, &program_id.0)?;
            Ok(data.len() as u64)
        }
        None => Ok(0),
    }
}

fn syscall_get_clock_sysvar(ctx: &mut SyscallContext, out_ptr: u64, _: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    // Write clock sysvar data (40 bytes)
    // slot(8) + epoch_start_timestamp(8) + epoch(8) + leader_schedule_epoch(8) + unix_timestamp(8)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    ctx.memory.write_u64(out_ptr, 0)?;      // slot
    ctx.memory.write_u64(out_ptr + 8, now as u64)?;  // epoch_start_timestamp
    ctx.memory.write_u64(out_ptr + 16, 0)?; // epoch
    ctx.memory.write_u64(out_ptr + 24, 0)?; // leader_schedule_epoch
    ctx.memory.write_u64(out_ptr + 32, now as u64)?; // unix_timestamp

    Ok(0)
}

fn syscall_get_rent_sysvar(ctx: &mut SyscallContext, out_ptr: u64, _: u64, _: u64, _: u64, _: u64) -> RuntimeResult<u64> {
    // Write rent sysvar data (17 bytes packed)
    // lamports_per_byte_year(8) + exemption_threshold(8) + burn_percent(1)
    ctx.memory.write_u64(out_ptr, 3480)?;   // lamports_per_byte_year
    ctx.memory.write_u64(out_ptr + 8, 2)?;  // exemption_threshold (as f64 bits, simplified)
    ctx.memory.write_u8(out_ptr + 16, 50)?; // burn_percent

    Ok(0)
}

// ============================================================================
// Cross-VM Bridge Syscalls (SVM ↔ EVM)
// ============================================================================

fn syscall_call_evm(
    ctx: &mut SyscallContext,
    evm_addr_ptr: u64,
    calldata_ptr: u64,
    calldata_len: u64,
    value_ptr: u64,
    result_ptr: u64,
) -> RuntimeResult<u64> {
    // Read EVM address (20 bytes)
    let evm_addr = ctx.memory.read_slice(evm_addr_ptr, 20)?;

    // Read calldata
    if calldata_len > 100_000 {
        return Err(RuntimeError::InvalidArgument("calldata too large".into()));
    }
    let calldata = ctx.memory.read_slice(calldata_ptr, calldata_len as usize)?;

    // Read value (256-bit, 32 bytes)
    let _value = ctx.memory.read_slice(value_ptr, 32)?;

    // Log the cross-VM call
    let msg = format!(
        "EVM call: to={} data_len={}",
        hex::encode(&evm_addr),
        calldata_len
    );
    ctx.logs.push(msg);

    // TODO: Actually execute EVM call via revm
    // For now, write empty result
    ctx.memory.write_u64(result_ptr, 0)?; // result length = 0

    Ok(0) // Success
}

fn syscall_evm_balance(
    ctx: &mut SyscallContext,
    evm_addr_ptr: u64,
    result_ptr: u64,
    _: u64,
    _: u64,
    _: u64,
) -> RuntimeResult<u64> {
    // Read EVM address (20 bytes)
    let evm_addr = ctx.memory.read_slice(evm_addr_ptr, 20)?;

    // Log the balance query
    let msg = format!("EVM balance query: {}", hex::encode(&evm_addr));
    ctx.logs.push(msg);

    // TODO: Actually query EVM state
    // For now, return 0 balance
    ctx.memory.write_slice(result_ptr, &[0u8; 32])?;

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_registry() {
        let registry = SyscallRegistry::new();
        assert!(registry.syscalls.contains_key(&0x2)); // sol_log_
        assert!(registry.syscalls.contains_key(&0xa)); // sol_sha256
        assert!(registry.syscalls.contains_key(&0x100)); // sol_call_evm
    }
}
