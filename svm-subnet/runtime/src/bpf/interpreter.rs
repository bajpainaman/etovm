use crate::{RuntimeError, RuntimeResult, Account, Pubkey};
use super::{BpfInstruction, opcodes, parse_program};
use super::memory::BpfMemory;
use super::syscalls::{SyscallRegistry, SyscallContext};

/// BPF registers (r0-r10)
const NUM_REGISTERS: usize = 11;

/// Maximum instructions to execute (prevent infinite loops)
const MAX_INSTRUCTIONS: u64 = 200_000;

/// Default compute units for a transaction
pub const DEFAULT_COMPUTE_UNITS: u64 = 200_000;

/// BPF Virtual Machine
pub struct BpfVm {
    /// Registers r0-r10
    registers: [u64; NUM_REGISTERS],
    /// Program counter (instruction index)
    pc: usize,
    /// Memory manager
    memory: BpfMemory,
    /// Parsed instructions
    instructions: Vec<BpfInstruction>,
    /// Syscall registry
    syscalls: SyscallRegistry,
    /// Compute units remaining
    compute_units: u64,
    /// Program logs
    logs: Vec<String>,
    /// Return data from last CPI
    return_data: Option<(Pubkey, Vec<u8>)>,
}

impl BpfVm {
    /// Create a new BPF VM with program bytecode
    pub fn new(bytecode: &[u8], input: Vec<u8>) -> RuntimeResult<Self> {
        let instructions = parse_program(bytecode)?;
        let memory = BpfMemory::new(bytecode.to_vec(), input)?;

        Ok(BpfVm {
            registers: [0u64; NUM_REGISTERS],
            pc: 0,
            memory,
            instructions,
            syscalls: SyscallRegistry::new(),
            compute_units: DEFAULT_COMPUTE_UNITS,
            logs: Vec::new(),
            return_data: None,
        })
    }

    /// Execute the BPF program
    pub fn execute(&mut self, accounts: &mut [Account], program_id: Pubkey) -> RuntimeResult<u64> {
        // Initialize stack pointer (r10) - points to top of stack (grows down)
        self.registers[10] = self.memory.stack_base();

        // r1 = pointer to input data
        self.registers[1] = self.memory.input_start();

        let mut instruction_count = 0u64;

        while self.pc < self.instructions.len() {
            instruction_count += 1;
            if instruction_count > MAX_INSTRUCTIONS {
                return Err(RuntimeError::ComputeExhausted);
            }

            // Deduct compute units
            if self.compute_units == 0 {
                return Err(RuntimeError::ComputeExhausted);
            }
            self.compute_units -= 1;

            let insn = self.instructions[self.pc];
            self.pc += 1;

            match self.execute_instruction(insn, accounts, program_id)? {
                ExecutionResult::Continue => {}
                ExecutionResult::Exit => break,
            }
        }

        // Return value is in r0
        Ok(self.registers[0])
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        insn: BpfInstruction,
        accounts: &mut [Account],
        program_id: Pubkey,
    ) -> RuntimeResult<ExecutionResult> {
        let dst = insn.dst as usize;
        let src = insn.src as usize;
        let imm = insn.imm as i64;
        let offset = insn.offset as i64;

        match insn.opcode {
            // ================================================================
            // ALU 64-bit operations
            // ================================================================
            opcodes::ADD64_IMM => {
                self.registers[dst] = self.registers[dst].wrapping_add(imm as u64);
            }
            opcodes::ADD64_REG => {
                self.registers[dst] = self.registers[dst].wrapping_add(self.registers[src]);
            }
            opcodes::SUB64_IMM => {
                self.registers[dst] = self.registers[dst].wrapping_sub(imm as u64);
            }
            opcodes::SUB64_REG => {
                self.registers[dst] = self.registers[dst].wrapping_sub(self.registers[src]);
            }
            opcodes::MUL64_IMM => {
                self.registers[dst] = self.registers[dst].wrapping_mul(imm as u64);
            }
            opcodes::MUL64_REG => {
                self.registers[dst] = self.registers[dst].wrapping_mul(self.registers[src]);
            }
            opcodes::DIV64_IMM => {
                if imm == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                self.registers[dst] = self.registers[dst] / (imm as u64);
            }
            opcodes::DIV64_REG => {
                if self.registers[src] == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                self.registers[dst] = self.registers[dst] / self.registers[src];
            }
            opcodes::MOD64_IMM => {
                if imm == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                self.registers[dst] = self.registers[dst] % (imm as u64);
            }
            opcodes::MOD64_REG => {
                if self.registers[src] == 0 {
                    return Err(RuntimeError::DivisionByZero);
                }
                self.registers[dst] = self.registers[dst] % self.registers[src];
            }
            opcodes::OR64_IMM => {
                self.registers[dst] |= imm as u64;
            }
            opcodes::OR64_REG => {
                self.registers[dst] |= self.registers[src];
            }
            opcodes::AND64_IMM => {
                self.registers[dst] &= imm as u64;
            }
            opcodes::AND64_REG => {
                self.registers[dst] &= self.registers[src];
            }
            opcodes::XOR64_IMM => {
                self.registers[dst] ^= imm as u64;
            }
            opcodes::XOR64_REG => {
                self.registers[dst] ^= self.registers[src];
            }
            opcodes::LSH64_IMM => {
                self.registers[dst] <<= imm as u32;
            }
            opcodes::LSH64_REG => {
                self.registers[dst] <<= self.registers[src] as u32;
            }
            opcodes::RSH64_IMM => {
                self.registers[dst] >>= imm as u32;
            }
            opcodes::RSH64_REG => {
                self.registers[dst] >>= self.registers[src] as u32;
            }
            opcodes::ARSH64_IMM => {
                self.registers[dst] = ((self.registers[dst] as i64) >> (imm as u32)) as u64;
            }
            opcodes::ARSH64_REG => {
                self.registers[dst] = ((self.registers[dst] as i64) >> (self.registers[src] as u32)) as u64;
            }
            opcodes::NEG64 => {
                self.registers[dst] = (-(self.registers[dst] as i64)) as u64;
            }
            opcodes::MOV64_IMM => {
                self.registers[dst] = imm as u64;
            }
            opcodes::MOV64_REG => {
                self.registers[dst] = self.registers[src];
            }

            // ================================================================
            // ALU 32-bit operations (zero-extend result to 64-bit)
            // ================================================================
            opcodes::ADD32_IMM => {
                self.registers[dst] = (self.registers[dst] as u32).wrapping_add(imm as u32) as u64;
            }
            opcodes::ADD32_REG => {
                self.registers[dst] = (self.registers[dst] as u32).wrapping_add(self.registers[src] as u32) as u64;
            }
            opcodes::MOV32_IMM => {
                self.registers[dst] = imm as u32 as u64;
            }
            opcodes::MOV32_REG => {
                self.registers[dst] = self.registers[src] as u32 as u64;
            }

            // ================================================================
            // Memory load operations
            // ================================================================
            opcodes::LDXB => {
                let addr = self.registers[src].wrapping_add(offset as u64);
                self.registers[dst] = self.memory.read_u8(addr)? as u64;
            }
            opcodes::LDXH => {
                let addr = self.registers[src].wrapping_add(offset as u64);
                self.registers[dst] = self.memory.read_u16(addr)? as u64;
            }
            opcodes::LDXW => {
                let addr = self.registers[src].wrapping_add(offset as u64);
                self.registers[dst] = self.memory.read_u32(addr)? as u64;
            }
            opcodes::LDXDW => {
                let addr = self.registers[src].wrapping_add(offset as u64);
                self.registers[dst] = self.memory.read_u64(addr)?;
            }

            // ================================================================
            // Memory store operations (immediate)
            // ================================================================
            opcodes::STB => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u8(addr, imm as u8)?;
            }
            opcodes::STH => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u16(addr, imm as u16)?;
            }
            opcodes::STW => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u32(addr, imm as u32)?;
            }
            opcodes::STDW => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u64(addr, imm as u64)?;
            }

            // ================================================================
            // Memory store operations (from register)
            // ================================================================
            opcodes::STXB => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u8(addr, self.registers[src] as u8)?;
            }
            opcodes::STXH => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u16(addr, self.registers[src] as u16)?;
            }
            opcodes::STXW => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u32(addr, self.registers[src] as u32)?;
            }
            opcodes::STXDW => {
                let addr = self.registers[dst].wrapping_add(offset as u64);
                self.memory.write_u64(addr, self.registers[src])?;
            }

            // ================================================================
            // Load 64-bit immediate (2-instruction sequence)
            // ================================================================
            opcodes::LDDW => {
                // This instruction uses the next instruction's imm field for high 32 bits
                if self.pc >= self.instructions.len() {
                    return Err(RuntimeError::InvalidInstruction("LDDW missing second instruction".into()));
                }
                let next_insn = self.instructions[self.pc];
                self.pc += 1;

                let low = imm as u32 as u64;
                let high = (next_insn.imm as u32 as u64) << 32;
                self.registers[dst] = low | high;
            }

            // ================================================================
            // Jump operations
            // ================================================================
            opcodes::JA => {
                self.pc = (self.pc as i64 + offset) as usize;
            }
            opcodes::JEQ_IMM => {
                if self.registers[dst] == imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JEQ_REG => {
                if self.registers[dst] == self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JNE_IMM => {
                if self.registers[dst] != imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JNE_REG => {
                if self.registers[dst] != self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JGT_IMM => {
                if self.registers[dst] > imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JGT_REG => {
                if self.registers[dst] > self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JGE_IMM => {
                if self.registers[dst] >= imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JGE_REG => {
                if self.registers[dst] >= self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JLT_IMM => {
                if self.registers[dst] < imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JLT_REG => {
                if self.registers[dst] < self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JLE_IMM => {
                if self.registers[dst] <= imm as u64 {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JLE_REG => {
                if self.registers[dst] <= self.registers[src] {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JSGT_IMM => {
                if (self.registers[dst] as i64) > imm {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JSGT_REG => {
                if (self.registers[dst] as i64) > (self.registers[src] as i64) {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JSGE_IMM => {
                if (self.registers[dst] as i64) >= imm {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }
            opcodes::JSGE_REG => {
                if (self.registers[dst] as i64) >= (self.registers[src] as i64) {
                    self.pc = (self.pc as i64 + offset) as usize;
                }
            }

            // ================================================================
            // Call and Exit
            // ================================================================
            opcodes::CALL => {
                // Syscall - imm contains syscall ID
                let syscall_id = imm as u32;
                let mut ctx = SyscallContext {
                    memory: &mut self.memory,
                    accounts,
                    program_id,
                    compute_units: &mut self.compute_units,
                    return_data: &mut self.return_data,
                    logs: &mut self.logs,
                };

                // r1-r5 are syscall arguments, result goes in r0
                self.registers[0] = self.syscalls.invoke(
                    &mut ctx,
                    syscall_id,
                    self.registers[1],
                    self.registers[2],
                    self.registers[3],
                    self.registers[4],
                    self.registers[5],
                )?;
            }
            opcodes::EXIT => {
                return Ok(ExecutionResult::Exit);
            }

            _ => {
                return Err(RuntimeError::InvalidInstruction(
                    format!("unknown opcode: 0x{:02x}", insn.opcode)
                ));
            }
        }

        Ok(ExecutionResult::Continue)
    }

    /// Get program logs
    pub fn logs(&self) -> &[String] {
        &self.logs
    }

    /// Get remaining compute units
    pub fn compute_units(&self) -> u64 {
        self.compute_units
    }

    /// Set compute units limit
    pub fn set_compute_units(&mut self, units: u64) {
        self.compute_units = units;
    }
}

/// Result of executing a single instruction
enum ExecutionResult {
    Continue,
    Exit,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_program() -> Vec<u8> {
        // mov64 r0, 42
        // exit
        let mut bytecode = Vec::new();

        // mov64 r0, 42 -> opcode=0xb7, dst=0, src=0, offset=0, imm=42
        bytecode.extend_from_slice(&[0xb7, 0x00, 0x00, 0x00, 42, 0, 0, 0]);

        // exit -> opcode=0x95, dst=0, src=0, offset=0, imm=0
        bytecode.extend_from_slice(&[0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0]);

        bytecode
    }

    #[test]
    fn test_simple_execution() {
        let bytecode = simple_program();
        let mut vm = BpfVm::new(&bytecode, vec![]).unwrap();
        let mut accounts = vec![];
        let program_id = Pubkey([0u8; 32]);

        let result = vm.execute(&mut accounts, program_id).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_arithmetic() {
        // r0 = 10
        // r0 = r0 + 5
        // r0 = r0 * 2
        // exit
        let bytecode = vec![
            0xb7, 0x00, 0x00, 0x00, 10, 0, 0, 0,  // mov64 r0, 10
            0x07, 0x00, 0x00, 0x00, 5, 0, 0, 0,   // add64 r0, 5
            0x27, 0x00, 0x00, 0x00, 2, 0, 0, 0,   // mul64 r0, 2
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,   // exit
        ];

        let mut vm = BpfVm::new(&bytecode, vec![]).unwrap();
        let mut accounts = vec![];
        let program_id = Pubkey([0u8; 32]);

        let result = vm.execute(&mut accounts, program_id).unwrap();
        assert_eq!(result, 30); // (10 + 5) * 2 = 30
    }

    #[test]
    fn test_conditional_jump() {
        // r0 = 5
        // if r0 == 5: jump +1
        // r0 = 99  (skipped)
        // r0 = r0 + 1
        // exit
        let bytecode = vec![
            0xb7, 0x00, 0x00, 0x00, 5, 0, 0, 0,   // mov64 r0, 5
            0x15, 0x00, 0x01, 0x00, 5, 0, 0, 0,   // jeq r0, 5, +1
            0xb7, 0x00, 0x00, 0x00, 99, 0, 0, 0,  // mov64 r0, 99 (skipped)
            0x07, 0x00, 0x00, 0x00, 1, 0, 0, 0,   // add64 r0, 1
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,   // exit
        ];

        let mut vm = BpfVm::new(&bytecode, vec![]).unwrap();
        let mut accounts = vec![];
        let program_id = Pubkey([0u8; 32]);

        let result = vm.execute(&mut accounts, program_id).unwrap();
        assert_eq!(result, 6); // 5 + 1 = 6 (99 was skipped)
    }

    #[test]
    fn test_division_by_zero() {
        let bytecode = vec![
            0xb7, 0x00, 0x00, 0x00, 10, 0, 0, 0,  // mov64 r0, 10
            0x37, 0x00, 0x00, 0x00, 0, 0, 0, 0,   // div64 r0, 0
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,   // exit
        ];

        let mut vm = BpfVm::new(&bytecode, vec![]).unwrap();
        let mut accounts = vec![];
        let program_id = Pubkey([0u8; 32]);

        let result = vm.execute(&mut accounts, program_id);
        assert!(matches!(result, Err(RuntimeError::DivisionByZero)));
    }
}
