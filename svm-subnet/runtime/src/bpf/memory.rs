use crate::{RuntimeError, RuntimeResult};

/// Memory regions for BPF VM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    Program,
    Stack,
    Heap,
    Input,
}

/// BPF memory configuration
pub const STACK_SIZE: usize = 4096;
pub const HEAP_SIZE: usize = 32 * 1024; // 32KB heap
pub const MAX_INPUT_SIZE: usize = 10 * 1024; // 10KB input data

/// Memory address layout (virtual addresses)
pub const PROGRAM_START: u64 = 0x100000000;
pub const STACK_START: u64 = 0x200000000;
pub const HEAP_START: u64 = 0x300000000;
pub const INPUT_START: u64 = 0x400000000;

/// BPF memory manager
pub struct BpfMemory {
    /// Program bytecode (read-only)
    program: Vec<u8>,
    /// Stack memory
    stack: Vec<u8>,
    /// Heap memory
    heap: Vec<u8>,
    /// Input data (accounts, instruction data)
    input: Vec<u8>,
    /// Heap allocation pointer
    heap_ptr: usize,
}

impl BpfMemory {
    /// Create new BPF memory with program bytecode
    pub fn new(program: Vec<u8>, input: Vec<u8>) -> RuntimeResult<Self> {
        if input.len() > MAX_INPUT_SIZE {
            return Err(RuntimeError::OutOfMemory);
        }

        Ok(BpfMemory {
            program,
            stack: vec![0u8; STACK_SIZE],
            heap: vec![0u8; HEAP_SIZE],
            input,
            heap_ptr: 0,
        })
    }

    /// Translate virtual address to memory region and offset
    fn translate(&self, addr: u64) -> RuntimeResult<(MemoryRegion, usize)> {
        if addr >= PROGRAM_START && addr < STACK_START {
            let offset = (addr - PROGRAM_START) as usize;
            if offset < self.program.len() {
                return Ok((MemoryRegion::Program, offset));
            }
        } else if addr >= STACK_START && addr < HEAP_START {
            let offset = (addr - STACK_START) as usize;
            if offset < self.stack.len() {
                return Ok((MemoryRegion::Stack, offset));
            }
        } else if addr >= HEAP_START && addr < INPUT_START {
            let offset = (addr - HEAP_START) as usize;
            if offset < self.heap.len() {
                return Ok((MemoryRegion::Heap, offset));
            }
        } else if addr >= INPUT_START {
            let offset = (addr - INPUT_START) as usize;
            if offset < self.input.len() {
                return Ok((MemoryRegion::Input, offset));
            }
        }

        Err(RuntimeError::MemoryAccessViolation(addr))
    }

    /// Read a byte from memory
    pub fn read_u8(&self, addr: u64) -> RuntimeResult<u8> {
        let (region, offset) = self.translate(addr)?;
        let data = match region {
            MemoryRegion::Program => &self.program,
            MemoryRegion::Stack => &self.stack,
            MemoryRegion::Heap => &self.heap,
            MemoryRegion::Input => &self.input,
        };
        Ok(data[offset])
    }

    /// Read a 16-bit value from memory
    pub fn read_u16(&self, addr: u64) -> RuntimeResult<u16> {
        let (region, offset) = self.translate(addr)?;
        let data = match region {
            MemoryRegion::Program => &self.program,
            MemoryRegion::Stack => &self.stack,
            MemoryRegion::Heap => &self.heap,
            MemoryRegion::Input => &self.input,
        };
        if offset + 2 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
    }

    /// Read a 32-bit value from memory
    pub fn read_u32(&self, addr: u64) -> RuntimeResult<u32> {
        let (region, offset) = self.translate(addr)?;
        let data = match region {
            MemoryRegion::Program => &self.program,
            MemoryRegion::Stack => &self.stack,
            MemoryRegion::Heap => &self.heap,
            MemoryRegion::Input => &self.input,
        };
        if offset + 4 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        Ok(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]))
    }

    /// Read a 64-bit value from memory
    pub fn read_u64(&self, addr: u64) -> RuntimeResult<u64> {
        let (region, offset) = self.translate(addr)?;
        let data = match region {
            MemoryRegion::Program => &self.program,
            MemoryRegion::Stack => &self.stack,
            MemoryRegion::Heap => &self.heap,
            MemoryRegion::Input => &self.input,
        };
        if offset + 8 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        Ok(u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]))
    }

    /// Write a byte to memory
    pub fn write_u8(&mut self, addr: u64, value: u8) -> RuntimeResult<()> {
        let (region, offset) = self.translate(addr)?;
        if region == MemoryRegion::Program {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let data = match region {
            MemoryRegion::Stack => &mut self.stack,
            MemoryRegion::Heap => &mut self.heap,
            MemoryRegion::Input => &mut self.input,
            MemoryRegion::Program => unreachable!(),
        };
        data[offset] = value;
        Ok(())
    }

    /// Write a 16-bit value to memory
    pub fn write_u16(&mut self, addr: u64, value: u16) -> RuntimeResult<()> {
        let (region, offset) = self.translate(addr)?;
        if region == MemoryRegion::Program {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let data = match region {
            MemoryRegion::Stack => &mut self.stack,
            MemoryRegion::Heap => &mut self.heap,
            MemoryRegion::Input => &mut self.input,
            MemoryRegion::Program => unreachable!(),
        };
        if offset + 2 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let bytes = value.to_le_bytes();
        data[offset] = bytes[0];
        data[offset + 1] = bytes[1];
        Ok(())
    }

    /// Write a 32-bit value to memory
    pub fn write_u32(&mut self, addr: u64, value: u32) -> RuntimeResult<()> {
        let (region, offset) = self.translate(addr)?;
        if region == MemoryRegion::Program {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let data = match region {
            MemoryRegion::Stack => &mut self.stack,
            MemoryRegion::Heap => &mut self.heap,
            MemoryRegion::Input => &mut self.input,
            MemoryRegion::Program => unreachable!(),
        };
        if offset + 4 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let bytes = value.to_le_bytes();
        data[offset..offset + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a 64-bit value to memory
    pub fn write_u64(&mut self, addr: u64, value: u64) -> RuntimeResult<()> {
        let (region, offset) = self.translate(addr)?;
        if region == MemoryRegion::Program {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let data = match region {
            MemoryRegion::Stack => &mut self.stack,
            MemoryRegion::Heap => &mut self.heap,
            MemoryRegion::Input => &mut self.input,
            MemoryRegion::Program => unreachable!(),
        };
        if offset + 8 > data.len() {
            return Err(RuntimeError::MemoryAccessViolation(addr));
        }
        let bytes = value.to_le_bytes();
        data[offset..offset + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Allocate heap memory
    pub fn alloc(&mut self, size: usize) -> RuntimeResult<u64> {
        let aligned_size = (size + 7) & !7; // 8-byte alignment
        if self.heap_ptr + aligned_size > HEAP_SIZE {
            return Err(RuntimeError::OutOfMemory);
        }

        let addr = HEAP_START + self.heap_ptr as u64;
        self.heap_ptr += aligned_size;
        Ok(addr)
    }

    /// Get stack pointer base address
    pub fn stack_base(&self) -> u64 {
        STACK_START + STACK_SIZE as u64
    }

    /// Get input data start address
    pub fn input_start(&self) -> u64 {
        INPUT_START
    }

    /// Get input data length
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    /// Read slice from memory
    pub fn read_slice(&self, addr: u64, len: usize) -> RuntimeResult<Vec<u8>> {
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self.read_u8(addr + i as u64)?);
        }
        Ok(result)
    }

    /// Write slice to memory
    pub fn write_slice(&mut self, addr: u64, data: &[u8]) -> RuntimeResult<()> {
        for (i, byte) in data.iter().enumerate() {
            self.write_u8(addr + i as u64, *byte)?;
        }
        Ok(())
    }
}
