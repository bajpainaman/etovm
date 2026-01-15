//! High-Performance Memory Pools
//!
//! Pre-allocated memory pools for zero-allocation transaction execution.
//! Uses bump allocation within blocks for cache-friendly sequential access.

use bumpalo::Bump;
use crossbeam_queue::ArrayQueue;
use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::sync::Arc;

/// Size of each memory arena (4MB - fits in L3 cache)
const ARENA_SIZE: usize = 4 * 1024 * 1024;

/// Maximum number of pooled arenas
const MAX_POOLED_ARENAS: usize = 64;

/// Thread-local execution context with pre-allocated buffers
pub struct ExecutionArena {
    /// Bump allocator for this arena
    bump: UnsafeCell<Bump>,
    /// Pre-allocated account data buffer
    account_buffer: Vec<u8>,
    /// Pre-allocated state change buffer
    change_buffer: Vec<u8>,
}

// Safety: ExecutionArena is only accessed by one thread at a time via the pool
unsafe impl Send for ExecutionArena {}
unsafe impl Sync for ExecutionArena {}

impl ExecutionArena {
    /// Create a new execution arena
    pub fn new() -> Self {
        Self {
            bump: UnsafeCell::new(Bump::with_capacity(ARENA_SIZE)),
            account_buffer: vec![0u8; 1024 * 1024], // 1MB for accounts
            change_buffer: vec![0u8; 1024 * 1024],  // 1MB for changes
        }
    }

    /// Reset the arena for reuse
    pub fn reset(&mut self) {
        // Safety: We have exclusive access via &mut self
        unsafe {
            (*self.bump.get()).reset();
        }
    }

    /// Get the bump allocator
    pub fn bump(&self) -> &Bump {
        // Safety: Single-threaded access guaranteed by pool checkout
        unsafe { &*self.bump.get() }
    }

    /// Get mutable bump allocator
    pub fn bump_mut(&mut self) -> &mut Bump {
        self.bump.get_mut()
    }

    /// Get account buffer slice
    pub fn account_buffer(&mut self, size: usize) -> &mut [u8] {
        if size > self.account_buffer.len() {
            self.account_buffer.resize(size, 0);
        }
        &mut self.account_buffer[..size]
    }

    /// Get change buffer slice
    pub fn change_buffer(&mut self, size: usize) -> &mut [u8] {
        if size > self.change_buffer.len() {
            self.change_buffer.resize(size, 0);
        }
        &mut self.change_buffer[..size]
    }
}

impl Default for ExecutionArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool of execution arenas for multi-threaded access
pub struct ArenaPool {
    /// Lock-free queue of available arenas
    available: ArrayQueue<Box<ExecutionArena>>,
    /// Fallback allocation under contention
    fallback_count: std::sync::atomic::AtomicUsize,
}

impl ArenaPool {
    /// Create a new arena pool with pre-allocated arenas
    pub fn new(initial_count: usize) -> Self {
        let available = ArrayQueue::new(MAX_POOLED_ARENAS);

        for _ in 0..initial_count.min(MAX_POOLED_ARENAS) {
            let _ = available.push(Box::new(ExecutionArena::new()));
        }

        Self {
            available,
            fallback_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Checkout an arena from the pool
    pub fn checkout(&self) -> ArenaGuard {
        match self.available.pop() {
            Some(mut arena) => {
                arena.reset();
                ArenaGuard {
                    arena: Some(arena),
                    pool: self,
                }
            }
            None => {
                // Pool exhausted - allocate new (this should be rare)
                self.fallback_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                ArenaGuard {
                    arena: Some(Box::new(ExecutionArena::new())),
                    pool: self,
                }
            }
        }
    }

    /// Return an arena to the pool
    fn return_arena(&self, mut arena: Box<ExecutionArena>) {
        arena.reset();
        // Try to return to pool, drop if full
        let _ = self.available.push(arena);
    }

    /// Get fallback allocation count (for monitoring)
    pub fn fallback_count(&self) -> usize {
        self.fallback_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get current pool size
    pub fn available_count(&self) -> usize {
        self.available.len()
    }
}

/// RAII guard for arena checkout
pub struct ArenaGuard<'a> {
    arena: Option<Box<ExecutionArena>>,
    pool: &'a ArenaPool,
}

impl<'a> ArenaGuard<'a> {
    /// Get reference to the arena
    pub fn arena(&self) -> &ExecutionArena {
        self.arena.as_ref().unwrap()
    }

    /// Get mutable reference to the arena
    pub fn arena_mut(&mut self) -> &mut ExecutionArena {
        self.arena.as_mut().unwrap()
    }
}

impl<'a> Drop for ArenaGuard<'a> {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.take() {
            self.pool.return_arena(arena);
        }
    }
}

impl<'a> std::ops::Deref for ArenaGuard<'a> {
    type Target = ExecutionArena;

    fn deref(&self) -> &Self::Target {
        self.arena.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for ArenaGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arena.as_mut().unwrap()
    }
}

/// Pre-allocated buffer for transaction results
pub struct ResultBuffer {
    /// Success flags
    pub success: Vec<bool>,
    /// Compute units used per tx
    pub compute_units: Vec<u64>,
    /// Fees per tx
    pub fees: Vec<u64>,
    /// Error messages (optional, sparse)
    pub errors: Vec<Option<String>>,
}

impl ResultBuffer {
    /// Create a new result buffer with capacity
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            success: Vec::with_capacity(cap),
            compute_units: Vec::with_capacity(cap),
            fees: Vec::with_capacity(cap),
            errors: Vec::with_capacity(cap),
        }
    }

    /// Reset the buffer for reuse
    pub fn reset(&mut self) {
        self.success.clear();
        self.compute_units.clear();
        self.fees.clear();
        self.errors.clear();
    }

    /// Ensure capacity
    pub fn ensure_capacity(&mut self, cap: usize) {
        if self.success.capacity() < cap {
            self.success.reserve(cap - self.success.capacity());
            self.compute_units.reserve(cap - self.compute_units.capacity());
            self.fees.reserve(cap - self.fees.capacity());
            self.errors.reserve(cap - self.errors.capacity());
        }
    }

    /// Record a successful execution
    #[inline]
    pub fn record_success(&mut self, compute: u64, fee: u64) {
        self.success.push(true);
        self.compute_units.push(compute);
        self.fees.push(fee);
        self.errors.push(None);
    }

    /// Record a failed execution
    #[inline]
    pub fn record_failure(&mut self, error: String) {
        self.success.push(false);
        self.compute_units.push(0);
        self.fees.push(0);
        self.errors.push(Some(error));
    }
}

/// Thread-local storage for per-thread execution state
pub struct ThreadLocalState {
    /// Per-thread arena
    arena: ExecutionArena,
    /// Per-thread result buffer
    results: ResultBuffer,
}

impl ThreadLocalState {
    pub fn new() -> Self {
        Self {
            arena: ExecutionArena::new(),
            results: ResultBuffer::with_capacity(10000),
        }
    }

    pub fn arena(&mut self) -> &mut ExecutionArena {
        &mut self.arena
    }

    pub fn results(&mut self) -> &mut ResultBuffer {
        &mut self.results
    }

    pub fn reset(&mut self) {
        self.arena.reset();
        self.results.reset();
    }
}

impl Default for ThreadLocalState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_pool() {
        let pool = ArenaPool::new(4);

        assert_eq!(pool.available_count(), 4);

        // Checkout
        let guard = pool.checkout();
        assert_eq!(pool.available_count(), 3);

        // Can use arena
        let _ = guard.arena().bump().alloc([0u8; 1024]);

        // Return on drop
        drop(guard);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_arena_pool_exhaustion() {
        let pool = ArenaPool::new(2);

        let _g1 = pool.checkout();
        let _g2 = pool.checkout();

        // Pool exhausted - should allocate new
        let _g3 = pool.checkout();
        assert_eq!(pool.fallback_count(), 1);
    }

    #[test]
    fn test_result_buffer() {
        let mut buf = ResultBuffer::with_capacity(100);

        buf.record_success(450, 5000);
        buf.record_success(300, 5000);
        buf.record_failure("test error".into());

        assert_eq!(buf.success.len(), 3);
        assert!(buf.success[0]);
        assert!(buf.success[1]);
        assert!(!buf.success[2]);

        buf.reset();
        assert_eq!(buf.success.len(), 0);
    }
}
