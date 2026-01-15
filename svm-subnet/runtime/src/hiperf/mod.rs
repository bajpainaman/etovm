//! High-Performance Execution Engine
//!
//! Target: 200k+ TPS sustained on mainnet
//!
//! Architecture:
//! ```text
//!                    ┌─────────────────────────────────────────────┐
//!                    │            Transaction Ingestion            │
//!                    └─────────────────┬───────────────────────────┘
//!                                      │
//!                    ┌─────────────────▼───────────────────────────┐
//!                    │     Stage 1: Parallel Batch Verification    │
//!                    │     - Ed25519 batch verify (8x speedup)     │
//!                    │     - SHA256 SIMD hashing                   │
//!                    └─────────────────┬───────────────────────────┘
//!                                      │ (lock-free channel)
//!                    ┌─────────────────▼───────────────────────────┐
//!                    │     Stage 2: Conflict Analysis & Schedule   │
//!                    │     - Parallel access set extraction        │
//!                    │     - Lock-free batch formation             │
//!                    └─────────────────┬───────────────────────────┘
//!                                      │ (lock-free channel)
//!    ┌──────────────────┬──────────────┼──────────────┬──────────────────┐
//!    │                  │              │              │                  │
//!    ▼                  ▼              ▼              ▼                  ▼
//! ┌──────┐          ┌──────┐      ┌──────┐      ┌──────┐          ┌──────┐
//! │Worker│          │Worker│      │Worker│      │Worker│          │Worker│
//! │  1   │          │  2   │      │  3   │      │  4   │          │  N   │
//! └──┬───┘          └──┬───┘      └──┬───┘      └──┬───┘          └──┬───┘
//!    │                 │             │             │                 │
//!    └────────────────────────────┬──┴─────────────┴─────────────────┘
//!                                 │ (aggregation)
//!                    ┌────────────▼────────────────────────────────┐
//!                    │     Stage 3: Parallel State Commit          │
//!                    │     - Lock-free changeset merge             │
//!                    │     - Parallel merkle tree computation      │
//!                    └─────────────────────────────────────────────┘
//! ```

mod batch_verifier;
mod fast_scheduler;
mod incremental_merkle;
mod memory_pool;
mod parallel_merkle;
mod pipeline;
mod simd_hash;
mod turbo_executor;

pub use batch_verifier::*;
pub use fast_scheduler::*;
pub use incremental_merkle::*;
pub use memory_pool::*;
pub use parallel_merkle::*;
pub use pipeline::*;
pub use simd_hash::*;
pub use turbo_executor::*;
