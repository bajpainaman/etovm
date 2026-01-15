//! Sealevel Performance Benchmark
//!
//! Benchmarks parallel transaction execution to measure TPS.
//! Includes both standard Sealevel and QMDB-optimized benchmarks.

use super::{AccessSet, BatchScheduler, ParallelExecutor, ParallelExecutorConfig};
use super::{QMDBParallelExecutor, QMDBExecutorConfig};
use crate::accounts::{AccountsManager, InMemoryAccountsDB};
use crate::executor::{ExecutionContext, ExecutorConfig};
use crate::qmdb_state::InMemoryQMDBState;
use crate::types::{Account, CompiledInstruction, Message, MessageHeader, Pubkey, Transaction};
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    /// Number of transactions to execute
    pub num_transactions: usize,
    /// Percentage of transactions that conflict (0-100)
    pub conflict_rate: u8,
    /// Number of unique accounts in the test
    pub num_accounts: usize,
    /// Number of warm-up transactions before measurement
    pub warmup_txs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_transactions: 30000,
            conflict_rate: 10, // 10% conflict rate
            num_accounts: 100000,
            warmup_txs: 1000,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub total_transactions: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_time_us: u64,
    pub tps: f64,
    pub avg_latency_us: f64,
    pub num_batches: usize,
    pub avg_batch_size: f64,
    pub parallelism_factor: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"
╔══════════════════════════════════════════════════════════════╗
║                    SEALEVEL BENCHMARK RESULTS                ║
╠══════════════════════════════════════════════════════════════╣
║  Total Transactions:    {:>10}                            ║
║  Successful:            {:>10}                            ║
║  Failed:                {:>10}                            ║
╠══════════════════════════════════════════════════════════════╣
║  Total Time:            {:>10.2} ms                        ║
║  TPS:                   {:>10.0}                            ║
║  Avg Latency:           {:>10.2} µs                        ║
╠══════════════════════════════════════════════════════════════╣
║  Batches:               {:>10}                            ║
║  Avg Batch Size:        {:>10.1}                            ║
║  Parallelism Factor:    {:>10.1}x                           ║
╚══════════════════════════════════════════════════════════════╝
"#,
            self.total_transactions,
            self.successful,
            self.failed,
            self.total_time_us as f64 / 1000.0,
            self.tps,
            self.avg_latency_us,
            self.num_batches,
            self.avg_batch_size,
            self.parallelism_factor,
        )
    }
}

/// Generate a test pubkey
fn make_pubkey(seed: u64) -> Pubkey {
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&seed.to_le_bytes());
    Pubkey(bytes)
}

/// Generate test transactions with configurable conflict rate
pub fn generate_test_transactions(config: &BenchmarkConfig) -> Vec<Transaction> {
    let mut transactions = Vec::with_capacity(config.num_transactions);

    for i in 0..config.num_transactions {
        // Determine if this transaction should conflict
        let should_conflict = (i as u8 % 100) < config.conflict_rate;

        // Pick accounts based on conflict rate
        let (from_seed, to_seed) = if should_conflict {
            // Use a small set of accounts to create conflicts
            let bucket = i % 10;
            (bucket as u64, (bucket + 1) as u64)
        } else {
            // Use unique accounts for no conflicts
            let base = i as u64 * 2;
            (base, base + 1)
        };

        let from = make_pubkey(from_seed);
        let to = make_pubkey(to_seed);

        let tx = Transaction {
            signatures: vec![[0u8; 64]], // Dummy signature
            message: Message {
                header: MessageHeader {
                    num_required_signatures: 1,
                    num_readonly_signed_accounts: 0,
                    num_readonly_unsigned_accounts: 1,
                },
                account_keys: vec![from, to, Pubkey::system_program()],
                recent_blockhash: [0u8; 32],
                instructions: vec![CompiledInstruction {
                    program_id_index: 2,
                    accounts: vec![0, 1],
                    data: vec![2, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0], // Transfer 100 lamports
                }],
            },
        };

        transactions.push(tx);
    }

    transactions
}

/// Generate access sets for scheduling benchmark (faster than full execution)
pub fn generate_access_sets(config: &BenchmarkConfig) -> Vec<AccessSet> {
    let mut access_sets = Vec::with_capacity(config.num_transactions);

    for i in 0..config.num_transactions {
        let should_conflict = (i as u8 % 100) < config.conflict_rate;

        let mut access = AccessSet::new();

        if should_conflict {
            // Conflicting transactions write to shared accounts
            let bucket = i % 10;
            access.add_write(make_pubkey(bucket as u64));
            access.add_write(make_pubkey((bucket + 1) as u64));
        } else {
            // Non-conflicting transactions have unique accounts
            let base = (i as u64 + 1000) * 2;
            access.add_write(make_pubkey(base));
            access.add_write(make_pubkey(base + 1));
        }

        access_sets.push(access);
    }

    access_sets
}

/// Run scheduling-only benchmark (tests parallelism without execution overhead)
pub fn benchmark_scheduling(config: &BenchmarkConfig) -> BenchmarkResult {
    println!("Generating {} access sets...", config.num_transactions);
    let access_sets = generate_access_sets(config);

    println!("Running scheduling benchmark...");
    let scheduler = BatchScheduler::new(config.num_transactions, 10000);

    let start = Instant::now();
    let batches = scheduler.schedule_with_dependencies(&access_sets);
    let elapsed = start.elapsed();

    let total_time_us = elapsed.as_micros() as u64;
    let num_batches = batches.len();
    let avg_batch_size = config.num_transactions as f64 / num_batches as f64;

    // Calculate TPS (scheduling is much faster than execution)
    // This gives theoretical max if execution was instant
    let tps = if total_time_us > 0 {
        (config.num_transactions as f64 * 1_000_000.0) / total_time_us as f64
    } else {
        f64::INFINITY
    };

    BenchmarkResult {
        total_transactions: config.num_transactions,
        successful: config.num_transactions,
        failed: 0,
        total_time_us,
        tps,
        avg_latency_us: total_time_us as f64 / config.num_transactions as f64,
        num_batches,
        avg_batch_size,
        parallelism_factor: avg_batch_size,
    }
}

/// Run full execution benchmark
pub fn benchmark_execution(config: &BenchmarkConfig) -> BenchmarkResult {
    println!("Setting up executor...");

    // Create accounts with initial balances
    let mut initial_accounts = Vec::new();
    for i in 0..config.num_accounts {
        let pubkey = make_pubkey(i as u64);
        let account = Account {
            lamports: 1_000_000_000, // 1 SOL
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        };
        initial_accounts.push((pubkey, account));
    }

    let db = InMemoryAccountsDB::with_accounts(initial_accounts);
    let accounts = AccountsManager::new(db);

    let exec_config = ParallelExecutorConfig {
        max_batch_size: config.num_transactions,
        num_threads: 0, // Use all CPUs
        speculative_execution: false,
        executor_config: ExecutorConfig::default(),
    };

    let executor = ParallelExecutor::new(accounts, exec_config);

    // Create execution context
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    // Generate transactions
    println!("Generating {} transactions...", config.num_transactions);
    let transactions = generate_test_transactions(config);

    // Warmup
    if config.warmup_txs > 0 {
        println!("Running {} warmup transactions...", config.warmup_txs);
        let warmup_txs = &transactions[..config.warmup_txs.min(transactions.len())];
        let _ = executor.execute_transactions(warmup_txs, &ctx);
    }

    // Benchmark
    println!("Running benchmark with {} transactions...", config.num_transactions);
    let start = Instant::now();
    let result = executor.execute_transactions(&transactions, &ctx);
    let elapsed = start.elapsed();

    let total_time_us = elapsed.as_micros() as u64;
    let tps = if total_time_us > 0 {
        (config.num_transactions as f64 * 1_000_000.0) / total_time_us as f64
    } else {
        f64::INFINITY
    };

    // Get stats
    let stats = executor.get_stats();

    BenchmarkResult {
        total_transactions: config.num_transactions,
        successful: result.successful,
        failed: result.failed,
        total_time_us,
        tps,
        avg_latency_us: total_time_us as f64 / config.num_transactions as f64,
        num_batches: stats.total_batches_processed as usize,
        avg_batch_size: if stats.total_batches_processed > 0 {
            config.num_transactions as f64 / stats.total_batches_processed as f64
        } else {
            0.0
        },
        parallelism_factor: if stats.total_batches_processed > 0 {
            config.num_transactions as f64 / stats.total_batches_processed as f64
        } else {
            1.0
        },
    }
}

/// Run progressive benchmark from 10k to 40k
pub fn benchmark_progressive() {
    println!("\n🚀 SEALEVEL PROGRESSIVE BENCHMARK\n");
    println!("Testing scheduling performance at different scales...\n");

    let test_sizes = [10_000, 20_000, 30_000, 40_000];
    let conflict_rates = [0, 10, 25, 50];

    println!("╔════════════╦═══════════╦═══════════╦═══════════╦══════════════╗");
    println!("║    Txs     ║ Conflicts ║  Batches  ║ Parallel  ║ Schedule TPS ║");
    println!("╠════════════╬═══════════╬═══════════╬═══════════╬══════════════╣");

    for &size in &test_sizes {
        for &conflict_rate in &conflict_rates {
            let config = BenchmarkConfig {
                num_transactions: size,
                conflict_rate,
                num_accounts: size * 2,
                warmup_txs: 0,
            };

            let result = benchmark_scheduling(&config);

            println!(
                "║ {:>10} ║ {:>8}% ║ {:>9} ║ {:>8.1}x ║ {:>12.0} ║",
                size,
                conflict_rate,
                result.num_batches,
                result.parallelism_factor,
                result.tps
            );
        }
        println!("╠════════════╬═══════════╬═══════════╬═══════════╬══════════════╣");
    }
    println!("╚════════════╩═══════════╩═══════════╩═══════════╩══════════════╝");
}

// ============================================================================
// QMDB BENCHMARKS
// ============================================================================

/// QMDB benchmark result
#[derive(Debug, Clone)]
pub struct QMDBBenchmarkResult {
    pub total_transactions: usize,
    pub successful: usize,
    pub failed: usize,
    pub total_time_us: u64,
    pub tps: f64,
    pub merkle_root: [u8; 32],
    pub num_batches: usize,
}

impl std::fmt::Display for QMDBBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"
╔══════════════════════════════════════════════════════════════╗
║                  QMDB EXECUTOR BENCHMARK                     ║
╠══════════════════════════════════════════════════════════════╣
║  Total Transactions:    {:>10}                            ║
║  Successful:            {:>10}                            ║
║  Failed:                {:>10}                            ║
╠══════════════════════════════════════════════════════════════╣
║  Total Time:            {:>10.2} ms                        ║
║  TPS:                   {:>10.0}                            ║
║  Batches:               {:>10}                            ║
║  State Root:            {:>10}...                          ║
╚══════════════════════════════════════════════════════════════╝
"#,
            self.total_transactions,
            self.successful,
            self.failed,
            self.total_time_us as f64 / 1000.0,
            self.tps,
            self.num_batches,
            hex::encode(&self.merkle_root[0..4]),
        )
    }
}

/// Setup QMDB state with test accounts
fn setup_qmdb_state(num_accounts: usize) -> InMemoryQMDBState {
    let state = InMemoryQMDBState::new();

    for i in 0..num_accounts {
        let pubkey = make_pubkey(i as u64);
        let account = Account {
            lamports: 1_000_000_000, // 1 SOL
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        };
        state.set_account(&pubkey, &account).unwrap();
    }

    state
}

/// Benchmark QMDB block execution
pub fn benchmark_qmdb_execution(config: &BenchmarkConfig) -> QMDBBenchmarkResult {
    use std::sync::Arc;

    println!("Setting up QMDB state with {} accounts...", config.num_accounts);
    let state = Arc::new(setup_qmdb_state(config.num_accounts));

    let exec_config = QMDBExecutorConfig {
        max_batch_size: config.num_transactions,
        num_threads: 0, // Use all CPUs
        speculative_execution: false,
        executor_config: ExecutorConfig::default(),
    };

    let executor = QMDBParallelExecutor::new(state, exec_config);

    // Create execution context
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    // Generate transactions
    println!("Generating {} transactions...", config.num_transactions);
    let transactions = generate_test_transactions(config);

    // Execute block
    println!("Executing block...");
    let start = Instant::now();
    let result = executor.execute_block(1, &transactions, &ctx).unwrap();
    let elapsed = start.elapsed();

    let total_time_us = elapsed.as_micros() as u64;
    let tps = if total_time_us > 0 {
        (config.num_transactions as f64 * 1_000_000.0) / total_time_us as f64
    } else {
        f64::INFINITY
    };

    QMDBBenchmarkResult {
        total_transactions: config.num_transactions,
        successful: result.successful,
        failed: result.failed,
        total_time_us,
        tps,
        merkle_root: result.merkle_root,
        num_batches: result.num_batches,
    }
}

/// Run progressive QMDB benchmark
pub fn benchmark_qmdb_progressive() {
    println!("\n🚀 QMDB PARALLEL EXECUTOR BENCHMARK\n");
    println!("Testing block execution with state commits...\n");

    let test_sizes = [1_000, 5_000, 10_000, 20_000];
    let conflict_rates = [0, 10, 25];

    println!("╔════════════╦═══════════╦═══════════╦═══════════╦══════════════╦══════════════╗");
    println!("║    Txs     ║ Conflicts ║ Successful║  Batches  ║    Time ms   ║     TPS      ║");
    println!("╠════════════╬═══════════╬═══════════╬═══════════╬══════════════╬══════════════╣");

    for &size in &test_sizes {
        for &conflict_rate in &conflict_rates {
            let config = BenchmarkConfig {
                num_transactions: size,
                conflict_rate,
                num_accounts: size * 2,
                warmup_txs: 0,
            };

            let result = benchmark_qmdb_execution(&config);

            println!(
                "║ {:>10} ║ {:>8}% ║ {:>9} ║ {:>9} ║ {:>12.2} ║ {:>12.0} ║",
                size,
                conflict_rate,
                result.successful,
                result.num_batches,
                result.total_time_us as f64 / 1000.0,
                result.tps
            );
        }
        println!("╠════════════╬═══════════╬═══════════╬═══════════╬══════════════╬══════════════╣");
    }
    println!("╚════════════╩═══════════╩═══════════╩═══════════╩══════════════╩══════════════╝");
}

/// Compare scheduling vs QMDB execution performance
pub fn benchmark_comparison() {
    println!("\n⚡ PERFORMANCE COMPARISON: Scheduling vs QMDB Execution\n");

    let config = BenchmarkConfig {
        num_transactions: 10_000,
        conflict_rate: 10,
        num_accounts: 20_000,
        warmup_txs: 0,
    };

    println!("Configuration: {} txs, {}% conflicts\n", config.num_transactions, config.conflict_rate);

    // Scheduling benchmark
    println!("Running scheduling benchmark...");
    let sched_result = benchmark_scheduling(&config);

    // QMDB execution benchmark
    println!("Running QMDB execution benchmark...");
    let qmdb_result = benchmark_qmdb_execution(&config);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON RESULTS                        ║");
    println!("╠═══════════════════════╦══════════════╦══════════════════════╣");
    println!("║       Metric          ║  Scheduling  ║  QMDB Execution      ║");
    println!("╠═══════════════════════╬══════════════╬══════════════════════╣");
    println!("║ Total Time (ms)       ║ {:>12.2} ║ {:>20.2} ║",
        sched_result.total_time_us as f64 / 1000.0,
        qmdb_result.total_time_us as f64 / 1000.0);
    println!("║ TPS                   ║ {:>12.0} ║ {:>20.0} ║",
        sched_result.tps,
        qmdb_result.tps);
    println!("║ Batches               ║ {:>12} ║ {:>20} ║",
        sched_result.num_batches,
        qmdb_result.num_batches);
    println!("║ Successful            ║ {:>12} ║ {:>20} ║",
        sched_result.successful,
        qmdb_result.successful);
    println!("╚═══════════════════════╩══════════════╩══════════════════════╝");

    let overhead_factor = qmdb_result.total_time_us as f64 / sched_result.total_time_us as f64;
    println!("\n📊 Execution overhead: {:.1}x scheduling time", overhead_factor);
    println!("   (State access + Merkle root computation included)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_10k_no_conflicts() {
        let config = BenchmarkConfig {
            num_transactions: 10_000,
            conflict_rate: 0,
            num_accounts: 20_000,
            warmup_txs: 0,
        };

        let result = benchmark_scheduling(&config);
        println!("{}", result);

        // With 0% conflicts, should achieve high parallelism
        assert!(result.parallelism_factor > 100.0,
            "Expected high parallelism with no conflicts");
    }

    #[test]
    fn test_benchmark_30k_low_conflicts() {
        let config = BenchmarkConfig {
            num_transactions: 30_000,
            conflict_rate: 10,
            num_accounts: 60_000,
            warmup_txs: 0,
        };

        let result = benchmark_scheduling(&config);
        println!("{}", result);

        assert!(result.tps > 100_000.0,
            "Expected >100k scheduling TPS");
    }

    #[test]
    fn test_benchmark_40k_mixed() {
        let config = BenchmarkConfig {
            num_transactions: 40_000,
            conflict_rate: 25,
            num_accounts: 80_000,
            warmup_txs: 0,
        };

        let result = benchmark_scheduling(&config);
        println!("{}", result);

        assert!(result.num_batches > 0);
    }

    #[test]
    fn run_progressive_benchmark() {
        benchmark_progressive();
    }
}
