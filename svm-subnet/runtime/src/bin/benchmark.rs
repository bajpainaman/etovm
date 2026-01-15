//! Sealevel TPS Benchmark Runner
//!
//! Run with: cargo run --release --bin benchmark

use svm_runtime::sealevel::benchmark::{
    benchmark_progressive, benchmark_scheduling, BenchmarkConfig,
    benchmark_qmdb_progressive, benchmark_qmdb_execution, benchmark_comparison,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            SEALEVEL PARALLEL EXECUTOR BENCHMARK              ║");
    println!("║                    Target: 40,000 TPS                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Run progressive benchmark
    benchmark_progressive();

    println!("\n");

    // Run specific high-load tests
    println!("🔥 HIGH-LOAD STRESS TESTS\n");

    // Test 1: 40k transactions, 0% conflicts (best case)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: 40,000 txs, 0% conflict rate (best case)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let config = BenchmarkConfig {
        num_transactions: 40_000,
        conflict_rate: 0,
        num_accounts: 100_000,
        warmup_txs: 0,
    };
    let result = benchmark_scheduling(&config);
    println!("{}", result);

    // Test 2: 40k transactions, 10% conflicts (realistic)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: 40,000 txs, 10% conflict rate (realistic)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let config = BenchmarkConfig {
        num_transactions: 40_000,
        conflict_rate: 10,
        num_accounts: 100_000,
        warmup_txs: 0,
    };
    let result = benchmark_scheduling(&config);
    println!("{}", result);

    // Test 3: 40k transactions, 25% conflicts (moderate)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: 40,000 txs, 25% conflict rate (moderate)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let config = BenchmarkConfig {
        num_transactions: 40_000,
        conflict_rate: 25,
        num_accounts: 100_000,
        warmup_txs: 0,
    };
    let result = benchmark_scheduling(&config);
    println!("{}", result);

    // Test 4: 40k transactions, 50% conflicts (worst realistic)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: 40,000 txs, 50% conflict rate (high contention)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let config = BenchmarkConfig {
        num_transactions: 40_000,
        conflict_rate: 50,
        num_accounts: 100_000,
        warmup_txs: 0,
    };
    let result = benchmark_scheduling(&config);
    println!("{}", result);

    // Summary for scheduling
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    SCHEDULING SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Scheduling TPS measures how fast we can batch transactions  ║");
    println!("║  Actual TPS = Scheduling TPS × Execution efficiency          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // QMDB Benchmarks
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              QMDB PARALLEL EXECUTOR BENCHMARK                ║");
    println!("║            (With Block Batching & State Commits)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Run QMDB progressive benchmark
    benchmark_qmdb_progressive();

    // Run comparison benchmark
    benchmark_comparison();

    // Final summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      FINAL SUMMARY                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  QMDB Integration Status: ✓ Complete                         ║");
    println!("║                                                              ║");
    println!("║  Features:                                                   ║");
    println!("║  - Block-level state batching                                ║");
    println!("║  - Atomic commit with Merkle root                            ║");
    println!("║  - Parallel transaction execution                            ║");
    println!("║  - O(1) I/O per update (when using full QMDB)                ║");
    println!("║                                                              ║");
    println!("║  Next Steps for Production:                                  ║");
    println!("║  - Enable full QMDB with SSD storage                         ║");
    println!("║  - Configure Prefetcher-Updater-Flusher pipeline             ║");
    println!("║  - Tune shard count and twig size                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
