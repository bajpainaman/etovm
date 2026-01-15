//! Turbo Executor Benchmark - MAXIMUM PERFORMANCE TEST
//!
//! Tests the high-performance executor with:
//! - Batch signature verification
//! - Lock-free parallel execution
//! - Memory pools
//! - Native system program execution

use svm_runtime::hiperf::{TurboExecutor, TurboConfig};
use svm_runtime::qmdb_state::InMemoryQMDBState;
use svm_runtime::executor::{ExecutionContext, ExecutorConfig};
use svm_runtime::types::{Account, Pubkey, Transaction, Message, MessageHeader, CompiledInstruction};
use std::sync::Arc;
use std::time::Instant;

fn make_transfer_tx(from_seed: u64, to_seed: u64) -> Transaction {
    let mut from = [0u8; 32];
    let mut to = [0u8; 32];
    from[0..8].copy_from_slice(&from_seed.to_le_bytes());
    to[0..8].copy_from_slice(&to_seed.to_le_bytes());

    let mut data = vec![2, 0, 0, 0]; // Transfer instruction
    data.extend_from_slice(&100u64.to_le_bytes());

    Transaction {
        signatures: vec![[0u8; 64]], // Mock signature for benchmarking
        message: Message {
            header: MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: 1,
            },
            account_keys: vec![Pubkey(from), Pubkey(to), Pubkey::system_program()],
            recent_blockhash: [0u8; 32],
            instructions: vec![CompiledInstruction {
                program_id_index: 2,
                accounts: vec![0, 1],
                data,
            }],
        },
    }
}

fn run_turbo_test(size: usize, verify_signatures: bool) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ TURBO TEST: {} transactions (verify={})", size, verify_signatures);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Setup
    print!("│ [1/4] Setting up {} accounts... ", size * 2);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();

    let state = Arc::new(InMemoryQMDBState::new());
    let num_accounts = size * 2;

    for i in 0..num_accounts {
        let mut pk = [0u8; 32];
        pk[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        state.set_account(&Pubkey(pk), &Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();
    }
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    // Create executor
    let config = TurboConfig {
        max_batch_size: size,
        num_threads: 0, // Auto
        verify_signatures,
        speculative_execution: false,
        arena_pool_size: 32,
        parallel_batch_size: 1000,
    };

    let executor = TurboExecutor::new(state, config);
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    // Generate transactions
    print!("│ [2/4] Generating {} transactions... ", size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let txs: Vec<Transaction> = (0..size)
        .map(|i| make_transfer_tx((i * 2) as u64, (i * 2 + 1) as u64))
        .collect();
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // Execute
    print!("│ [3/4] Executing block... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let exec_start = Instant::now();
    let result = executor.execute_block(1, &txs, &ctx).unwrap();
    let exec_time = exec_start.elapsed();
    println!("{:.2}s", exec_time.as_secs_f64());

    println!("│ [4/4] Results computed");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ 📊 RESULTS:");
    println!("│    Transactions: {:>12}", size);
    println!("│    Successful:   {:>12}", result.successful);
    println!("│    Failed:       {:>12}", result.failed);
    println!("│    Verify Fails: {:>12}", result.verification_failures);
    println!("│    Total Time:   {:>12.2}s", result.timing.total_us as f64 / 1_000_000.0);
    println!("│    TPS:          {:>12.0}", result.timing.tps(size));
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ ⏱️  TIMING BREAKDOWN:");
    println!("│    Verify:       {:>12.2}ms ({:.1}%)",
        result.timing.verify_us as f64 / 1000.0,
        result.timing.verify_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Analyze:      {:>12.2}ms ({:.1}%)",
        result.timing.analyze_us as f64 / 1000.0,
        result.timing.analyze_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Schedule:     {:>12.2}ms ({:.1}%)",
        result.timing.schedule_us as f64 / 1000.0,
        result.timing.schedule_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Execute:      {:>12.2}ms ({:.1}%)",
        result.timing.execute_us as f64 / 1000.0,
        result.timing.execute_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Commit:       {:>12.2}ms ({:.1}%)",
        result.timing.commit_us as f64 / 1000.0,
        result.timing.commit_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Merkle:       {:>12.2}ms ({:.1}%)",
        result.timing.merkle_us as f64 / 1000.0,
        result.timing.merkle_us as f64 / result.timing.total_us as f64 * 100.0);
    println!("│    Merkle Root:  {:x?}...", &result.merkle_root[..8]);
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║        🚀🚀🚀 TURBO EXECUTOR - MAXIMUM PERFORMANCE 🚀🚀🚀           ║");
    println!("║              High-Performance SVM Execution Engine                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("System: {} CPU cores", rayon::current_num_threads());
    println!("Target: 200k+ TPS sustained");
    println!();

    // Warmup (no verification)
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PHASE 1: RAW EXECUTION SPEED                       ");
    println!("                    (Signature verification OFF)                       ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    run_turbo_test(100_000, false);
    run_turbo_test(500_000, false);
    run_turbo_test(1_000_000, false);
    run_turbo_test(2_000_000, false);
    run_turbo_test(5_000_000, false);

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PHASE 2: REALISTIC WORKLOAD                        ");
    println!("                    (Signature verification ON)                        ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // With verification (more realistic)
    run_turbo_test(10_000, true);
    run_turbo_test(50_000, true);
    run_turbo_test(100_000, true);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK COMPLETE                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
