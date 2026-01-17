//! Turbo Executor Benchmark - MAXIMUM PERFORMANCE TEST
//!
//! Optimized benchmark focusing on fastest paths:
//! - Delta-based execution (zero-copy transfers)
//! - Pipelined block execution

use svm_runtime::hiperf::{TurboExecutor, TurboConfig, PipelinedExecutor, BlockPipelineConfig};
use svm_runtime::qmdb_state::InMemoryQMDBState;
use svm_runtime::executor::{ExecutionContext, ExecutorConfig};
use svm_runtime::types::{Account, Pubkey, Transaction, Message, MessageHeader, CompiledInstruction};
use ed25519_dalek::{SigningKey, Signer};
use sha2::{Sha256, Digest};
use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;

/// Generate a deterministic keypair from a seed
fn keypair_from_seed(seed: u64) -> SigningKey {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[0..8].copy_from_slice(&seed.to_le_bytes());
    let hash: [u8; 32] = Sha256::digest(&seed_bytes).into();
    SigningKey::from_bytes(&hash)
}

/// Serialize message for signing
fn serialize_message(msg: &Message) -> Vec<u8> {
    let mut data = Vec::with_capacity(256);
    data.push(msg.header.num_required_signatures);
    data.push(msg.header.num_readonly_signed_accounts);
    data.push(msg.header.num_readonly_unsigned_accounts);
    for key in &msg.account_keys {
        data.extend_from_slice(&key.0);
    }
    data.extend_from_slice(&msg.recent_blockhash);
    for ix in &msg.instructions {
        data.push(ix.program_id_index);
        data.push(ix.accounts.len() as u8);
        data.extend_from_slice(&ix.accounts);
        data.extend_from_slice(&(ix.data.len() as u16).to_le_bytes());
        data.extend_from_slice(&ix.data);
    }
    data
}

fn make_transfer_tx(from_seed: u64, to_seed: u64) -> Transaction {
    let keypair = keypair_from_seed(from_seed);
    let from = keypair.verifying_key().to_bytes();

    let mut to = [0u8; 32];
    to[0..8].copy_from_slice(&to_seed.to_le_bytes());

    let mut data = vec![2, 0, 0, 0]; // Transfer instruction
    data.extend_from_slice(&100u64.to_le_bytes());

    let message = Message {
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
    };

    let msg_bytes = serialize_message(&message);
    let msg_hash: [u8; 32] = Sha256::digest(&msg_bytes).into();
    let signature = keypair.sign(&msg_hash);

    Transaction {
        signatures: vec![signature.to_bytes()],
        message,
    }
}

/// Setup state and executor for benchmarks
fn setup_benchmark(size: usize) -> (Arc<InMemoryQMDBState>, Vec<Transaction>) {
    let state = Arc::new(InMemoryQMDBState::new());

    // PARALLEL account setup
    (0..size).into_par_iter().for_each(|i| {
        let from_seed = (i * 2) as u64;
        let keypair = keypair_from_seed(from_seed);
        let from_pk = Pubkey(keypair.verifying_key().to_bytes());
        state.set_account(&from_pk, &Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();

        let to_seed = (i * 2 + 1) as u64;
        let mut to_pk = [0u8; 32];
        to_pk[0..8].copy_from_slice(&to_seed.to_le_bytes());
        state.set_account(&Pubkey(to_pk), &Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();
    });

    // PARALLEL tx generation
    let txs: Vec<Transaction> = (0..size)
        .into_par_iter()
        .map(|i| make_transfer_tx((i * 2) as u64, (i * 2 + 1) as u64))
        .collect();

    (state, txs)
}

/// DELTA BENCHMARK - Zero-copy transfer execution
fn run_delta_test(size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ DELTA TEST: {} transactions (zero-copy)", size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    print!("│ [1/3] Setup & generation... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();
    let (state, txs) = setup_benchmark(size);
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    let config = TurboConfig {
        max_batch_size: size,
        num_threads: 0,
        verify_signatures: false,
        speculative_execution: false,
        arena_pool_size: 32,
        parallel_batch_size: 1000,
    };

    let executor = TurboExecutor::new(state, config);
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    print!("│ [2/3] Executing (delta mode)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let exec_start = Instant::now();
    let result = executor.execute_block_delta(1, &txs, &ctx).unwrap();
    println!("{:.2}s", exec_start.elapsed().as_secs_f64());

    println!("│ [3/3] Results computed");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ DELTA RESULTS:");
    println!("│    Transactions: {:>12}", size);
    println!("│    Successful:   {:>12}", result.successful);
    println!("│    Failed:       {:>12}", result.failed);
    println!("│    Total Time:   {:>12.2}s", result.timing.total_us as f64 / 1_000_000.0);
    println!("│    TPS:          {:>12.0}", result.timing.tps(size));
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ TIMING (Delta skips verify/analyze/schedule):");
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

/// STANDARD TURBO TEST - For comparison
fn run_turbo_test(size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ STANDARD TEST: {} transactions", size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    print!("│ [1/3] Setup & generation... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();
    let (state, txs) = setup_benchmark(size);
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    let config = TurboConfig {
        max_batch_size: size,
        num_threads: 0,
        verify_signatures: false,
        speculative_execution: false,
        arena_pool_size: 32,
        parallel_batch_size: 1000,
    };

    let executor = TurboExecutor::new(state, config);
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    print!("│ [2/3] Executing (standard)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let exec_start = Instant::now();
    let result = executor.execute_block(1, &txs, &ctx).unwrap();
    println!("{:.2}s", exec_start.elapsed().as_secs_f64());

    println!("│ [3/3] Results computed");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ STANDARD RESULTS:");
    println!("│    Transactions: {:>12}", size);
    println!("│    Successful:   {:>12}", result.successful);
    println!("│    Failed:       {:>12}", result.failed);
    println!("│    Total Time:   {:>12.2}s", result.timing.total_us as f64 / 1_000_000.0);
    println!("│    TPS:          {:>12.0}", result.timing.tps(size));
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ TIMING BREAKDOWN:");
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

/// PIPELINED TEST - Max throughput with overlapping blocks
fn run_pipelined_test(tx_per_block: usize, num_blocks: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PIPELINE TEST: {} blocks x {} txs = {} total", num_blocks, tx_per_block, num_blocks * tx_per_block);
    println!("├─────────────────────────────────────────────────────────────────┤");

    print!("│ [1/4] Setup accounts... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();

    let state = Arc::new(InMemoryQMDBState::new());
    let total_txs = tx_per_block * num_blocks;

    (0..total_txs).into_par_iter().for_each(|i| {
        let from_seed = (i * 2) as u64;
        let keypair = keypair_from_seed(from_seed);
        let from_pk = Pubkey(keypair.verifying_key().to_bytes());
        state.set_account(&from_pk, &Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();

        let to_seed = (i * 2 + 1) as u64;
        let mut to_pk = [0u8; 32];
        to_pk[0..8].copy_from_slice(&to_seed.to_le_bytes());
        state.set_account(&Pubkey(to_pk), &Account {
            lamports: 1_000_000_000,
            data: vec![],
            owner: Pubkey::system_program(),
            executable: false,
            rent_epoch: 0,
        }).unwrap();
    });
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    let config = BlockPipelineConfig::default();
    let executor = PipelinedExecutor::new(state.clone(), config);

    print!("│ [2/4] Generating {} txs... ", total_txs);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();

    let blocks: Vec<Vec<Transaction>> = (0..num_blocks)
        .into_par_iter()
        .map(|block_idx| {
            let base_offset = block_idx * tx_per_block;
            (0..tx_per_block)
                .into_par_iter()
                .map(|i| {
                    let idx = base_offset + i;
                    make_transfer_tx((idx * 2) as u64, (idx * 2 + 1) as u64)
                })
                .collect()
        })
        .collect();
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    print!("│ [3/4] Submitting blocks... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let submit_start = Instant::now();

    for (block_idx, block_txs) in blocks.into_iter().enumerate() {
        let ctx = ExecutionContext::new((block_idx + 1) as u64, 0, ExecutorConfig::default());
        executor.submit_block((block_idx + 1) as u64, block_txs, ctx).unwrap();
    }
    println!("{:.2}s", submit_start.elapsed().as_secs_f64());

    print!("│ [4/4] Collecting results... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let collect_start = Instant::now();

    let mut total_successful = 0u64;
    let mut total_failed = 0u64;

    for _ in 0..num_blocks {
        let result = executor.get_result().unwrap();
        total_successful += result.successful;
        total_failed += result.failed;
    }
    let total_time = collect_start.elapsed();
    println!("{:.2}s", total_time.as_secs_f64());

    let tps = (total_txs as f64) * 1_000_000.0 / total_time.as_micros() as f64;

    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ PIPELINE RESULTS:");
    println!("│    Total Txs:    {:>12}", total_txs);
    println!("│    Successful:   {:>12}", total_successful);
    println!("│    Failed:       {:>12}", total_failed);
    println!("│    Pipeline Time:{:>12.2}s", total_time.as_secs_f64());
    println!("│    SUSTAINED TPS:{:>12.0}", tps);
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    executor.shutdown();
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           TURBO EXECUTOR - OPTIMIZED BENCHMARK                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("System: {} CPU cores", rayon::current_num_threads());
    println!();

    // === PHASE 1: Delta vs Standard comparison ===
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("           PHASE 1: DELTA vs STANDARD (1M transactions)               ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    run_turbo_test(1_000_000);
    run_delta_test(1_000_000);

    // === PHASE 2: Delta scaling ===
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("           PHASE 2: DELTA SCALING TEST                                ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    run_delta_test(2_000_000);
    run_delta_test(5_000_000);

    // === PHASE 3: Pipelined execution ===
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("           PHASE 3: PIPELINED EXECUTION                               ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    run_pipelined_test(50_000, 20);   // 1M total
    run_pipelined_test(100_000, 20);  // 2M total

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
