//! Turbo Executor Benchmark - MAXIMUM PERFORMANCE TEST
//!
//! Tests the high-performance executor with:
//! - Batch signature verification
//! - Lock-free parallel execution
//! - Memory pools
//! - Native system program execution

use svm_runtime::hiperf::{TurboExecutor, TurboConfig, PipelinedExecutor, BlockPipelineConfig};
use svm_runtime::qmdb_state::InMemoryQMDBState;
use svm_runtime::executor::{ExecutionContext, ExecutorConfig};
use svm_runtime::types::{Account, Pubkey, Transaction, Message, MessageHeader, CompiledInstruction};
use ed25519_dalek::{SigningKey, Signer};
use sha2::{Sha256, Digest};
use std::sync::Arc;
use std::time::Instant;

/// Generate a deterministic keypair from a seed
fn keypair_from_seed(seed: u64) -> SigningKey {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[0..8].copy_from_slice(&seed.to_le_bytes());
    // Hash to get uniform distribution
    let hash: [u8; 32] = Sha256::digest(&seed_bytes).into();
    SigningKey::from_bytes(&hash)
}

/// Serialize message for signing (matches batch_verifier.rs)
fn serialize_message(msg: &Message) -> Vec<u8> {
    let mut data = Vec::with_capacity(256);
    // Header
    data.push(msg.header.num_required_signatures);
    data.push(msg.header.num_readonly_signed_accounts);
    data.push(msg.header.num_readonly_unsigned_accounts);
    // Account keys
    for key in &msg.account_keys {
        data.extend_from_slice(&key.0);
    }
    // Recent blockhash
    data.extend_from_slice(&msg.recent_blockhash);
    // Instructions
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
    // Generate keypair for the sender
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

    // Sign the serialized message
    let msg_bytes = serialize_message(&message);
    let msg_hash: [u8; 32] = Sha256::digest(&msg_bytes).into();
    let signature = keypair.sign(&msg_hash);

    Transaction {
        signatures: vec![signature.to_bytes()],
        message,
    }
}

fn run_turbo_test(size: usize, verify_signatures: bool) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ TURBO TEST: {} transactions (verify={})", size, verify_signatures);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Setup - create accounts with keypair-derived pubkeys
    print!("│ [1/4] Setting up {} accounts... ", size * 2);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();

    let state = Arc::new(InMemoryQMDBState::new());

    for i in 0..size {
        // Sender account (from keypair)
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

        // Receiver account (seed-based, matches make_transfer_tx)
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

/// Test with pre-verified transactions (simulating mempool pre-verification)
fn run_preverified_test(size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PRE-VERIFIED TEST: {} transactions", size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Setup
    print!("│ [1/4] Setting up {} accounts... ", size * 2);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();

    let state = Arc::new(InMemoryQMDBState::new());

    for i in 0..size {
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
    }
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    // Config with verification OFF (already verified in "mempool")
    let config = TurboConfig {
        max_batch_size: size,
        num_threads: 0,
        verify_signatures: false, // Pre-verified!
        speculative_execution: false,
        arena_pool_size: 32,
        parallel_batch_size: 1000,
    };

    let executor = TurboExecutor::new(state, config);
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    // Generate signed transactions (signatures valid but we skip verification)
    print!("│ [2/4] Generating {} signed transactions... ", size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let txs: Vec<Transaction> = (0..size)
        .map(|i| make_transfer_tx((i * 2) as u64, (i * 2 + 1) as u64))
        .collect();
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // Execute using pre-verified path
    print!("│ [3/4] Executing block (pre-verified)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let exec_start = Instant::now();
    let result = executor.execute_preverified_block(1, &txs, &ctx).unwrap();
    let exec_time = exec_start.elapsed();
    println!("{:.2}s", exec_time.as_secs_f64());

    println!("│ [4/4] Results computed");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ 📊 RESULTS:");
    println!("│    Transactions: {:>12}", size);
    println!("│    Successful:   {:>12}", result.successful);
    println!("│    Failed:       {:>12}", result.failed);
    println!("│    Total Time:   {:>12.2}s", result.timing.total_us as f64 / 1_000_000.0);
    println!("│    TPS:          {:>12.0}", result.timing.tps(size));
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ ⏱️  TIMING BREAKDOWN (verify=0, pre-verified!):");
    println!("│    Verify:       {:>12.2}ms ({:.1}%) ← ZERO!",
        result.timing.verify_us as f64 / 1000.0,
        if result.timing.total_us > 0 { result.timing.verify_us as f64 / result.timing.total_us as f64 * 100.0 } else { 0.0 });
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

/// Test pipelined block execution (overlapping multiple blocks)
fn run_pipelined_test(tx_per_block: usize, num_blocks: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ PIPELINED TEST: {} blocks x {} txs = {} total", num_blocks, tx_per_block, num_blocks * tx_per_block);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Setup accounts
    print!("│ [1/5] Setting up {} accounts... ", tx_per_block * num_blocks * 2);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let setup_start = Instant::now();

    let state = Arc::new(InMemoryQMDBState::new());
    let total_txs = tx_per_block * num_blocks;

    for i in 0..total_txs {
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
    }
    println!("{:.2}s", setup_start.elapsed().as_secs_f64());

    // Create pipelined executor
    let config = BlockPipelineConfig::default();
    let executor = PipelinedExecutor::new(state.clone(), config);

    // Generate all transactions for all blocks
    print!("│ [2/5] Generating {} transactions for {} blocks... ", total_txs, num_blocks);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();

    let mut blocks: Vec<Vec<Transaction>> = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let base_offset = block_idx * tx_per_block;
        let block_txs: Vec<Transaction> = (0..tx_per_block)
            .map(|i| {
                let idx = base_offset + i;
                make_transfer_tx((idx * 2) as u64, (idx * 2 + 1) as u64)
            })
            .collect();
        blocks.push(block_txs);
    }
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // Submit all blocks to the pipeline
    print!("│ [3/5] Submitting {} blocks to pipeline... ", num_blocks);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let submit_start = Instant::now();

    for (block_idx, block_txs) in blocks.into_iter().enumerate() {
        let ctx = ExecutionContext::new((block_idx + 1) as u64, 0, ExecutorConfig::default());
        executor.submit_block((block_idx + 1) as u64, block_txs, ctx).unwrap();
    }
    println!("{:.2}s", submit_start.elapsed().as_secs_f64());

    // Collect all results
    print!("│ [4/5] Collecting {} block results... ", num_blocks);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let collect_start = Instant::now();

    let mut total_successful = 0u64;
    let mut total_failed = 0u64;
    let mut results = Vec::with_capacity(num_blocks);

    for _ in 0..num_blocks {
        let result = executor.get_result().unwrap();
        total_successful += result.successful;
        total_failed += result.failed;
        results.push(result);
    }
    let total_time = collect_start.elapsed();
    println!("{:.2}s", total_time.as_secs_f64());

    // Calculate TPS
    let total_time_us = total_time.as_micros() as f64;
    let tps = if total_time_us > 0.0 {
        (total_txs as f64) * 1_000_000.0 / total_time_us
    } else {
        0.0
    };

    println!("│ [5/5] Results computed");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ 📊 PIPELINE RESULTS:");
    println!("│    Blocks:       {:>12}", num_blocks);
    println!("│    Txs/Block:    {:>12}", tx_per_block);
    println!("│    Total Txs:    {:>12}", total_txs);
    println!("│    Successful:   {:>12}", total_successful);
    println!("│    Failed:       {:>12}", total_failed);
    println!("│    Pipeline Time:{:>12.2}s", total_time.as_secs_f64());
    println!("│    Sustained TPS:{:>12.0} ← Pipeline Throughput!", tps);
    println!("│    Blocks/sec:   {:>12.1}", num_blocks as f64 / total_time.as_secs_f64());
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ 🔗 Block Merkle Roots:");
    for (i, result) in results.iter().take(3).enumerate() {
        println!("│    Block {}: {:x?}...", result.block_height, &result.merkle_root[..8]);
    }
    if results.len() > 3 {
        println!("│    ... ({} more blocks)", results.len() - 3);
    }
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    executor.shutdown();
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

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PHASE 3: PRE-VERIFIED MODE                         ");
    println!("              (Signatures verified in mempool - SKIP verify)           ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Pre-verified (mempool already verified signatures)
    run_preverified_test(10_000);
    run_preverified_test(50_000);
    run_preverified_test(100_000);

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PHASE 4: PIPELINED EXECUTION                        ");
    println!("               (Overlapping blocks for max throughput)                  ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Pipelined execution (overlapping multiple blocks)
    run_pipelined_test(10_000, 10);    // 10 blocks x 10k txs = 100k
    run_pipelined_test(10_000, 50);    // 50 blocks x 10k txs = 500k
    run_pipelined_test(20_000, 50);    // 50 blocks x 20k txs = 1M

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK COMPLETE                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
