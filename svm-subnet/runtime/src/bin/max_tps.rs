//! Maximum TPS Test - 10 MILLION TX EDITION
//!
//! ABSOLUTE LIMIT TEST

use svm_runtime::sealevel::{QMDBParallelExecutor, QMDBExecutorConfig};
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

    let mut data = vec![2, 0, 0, 0];
    data.extend_from_slice(&100u64.to_le_bytes());

    Transaction {
        signatures: vec![[0u8; 64]],
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

fn run_test(size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Testing {} transactions...", size);
    println!("├─────────────────────────────────────────────────────────────────┤");
    
    // Setup phase
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
    let setup_time = setup_start.elapsed();
    println!("{:.2}s", setup_time.as_secs_f64());

    // Config
    let config = QMDBExecutorConfig {
        max_batch_size: size,
        num_threads: 0,
        speculative_execution: false,
        executor_config: ExecutorConfig::default(),
    };

    let executor = QMDBParallelExecutor::new(state, config);
    let mut ctx = ExecutionContext::new(1, 0, ExecutorConfig::default());
    ctx.add_blockhash([0u8; 32]);

    // Generate transactions
    print!("│ [2/4] Generating {} transactions... ", size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let txs: Vec<Transaction> = (0..size).map(|i| {
        make_transfer_tx((i * 2) as u64, (i * 2 + 1) as u64)
    }).collect();
    let gen_time = gen_start.elapsed();
    println!("{:.2}s", gen_time.as_secs_f64());

    // Execute
    print!("│ [3/4] Executing block... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let exec_start = Instant::now();
    let result = executor.execute_block(1, &txs, &ctx).unwrap();
    let exec_time = exec_start.elapsed();
    println!("{:.2}s", exec_time.as_secs_f64());

    // Compute TPS
    let tps = (size as f64) / exec_time.as_secs_f64();
    
    println!("│ [4/4] Computing merkle root... done");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ 📊 RESULTS:");
    println!("│    Transactions: {:>12}", size);
    println!("│    Successful:   {:>12}", result.successful);
    println!("│    Failed:       {:>12}", result.failed);
    println!("│    Exec Time:    {:>12.2}s", exec_time.as_secs_f64());
    println!("│    TPS:          {:>12.0}", tps);
    println!("│    Merkle Root:  {:x?}...", &result.merkle_root[..8]);
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║            🔥🔥🔥 10 MILLION TRANSACTION CHALLENGE 🔥🔥🔥            ║");
    println!("║                  QMDB Parallel Executor - BEAST MODE                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("System: {} CPU cores", rayon::current_num_threads());
    println!("Target: 10,000,000 transactions in a single block");
    println!();

    // Warmup
    run_test(100_000);
    
    // Scale up
    run_test(1_000_000);
    run_test(2_000_000);
    run_test(5_000_000);
    run_test(10_000_000);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            CHALLENGE COMPLETE                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
