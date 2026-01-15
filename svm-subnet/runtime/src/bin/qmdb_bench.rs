//! QMDB Benchmark - Compare Real QMDB vs HashMap State
//!
//! This benchmarks:
//! - Single account read/write
//! - Batch account operations
//! - Block commit with merkle root
//! - Parallel access patterns

use svm_runtime::{
    Account, Pubkey, RealQmdbState, InMemoryQMDBState,
};
use std::time::Instant;
use std::sync::Arc;

const WARMUP_OPS: usize = 1_000;

fn random_pubkey(seed: u64) -> Pubkey {
    let mut bytes = [0u8; 32];
    let seed_bytes = seed.to_le_bytes();
    for i in 0..32 {
        bytes[i] = seed_bytes[i % 8].wrapping_add(i as u8);
    }
    Pubkey(bytes)
}

fn random_account(seed: u64) -> Account {
    Account {
        lamports: seed * 1000,
        data: vec![seed as u8; 100], // 100 bytes of data
        owner: random_pubkey(seed + 1),
        executable: false,
        rent_epoch: 0,
    }
}

/// Benchmark HashMap-based state (baseline)
fn bench_hashmap_state(num_accounts: usize, num_reads: usize) -> (f64, f64, f64) {
    let state = InMemoryQMDBState::new();

    // Warmup
    for i in 0..WARMUP_OPS {
        let pk = random_pubkey(i as u64);
        let acc = random_account(i as u64);
        state.set_account(&pk, &acc).unwrap();
    }

    // Benchmark writes
    let start = Instant::now();
    for i in WARMUP_OPS..(WARMUP_OPS + num_accounts) {
        let pk = random_pubkey(i as u64);
        let acc = random_account(i as u64);
        state.set_account(&pk, &acc).unwrap();
    }
    let write_time = start.elapsed();
    let writes_per_sec = num_accounts as f64 / write_time.as_secs_f64();

    // Benchmark reads
    let start = Instant::now();
    for i in WARMUP_OPS..(WARMUP_OPS + num_reads) {
        let pk = random_pubkey(i as u64);
        let _ = state.get_account(&pk).unwrap();
    }
    let read_time = start.elapsed();
    let reads_per_sec = num_reads as f64 / read_time.as_secs_f64();

    // Benchmark block commit
    let start = Instant::now();
    let root = state.commit_block().unwrap();
    let commit_time = start.elapsed();
    let commits_per_sec = 1.0 / commit_time.as_secs_f64();

    println!("  Merkle root: {}", hex::encode(&root[..8]));

    (writes_per_sec, reads_per_sec, commits_per_sec)
}

/// Benchmark Real QMDB state
fn bench_real_qmdb(data_dir: &str, num_accounts: usize, num_reads: usize) -> (f64, f64, f64) {
    // Create fresh QMDB state
    let _ = std::fs::remove_dir_all(data_dir);
    let state = RealQmdbState::new(data_dir).expect("Failed to create QMDB");

    // Start block 1
    state.begin_block(1).expect("Failed to begin block");

    // Warmup
    for i in 0..WARMUP_OPS {
        let pk = random_pubkey(i as u64);
        let acc = random_account(i as u64);
        state.set_account(pk, acc).unwrap();
    }

    // Commit warmup block
    let _ = state.commit_block().unwrap();

    // Start block 2 for benchmarks
    state.begin_block(2).expect("Failed to begin block 2");

    // Benchmark writes
    let start = Instant::now();
    for i in WARMUP_OPS..(WARMUP_OPS + num_accounts) {
        let pk = random_pubkey(i as u64);
        let acc = random_account(i as u64);
        state.set_account(pk, acc).unwrap();
    }
    let write_time = start.elapsed();
    let writes_per_sec = num_accounts as f64 / write_time.as_secs_f64();

    // Benchmark reads
    let start = Instant::now();
    for i in WARMUP_OPS..(WARMUP_OPS + num_reads) {
        let pk = random_pubkey(i as u64);
        let _ = state.get_account(&pk).unwrap();
    }
    let read_time = start.elapsed();
    let reads_per_sec = num_reads as f64 / read_time.as_secs_f64();

    // Benchmark block commit
    let start = Instant::now();
    let root = state.commit_block().unwrap();
    let commit_time = start.elapsed();
    let commits_per_sec = 1.0 / commit_time.as_secs_f64();

    // Flush to disk
    state.flush().unwrap();

    println!("  Merkle root: {}", hex::encode(&root[..8]));

    (writes_per_sec, reads_per_sec, commits_per_sec)
}

/// Benchmark parallel reads with HashMap state
fn bench_parallel_reads(state: &Arc<InMemoryQMDBState>, num_threads: usize, ops_per_thread: usize) -> f64 {
    use std::thread;

    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads).map(|t| {
        let state = state.clone();
        thread::spawn(move || {
            for i in 0..ops_per_thread {
                let pk = random_pubkey((t * ops_per_thread + i) as u64);
                let _ = state.get_account(&pk);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = num_threads * ops_per_thread;
    total_ops as f64 / elapsed.as_secs_f64()
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           QMDB vs HashMap State Benchmark                 ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Comparing InMemoryQMDBState (HashMap) vs RealQmdbState   ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_sizes = [
        (10_000, "10K accounts"),
        (100_000, "100K accounts"),
        (500_000, "500K accounts"),
    ];

    for (num_accounts, label) in test_sizes.iter() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Benchmark: {} ", label);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // HashMap baseline
        println!("📦 HashMap State (baseline):");
        let (hash_writes, hash_reads, hash_commits) = bench_hashmap_state(*num_accounts, *num_accounts);
        println!("  Writes: {:.0}/sec", hash_writes);
        println!("  Reads:  {:.0}/sec", hash_reads);
        println!("  Commits: {:.2}/sec\n", hash_commits);

        // Real QMDB
        println!("🚀 Real QMDB State:");
        let data_dir = format!("/tmp/qmdb_bench_{}", num_accounts);
        let (qmdb_writes, qmdb_reads, qmdb_commits) = bench_real_qmdb(&data_dir, *num_accounts, *num_accounts);
        println!("  Writes: {:.0}/sec", qmdb_writes);
        println!("  Reads:  {:.0}/sec", qmdb_reads);
        println!("  Commits: {:.2}/sec\n", qmdb_commits);

        // Speedup comparison
        println!("📊 Speedup (QMDB vs HashMap):");
        println!("  Writes: {:.2}x", qmdb_writes / hash_writes);
        println!("  Reads:  {:.2}x", qmdb_reads / hash_reads);
        println!("  Commits: {:.2}x", qmdb_commits / hash_commits);

        // Cleanup
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // Parallel read benchmark
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Parallel Read Benchmark (HashMap - RwLock contention)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let state = Arc::new(InMemoryQMDBState::new());

    // Pre-populate state
    for i in 0..100_000u64 {
        let pk = random_pubkey(i);
        let acc = random_account(i);
        state.set_account(&pk, &acc).unwrap();
    }

    for num_threads in [1, 4, 8, 16, 32] {
        let ops_per_sec = bench_parallel_reads(&state, num_threads, 10_000);
        let expected_linear = bench_parallel_reads(&state, 1, 10_000) * num_threads as f64;
        let efficiency = ops_per_sec / expected_linear * 100.0;
        println!("  {} threads: {:.0}/sec (efficiency: {:.1}%)",
            num_threads, ops_per_sec, efficiency);
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Note: Real QMDB shines with persistent state and         ║");
    println!("║  concurrent access patterns. HashMap wins for pure        ║");
    println!("║  in-memory ops but loses on merkle proofs and durability. ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}
