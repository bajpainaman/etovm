//! GPU Benchmark - Tests CUDA-accelerated operations on H100
//!
//! Run with: cargo run --release --features cuda --bin gpu_bench
//!
//! Benchmarks:
//! 1. Batch SHA256 hashing (merkle tree)
//! 2. Batch Ed25519 verification
//! 3. Full merkle root computation

#[cfg(feature = "cuda")]
use svm_runtime::hiperf::{CudaExecutor, MultiGpuExecutor};

use sha2::{Sha256, Digest};
use std::time::Instant;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║            🚀 GPU BENCHMARK - H100 CUDA ACCELERATION 🚀              ║");
    println!("║              Testing CUDA-Accelerated SVM Operations                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA feature not enabled!");
        println!("   Build with: cargo run --release --features cuda --bin gpu_bench");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        run_gpu_benchmarks();
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_benchmarks() {
    // Check available GPUs
    println!("🔍 Detecting GPUs...");

    let executor = match CudaExecutor::new(0) {
        Ok(e) => {
            println!("   ✅ {}", e.device_info());
            e
        }
        Err(e) => {
            println!("   ❌ Failed to initialize CUDA: {}", e);
            return;
        }
    };

    println!();

    // =========================================================================
    // Benchmark 1: GPU SHA256 Batch Hashing
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    BENCHMARK 1: GPU SHA256 HASHING                     ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    for batch_size in [10_000, 100_000, 1_000_000, 10_000_000] {
        benchmark_sha256(&executor, batch_size);
    }

    // =========================================================================
    // Benchmark 2: GPU Merkle Root Computation
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                   BENCHMARK 2: GPU MERKLE ROOT                         ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    for leaf_count in [1_000, 10_000, 100_000, 1_000_000] {
        benchmark_merkle(&executor, leaf_count);
    }

    // =========================================================================
    // Benchmark 3: GPU vs CPU Comparison
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                   BENCHMARK 3: GPU vs CPU COMPARISON                   ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    compare_gpu_cpu(&executor, 1_000_000);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                       GPU BENCHMARK COMPLETE                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "cuda")]
fn benchmark_sha256(executor: &CudaExecutor, batch_size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ SHA256 Batch: {} hashes", batch_size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Generate test data (64 bytes each, like merkle pairs)
    print!("│ Generating test data... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let inputs: Vec<u8> = (0..batch_size)
        .flat_map(|i| {
            let mut data = [0u8; 64];
            data[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            data.to_vec()
        })
        .collect();
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // GPU execution
    print!("│ GPU hashing... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_start = Instant::now();
    let results = executor.batch_sha256(&inputs, 64).expect("GPU SHA256 failed");
    let gpu_time = gpu_start.elapsed();
    println!("{:.2}s", gpu_time.as_secs_f64());

    let hashes_per_sec = batch_size as f64 / gpu_time.as_secs_f64();
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ Results:       {} hashes", results.len());
    println!("│ GPU Time:      {:.2}ms", gpu_time.as_millis());
    println!("│ Throughput:    {:.0} hashes/sec", hashes_per_sec);
    println!("│ Throughput:    {:.2}M hashes/sec", hashes_per_sec / 1_000_000.0);
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}

#[cfg(feature = "cuda")]
fn benchmark_merkle(executor: &CudaExecutor, leaf_count: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Merkle Root: {} leaves", leaf_count);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Generate leaves
    print!("│ Generating {} leaves... ", leaf_count);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let leaves: Vec<[u8; 32]> = (0..leaf_count)
        .map(|i| {
            let mut leaf = [0u8; 32];
            leaf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            leaf
        })
        .collect();
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // GPU merkle computation
    print!("│ Computing merkle root on GPU... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_start = Instant::now();
    let root = executor.compute_merkle_root(&leaves).expect("GPU merkle failed");
    let gpu_time = gpu_start.elapsed();
    println!("{:.2}s", gpu_time.as_secs_f64());

    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ Merkle Root:   {:x?}...", &root[..8]);
    println!("│ GPU Time:      {:.2}ms", gpu_time.as_millis());
    println!("│ Leaves/sec:    {:.0}", leaf_count as f64 / gpu_time.as_secs_f64());
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}

#[cfg(feature = "cuda")]
fn compare_gpu_cpu(executor: &CudaExecutor, batch_size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ GPU vs CPU Comparison: {} SHA256 hashes", batch_size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Generate test data
    let inputs: Vec<[u8; 64]> = (0..batch_size)
        .map(|i| {
            let mut data = [0u8; 64];
            data[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            data
        })
        .collect();

    // CPU benchmark
    print!("│ CPU hashing (rayon parallel)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let cpu_start = Instant::now();
    use rayon::prelude::*;
    let _cpu_results: Vec<[u8; 32]> = inputs.par_iter()
        .map(|data| {
            let hash: [u8; 32] = Sha256::digest(data).into();
            hash
        })
        .collect();
    let cpu_time = cpu_start.elapsed();
    println!("{:.2}s", cpu_time.as_secs_f64());

    // GPU benchmark
    let flat_inputs: Vec<u8> = inputs.iter().flatten().copied().collect();
    print!("│ GPU hashing (CUDA)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_start = Instant::now();
    let _gpu_results = executor.batch_sha256(&flat_inputs, 64).expect("GPU failed");
    let gpu_time = gpu_start.elapsed();
    println!("{:.2}s", gpu_time.as_secs_f64());

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ CPU Time:      {:>12.2}ms", cpu_time.as_millis());
    println!("│ GPU Time:      {:>12.2}ms", gpu_time.as_millis());
    println!("│ Speedup:       {:>12.1}x ← GPU vs CPU", speedup);
    println!("│");
    println!("│ CPU Throughput: {:>10.2}M hashes/sec", batch_size as f64 / cpu_time.as_secs_f64() / 1_000_000.0);
    println!("│ GPU Throughput: {:>10.2}M hashes/sec", batch_size as f64 / gpu_time.as_secs_f64() / 1_000_000.0);
    println!("└─────────────────────────────────────────────────────────────────┘");
}
