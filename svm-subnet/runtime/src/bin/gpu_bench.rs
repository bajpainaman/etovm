//! GPU Benchmark - Tests CUDA-accelerated operations on H100
//!
//! Run with: cargo run --release --features cuda --bin gpu_bench
//!
//! Benchmarks:
//! 1. Batch SHA256 hashing (merkle tree)
//! 2. Batch Ed25519 verification
//! 3. Full merkle root computation

#[cfg(feature = "cuda")]
use svm_runtime::hiperf::{CudaExecutor, GpuEd25519Verifier};

use sha2::{Sha256, Digest};
use std::time::Instant;
use ed25519_dalek::{SigningKey, Signer, VerifyingKey, Signature, Verifier};

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

    // =========================================================================
    // Benchmark 4: GPU Ed25519 Signature Verification
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                   BENCHMARK 4: GPU Ed25519 VERIFICATION                ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    for batch_size in [1_000, 10_000, 100_000] {
        benchmark_ed25519_gpu(batch_size);
    }

    // Compare GPU vs CPU Ed25519
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                   BENCHMARK 5: GPU vs CPU Ed25519                      ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    compare_ed25519_gpu_cpu(50_000);

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

/// Generate valid Ed25519 signatures for benchmarking
fn generate_ed25519_batch(count: usize) -> (Vec<[u8; 32]>, Vec<[u8; 64]>, Vec<[u8; 32]>) {
    use rand::rngs::OsRng;

    let mut messages = Vec::with_capacity(count);
    let mut signatures = Vec::with_capacity(count);
    let mut pubkeys = Vec::with_capacity(count);

    for i in 0..count {
        // Generate keypair
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Create message (32 bytes - typical transaction hash)
        let mut msg = [0u8; 32];
        msg[0..8].copy_from_slice(&(i as u64).to_le_bytes());

        // Sign
        let sig = signing_key.sign(&msg);

        messages.push(msg);
        signatures.push(sig.to_bytes());
        pubkeys.push(verifying_key.to_bytes());
    }

    (messages, signatures, pubkeys)
}

#[cfg(feature = "cuda")]
fn benchmark_ed25519_gpu(batch_size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Ed25519 GPU Verification: {} signatures", batch_size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Generate valid signatures
    print!("│ Generating {} valid signatures... ", batch_size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gen_start = Instant::now();
    let (messages, signatures, pubkeys) = generate_ed25519_batch(batch_size);
    println!("{:.2}s", gen_start.elapsed().as_secs_f64());

    // Initialize GPU verifier
    print!("│ Initializing GPU Ed25519 verifier... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let init_start = Instant::now();
    let verifier = match GpuEd25519Verifier::new(0) {
        Ok(v) => {
            println!("{:.2}s", init_start.elapsed().as_secs_f64());
            v
        }
        Err(e) => {
            println!("FAILED: {}", e);
            return;
        }
    };

    // GPU verification
    print!("│ GPU verifying {} signatures... ", batch_size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_start = Instant::now();
    let results = verifier.batch_verify(&messages, &signatures, &pubkeys)
        .expect("GPU verification failed");
    let gpu_time = gpu_start.elapsed();
    println!("{:.2}s", gpu_time.as_secs_f64());

    let valid_count = results.iter().filter(|&&v| v).count();
    let verify_rate = batch_size as f64 / gpu_time.as_secs_f64();

    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ Valid:         {}/{} signatures", valid_count, batch_size);
    println!("│ GPU Time:      {:.2}ms", gpu_time.as_millis());
    println!("│ Throughput:    {:.0} verifications/sec", verify_rate);
    println!("│ Throughput:    {:.2}M verifications/sec", verify_rate / 1_000_000.0);

    // Calculate TPS impact (50% of time is verification)
    let tps_capacity = verify_rate;
    println!("│ TPS Capacity:  {:.2}M TPS (verify only)", tps_capacity / 1_000_000.0);
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
}

#[cfg(feature = "cuda")]
fn compare_ed25519_gpu_cpu(batch_size: usize) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ GPU vs CPU Ed25519 Verification: {} signatures", batch_size);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Generate valid signatures
    print!("│ Generating {} valid signatures... ", batch_size);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let (messages, signatures, pubkeys) = generate_ed25519_batch(batch_size);
    println!("done");

    // CPU verification (parallel with rayon)
    print!("│ CPU verifying (rayon parallel)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let cpu_start = Instant::now();
    use rayon::prelude::*;
    let _cpu_results: Vec<bool> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let verifying_key = VerifyingKey::from_bytes(&pubkeys[i]).ok();
            let sig = Signature::from_bytes(&signatures[i]);
            if let Some(vk) = verifying_key {
                vk.verify(&messages[i], &sig).is_ok()
            } else {
                false
            }
        })
        .collect();
    let cpu_time = cpu_start.elapsed();
    println!("{:.2}s", cpu_time.as_secs_f64());

    // GPU verification
    print!("│ GPU verifying (CUDA)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    let gpu_start = Instant::now();
    let verifier = GpuEd25519Verifier::new(0).expect("Failed to init GPU");
    let _gpu_results = verifier.batch_verify(&messages, &signatures, &pubkeys)
        .expect("GPU verification failed");
    let gpu_time = gpu_start.elapsed();
    println!("{:.2}s", gpu_time.as_secs_f64());

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│ CPU Time:      {:>12.2}ms ({} cores)", cpu_time.as_millis(), num_cpus::get());
    println!("│ GPU Time:      {:>12.2}ms (H100 NVL)", gpu_time.as_millis());
    println!("│ Speedup:       {:>12.1}x ← GPU vs CPU", speedup);
    println!("│");
    println!("│ CPU Throughput: {:>10.2}K verifications/sec", batch_size as f64 / cpu_time.as_secs_f64() / 1_000.0);
    println!("│ GPU Throughput: {:>10.2}K verifications/sec", batch_size as f64 / gpu_time.as_secs_f64() / 1_000.0);
    println!("│");

    let cpu_tps = batch_size as f64 / cpu_time.as_secs_f64();
    let gpu_tps = batch_size as f64 / gpu_time.as_secs_f64();
    println!("│ CPU Max TPS:   {:>10.2}K TPS (verify-bound)", cpu_tps / 1_000.0);
    println!("│ GPU Max TPS:   {:>10.2}K TPS (verify-bound)", gpu_tps / 1_000.0);
    println!("└─────────────────────────────────────────────────────────────────┘");
}
