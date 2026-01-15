package test

import (
	"math/big"
	"testing"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

func TestHybridExecutorCreation(t *testing.T) {
	executor, err := ffi.NewHybridExecutor(43114)
	if err != nil {
		t.Fatalf("Failed to create hybrid executor: %v", err)
	}
	defer executor.Close()

	t.Log("Hybrid executor created successfully")
}

func TestHybridExecutorAccountOperations(t *testing.T) {
	executor, err := ffi.NewHybridExecutor(43114)
	if err != nil {
		t.Fatalf("Failed to create hybrid executor: %v", err)
	}
	defer executor.Close()

	// Test loading an account
	pubkey := [32]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	owner := [32]byte{} // System program
	lamports := uint64(1000000000) // 1 SOL
	data := []byte("test data")

	err = executor.LoadAccount(pubkey, lamports, data, owner, false)
	if err != nil {
		t.Fatalf("Failed to load account: %v", err)
	}

	t.Log("Account loaded successfully")

	// Test retrieving the account
	account, err := executor.GetAccount(pubkey)
	if err != nil {
		t.Fatalf("Failed to get account: %v", err)
	}

	if account.Lamports != lamports {
		t.Errorf("Expected lamports %d, got %d", lamports, account.Lamports)
	}

	t.Logf("Account retrieved: lamports=%d, data_len=%d", account.Lamports, len(account.Data))
}

func TestHybridExecutorBPFExecution(t *testing.T) {
	executor, err := ffi.NewHybridExecutor(43114)
	if err != nil {
		t.Fatalf("Failed to create hybrid executor: %v", err)
	}
	defer executor.Close()

	// Set block context
	err = executor.SetBlock(1, 1704067200)
	if err != nil {
		t.Fatalf("Failed to set block: %v", err)
	}

	// Simple BPF program: mov r0, 42; exit
	bytecode := []byte{
		0xb7, 0x00, 0x00, 0x00, 42, 0, 0, 0, // mov64 r0, 42
		0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,  // exit
	}

	programID := [32]byte{1, 2, 3}
	inputData := []byte{}

	result, err := executor.ExecuteBPF(programID, bytecode, inputData)
	if err != nil {
		t.Fatalf("BPF execution failed: %v", err)
	}

	if !result.Success {
		t.Errorf("BPF execution was not successful: %s", result.Error)
	}

	t.Logf("BPF execution result: success=%v, compute_used=%d", result.Success, result.ComputeUnitsUsed)
}

func TestHybridExecutorEVMExecution(t *testing.T) {
	executor, err := ffi.NewHybridExecutor(43114)
	if err != nil {
		t.Fatalf("Failed to create hybrid executor: %v", err)
	}
	defer executor.Close()

	// Set block context
	err = executor.SetBlock(1, 1704067200)
	if err != nil {
		t.Fatalf("Failed to set block: %v", err)
	}

	// Simple EVM call (no contract, just testing the infrastructure)
	caller := [20]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14}
	to := [20]byte{0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34}
	value := big.NewInt(0)
	data := []byte{}
	gasLimit := uint64(21000)

	result, err := executor.ExecuteEVM(caller, &to, value, data, gasLimit)
	// EVM execution may fail due to missing accounts, but it should not panic
	if err != nil {
		t.Logf("EVM execution returned error (expected for empty state): %v", err)
	} else {
		t.Logf("EVM execution result: success=%v, gas_used=%d", result.Success, result.ComputeUnitsUsed)
	}
}

func TestHybridExecutorSetBlock(t *testing.T) {
	executor, err := ffi.NewHybridExecutor(43114)
	if err != nil {
		t.Fatalf("Failed to create hybrid executor: %v", err)
	}
	defer executor.Close()

	// Test setting block context
	slot := uint64(12345)
	timestamp := int64(1704067200)

	err = executor.SetBlock(slot, timestamp)
	if err != nil {
		t.Fatalf("Failed to set block: %v", err)
	}

	t.Logf("Block set successfully: slot=%d, timestamp=%d", slot, timestamp)
}
