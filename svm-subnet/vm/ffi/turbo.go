package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lsvm_runtime -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../../runtime

#include <stdint.h>
#include <stdlib.h>

// Forward declarations for Turbo executor
typedef struct TurboExecutorHandle TurboExecutorHandle;

// Turbo block result with timing breakdown
typedef struct {
	uint64_t successful;
	uint64_t failed;
	uint64_t verification_failures;
	uint64_t total_compute_units;
	uint64_t total_fees;
	uint64_t execution_time_us;
	uint8_t merkle_root[32];
	uint64_t block_height;
	// Timing breakdown
	uint64_t verify_us;
	uint64_t analyze_us;
	uint64_t schedule_us;
	uint64_t execute_us;
	uint64_t commit_us;
	uint64_t merkle_us;
} TurboBlockResultFFI;

// Turbo lifecycle
extern TurboExecutorHandle* turbo_executor_new(uint64_t num_threads, int verify_signatures);
extern void turbo_executor_free(TurboExecutorHandle* handle);

// Block context
extern int turbo_set_block_context(TurboExecutorHandle* handle, uint64_t slot, int64_t timestamp, const uint8_t* blockhash);

// Block execution - DELTA MODE (11M+ TPS)
extern int turbo_execute_block_delta(
	TurboExecutorHandle* handle,
	uint64_t block_height,
	const uint8_t** tx_data,
	const size_t* tx_lens,
	size_t tx_count,
	TurboBlockResultFFI** result_out
);

// Block execution - STANDARD MODE
extern int turbo_execute_block_standard(
	TurboExecutorHandle* handle,
	uint64_t block_height,
	const uint8_t** tx_data,
	const size_t* tx_lens,
	size_t tx_count,
	TurboBlockResultFFI** result_out
);

extern void turbo_free_result(TurboBlockResultFFI* result);

// State access
extern int turbo_get_merkle_root(TurboExecutorHandle* handle, uint8_t* root_out);

extern int turbo_load_account(
	TurboExecutorHandle* handle,
	const uint8_t* pubkey,
	uint64_t lamports,
	const uint8_t* data,
	size_t data_len,
	const uint8_t* owner,
	int executable
);

extern int turbo_get_account(
	TurboExecutorHandle* handle,
	const uint8_t* pubkey,
	uint64_t* lamports_out,
	uint8_t** data_out,
	size_t* data_len_out,
	uint8_t* owner_out,
	int* executable_out
);

// Statistics
extern int turbo_get_stats(
	TurboExecutorHandle* handle,
	uint64_t* total_txs,
	uint64_t* total_blocks,
	uint64_t* signatures_verified,
	uint64_t* num_threads
);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// TurboExecutor wraps the Rust high-performance executor
// This achieves 11M+ TPS using delta execution mode
type TurboExecutor struct {
	handle *C.TurboExecutorHandle
}

// TurboBlockResult contains the result of turbo block execution
type TurboBlockResult struct {
	Successful           uint64
	Failed               uint64
	VerificationFailures uint64
	TotalComputeUnits    uint64
	TotalFees            uint64
	ExecutionTimeUs      uint64
	MerkleRoot           [32]byte
	BlockHeight          uint64
	// Timing breakdown
	VerifyUs   uint64
	AnalyzeUs  uint64
	ScheduleUs uint64
	ExecuteUs  uint64
	CommitUs   uint64
	MerkleUs   uint64
}

// TurboStats contains executor statistics
type TurboStats struct {
	TotalTransactions  uint64
	TotalBlocks        uint64
	SignaturesVerified uint64
	NumThreads         uint64
}

// NewTurboExecutor creates a new high-performance executor
// numThreads: 0 = use all available CPUs
// verifySignatures: whether to verify Ed25519 signatures
func NewTurboExecutor(numThreads uint64, verifySignatures bool) (*TurboExecutor, error) {
	if numThreads == 0 {
		numThreads = uint64(runtime.NumCPU())
	}

	verify := C.int(0)
	if verifySignatures {
		verify = 1
	}

	handle := C.turbo_executor_new(C.uint64_t(numThreads), verify)
	if handle == nil {
		return nil, errors.New("failed to create Turbo executor")
	}

	return &TurboExecutor{handle: handle}, nil
}

// Close frees the executor resources
func (t *TurboExecutor) Close() {
	if t.handle != nil {
		C.turbo_executor_free(t.handle)
		t.handle = nil
	}
}

// SetBlockContext sets the current block context for execution
func (t *TurboExecutor) SetBlockContext(slot uint64, timestamp int64, blockhash []byte) error {
	if t.handle == nil {
		return errors.New("executor is closed")
	}

	if len(blockhash) != 32 {
		return errors.New("blockhash must be 32 bytes")
	}

	result := C.turbo_set_block_context(
		t.handle,
		C.uint64_t(slot),
		C.int64_t(timestamp),
		(*C.uint8_t)(unsafe.Pointer(&blockhash[0])),
	)

	if result != 0 {
		return errors.New("failed to set block context")
	}

	return nil
}

// ExecuteBlockDelta executes a block using delta mode - 11M+ TPS
// This is the FASTEST execution path for transfer transactions.
// It bypasses StateChangeSet and collects LamportDeltas directly,
// applying them in a single parallel pass at commit time.
func (t *TurboExecutor) ExecuteBlockDelta(blockHeight uint64, txDataList [][]byte) (*TurboBlockResult, error) {
	if t.handle == nil {
		return nil, errors.New("executor is closed")
	}

	if len(txDataList) == 0 {
		return &TurboBlockResult{
			BlockHeight: blockHeight,
		}, nil
	}

	// Prepare C arrays
	txCount := len(txDataList)
	txPtrs := make([]*C.uint8_t, txCount)
	txLens := make([]C.size_t, txCount)

	for i, txData := range txDataList {
		if len(txData) > 0 {
			txPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&txData[0]))
			txLens[i] = C.size_t(len(txData))
		}
	}

	var resultPtr *C.TurboBlockResultFFI

	status := C.turbo_execute_block_delta(
		t.handle,
		C.uint64_t(blockHeight),
		(**C.uint8_t)(unsafe.Pointer(&txPtrs[0])),
		(*C.size_t)(unsafe.Pointer(&txLens[0])),
		C.size_t(txCount),
		&resultPtr,
	)

	if status != 0 {
		return nil, errors.New("delta block execution failed")
	}

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}

	defer C.turbo_free_result(resultPtr)

	return convertTurboResult(resultPtr), nil
}

// ExecuteBlockStandard executes a block using standard mode (full pipeline)
// This includes signature verification, access analysis, scheduling, etc.
func (t *TurboExecutor) ExecuteBlockStandard(blockHeight uint64, txDataList [][]byte) (*TurboBlockResult, error) {
	if t.handle == nil {
		return nil, errors.New("executor is closed")
	}

	if len(txDataList) == 0 {
		return &TurboBlockResult{
			BlockHeight: blockHeight,
		}, nil
	}

	txCount := len(txDataList)
	txPtrs := make([]*C.uint8_t, txCount)
	txLens := make([]C.size_t, txCount)

	for i, txData := range txDataList {
		if len(txData) > 0 {
			txPtrs[i] = (*C.uint8_t)(unsafe.Pointer(&txData[0]))
			txLens[i] = C.size_t(len(txData))
		}
	}

	var resultPtr *C.TurboBlockResultFFI

	status := C.turbo_execute_block_standard(
		t.handle,
		C.uint64_t(blockHeight),
		(**C.uint8_t)(unsafe.Pointer(&txPtrs[0])),
		(*C.size_t)(unsafe.Pointer(&txLens[0])),
		C.size_t(txCount),
		&resultPtr,
	)

	if status != 0 {
		return nil, errors.New("standard block execution failed")
	}

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}

	defer C.turbo_free_result(resultPtr)

	return convertTurboResult(resultPtr), nil
}

// convertTurboResult converts C result to Go
func convertTurboResult(r *C.TurboBlockResultFFI) *TurboBlockResult {
	result := &TurboBlockResult{
		Successful:           uint64(r.successful),
		Failed:               uint64(r.failed),
		VerificationFailures: uint64(r.verification_failures),
		TotalComputeUnits:    uint64(r.total_compute_units),
		TotalFees:            uint64(r.total_fees),
		ExecutionTimeUs:      uint64(r.execution_time_us),
		BlockHeight:          uint64(r.block_height),
		VerifyUs:             uint64(r.verify_us),
		AnalyzeUs:            uint64(r.analyze_us),
		ScheduleUs:           uint64(r.schedule_us),
		ExecuteUs:            uint64(r.execute_us),
		CommitUs:             uint64(r.commit_us),
		MerkleUs:             uint64(r.merkle_us),
	}

	// Copy merkle root
	for i := 0; i < 32; i++ {
		result.MerkleRoot[i] = byte(r.merkle_root[i])
	}

	return result
}

// GetMerkleRoot returns the current state merkle root
func (t *TurboExecutor) GetMerkleRoot() ([32]byte, error) {
	if t.handle == nil {
		return [32]byte{}, errors.New("executor is closed")
	}

	var root [32]byte
	status := C.turbo_get_merkle_root(t.handle, (*C.uint8_t)(unsafe.Pointer(&root[0])))

	if status != 0 {
		return [32]byte{}, errors.New("failed to get merkle root")
	}

	return root, nil
}

// LoadAccount loads an account into the executor state
func (t *TurboExecutor) LoadAccount(pubkey [32]byte, lamports uint64, data []byte, owner [32]byte, executable bool) error {
	if t.handle == nil {
		return errors.New("executor is closed")
	}

	var dataPtr *C.uint8_t
	var dataLen C.size_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
		dataLen = C.size_t(len(data))
	}

	execFlag := C.int(0)
	if executable {
		execFlag = 1
	}

	status := C.turbo_load_account(
		t.handle,
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		C.uint64_t(lamports),
		dataPtr,
		dataLen,
		(*C.uint8_t)(unsafe.Pointer(&owner[0])),
		execFlag,
	)

	if status != 0 {
		return errors.New("failed to load account")
	}

	return nil
}

// GetAccount retrieves an account from the executor state
func (t *TurboExecutor) GetAccount(pubkey [32]byte) (*StateChange, error) {
	if t.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var lamports C.uint64_t
	var dataPtr *C.uint8_t
	var dataLen C.size_t
	var owner [32]byte
	var executable C.int

	status := C.turbo_get_account(
		t.handle,
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		&lamports,
		&dataPtr,
		&dataLen,
		(*C.uint8_t)(unsafe.Pointer(&owner[0])),
		&executable,
	)

	if status != 0 {
		return nil, nil // Account not found
	}

	account := &StateChange{
		Pubkey:     pubkey,
		Lamports:   uint64(lamports),
		Owner:      owner,
		Executable: executable != 0,
	}

	if dataPtr != nil && dataLen > 0 {
		account.Data = C.GoBytes(unsafe.Pointer(dataPtr), C.int(dataLen))
		C.free(unsafe.Pointer(dataPtr))
	}

	return account, nil
}

// GetStats returns executor statistics
func (t *TurboExecutor) GetStats() (*TurboStats, error) {
	if t.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var totalTxs, totalBlocks, sigsVerified, numThreads C.uint64_t

	status := C.turbo_get_stats(
		t.handle,
		&totalTxs,
		&totalBlocks,
		&sigsVerified,
		&numThreads,
	)

	if status != 0 {
		return nil, errors.New("failed to get stats")
	}

	return &TurboStats{
		TotalTransactions:  uint64(totalTxs),
		TotalBlocks:        uint64(totalBlocks),
		SignaturesVerified: uint64(sigsVerified),
		NumThreads:         uint64(numThreads),
	}, nil
}

// TPS calculates transactions per second from execution time
func (r *TurboBlockResult) TPS() float64 {
	if r.ExecutionTimeUs == 0 {
		return 0
	}
	return float64(r.Successful) * 1_000_000.0 / float64(r.ExecutionTimeUs)
}

// String returns a string representation of the turbo block result
func (r *TurboBlockResult) String() string {
	return fmt.Sprintf(
		"TurboBlockResult{height=%d, successful=%d, failed=%d, verify_fail=%d, compute=%d, fees=%d, time=%dus (%.0f TPS), merkle=%x}\n"+
			"  Timing: verify=%dus, analyze=%dus, schedule=%dus, execute=%dus, commit=%dus, merkle=%dus",
		r.BlockHeight, r.Successful, r.Failed, r.VerificationFailures,
		r.TotalComputeUnits, r.TotalFees, r.ExecutionTimeUs, r.TPS(), r.MerkleRoot[:8],
		r.VerifyUs, r.AnalyzeUs, r.ScheduleUs, r.ExecuteUs, r.CommitUs, r.MerkleUs,
	)
}
