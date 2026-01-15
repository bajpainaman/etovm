package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lsvm_runtime -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../../runtime

#include <stdint.h>
#include <stdlib.h>

// Forward declarations for QMDB executor
typedef struct QMDBExecutorHandle QMDBExecutorHandle;

// Reuse ExecutionResultFFI from runtime.go
typedef struct {
	int success;
	uint64_t compute_units_used;
	uint64_t fee;
	char* error_msg;
	char** logs;
	size_t logs_len;
	void* state_changes;
	size_t state_changes_len;
} QMDBExecResultFFI;

typedef struct {
	uint64_t successful;
	uint64_t failed;
	uint64_t total_compute_units;
	uint64_t total_fees;
	uint64_t execution_time_us;
	uint8_t merkle_root[32];
	uint64_t block_height;
	QMDBExecResultFFI* results;
	size_t results_len;
} BlockResultFFI;

// QMDB lifecycle
extern QMDBExecutorHandle* qmdb_executor_new(uint64_t num_threads);
extern void qmdb_executor_free(QMDBExecutorHandle* handle);

// Block context
extern int qmdb_set_block_context(QMDBExecutorHandle* handle, uint64_t slot, int64_t timestamp, const uint8_t* blockhash);

// Block execution
extern int qmdb_execute_block(
	QMDBExecutorHandle* handle,
	uint64_t block_height,
	const uint8_t** tx_data,
	const size_t* tx_lens,
	size_t tx_count,
	BlockResultFFI** result_out
);
extern void qmdb_free_block_result(BlockResultFFI* result);

// State access
extern int qmdb_get_merkle_root(QMDBExecutorHandle* handle, uint8_t* root_out);

extern int qmdb_load_account(
	QMDBExecutorHandle* handle,
	const uint8_t* pubkey,
	uint64_t lamports,
	const uint8_t* data,
	size_t data_len,
	const uint8_t* owner,
	int executable
);

extern int qmdb_get_account(
	QMDBExecutorHandle* handle,
	const uint8_t* pubkey,
	uint64_t* lamports_out,
	uint8_t** data_out,
	size_t* data_len_out,
	uint8_t* owner_out,
	int* executable_out
);

// Statistics
extern int qmdb_get_stats(
	QMDBExecutorHandle* handle,
	uint64_t* total_txs,
	uint64_t* total_blocks,
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

// QMDBExecutor wraps the Rust QMDB parallel executor
type QMDBExecutor struct {
	handle *C.QMDBExecutorHandle
}

// BlockResult contains the result of block execution
type BlockResult struct {
	Successful        uint64
	Failed            uint64
	TotalComputeUnits uint64
	TotalFees         uint64
	ExecutionTimeUs   uint64
	MerkleRoot        [32]byte
	BlockHeight       uint64
	Results           []*ExecutionResult
}

// QMDBStats contains executor statistics
type QMDBStats struct {
	TotalTxsProcessed    uint64
	TotalBlocksProcessed uint64
	NumThreads           uint64
}

// NewQMDBExecutor creates a new QMDB parallel executor
// numThreads: 0 = use all available CPUs
func NewQMDBExecutor(numThreads uint64) (*QMDBExecutor, error) {
	if numThreads == 0 {
		numThreads = uint64(runtime.NumCPU())
	}

	handle := C.qmdb_executor_new(C.uint64_t(numThreads))
	if handle == nil {
		return nil, errors.New("failed to create QMDB executor")
	}

	return &QMDBExecutor{handle: handle}, nil
}

// Close frees the executor resources
func (q *QMDBExecutor) Close() {
	if q.handle != nil {
		C.qmdb_executor_free(q.handle)
		q.handle = nil
	}
}

// SetBlockContext sets the current block context for execution
func (q *QMDBExecutor) SetBlockContext(slot uint64, timestamp int64, blockhash []byte) error {
	if q.handle == nil {
		return errors.New("executor is closed")
	}

	if len(blockhash) != 32 {
		return errors.New("blockhash must be 32 bytes")
	}

	result := C.qmdb_set_block_context(
		q.handle,
		C.uint64_t(slot),
		C.int64_t(timestamp),
		(*C.uint8_t)(unsafe.Pointer(&blockhash[0])),
	)

	if result != 0 {
		return errors.New("failed to set block context")
	}

	return nil
}

// ExecuteBlock executes a full block of transactions with atomic state commit
// Returns the block result including merkle root
func (q *QMDBExecutor) ExecuteBlock(blockHeight uint64, txDataList [][]byte) (*BlockResult, error) {
	if q.handle == nil {
		return nil, errors.New("executor is closed")
	}

	if len(txDataList) == 0 {
		return &BlockResult{
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

	var resultPtr *C.BlockResultFFI

	status := C.qmdb_execute_block(
		q.handle,
		C.uint64_t(blockHeight),
		(**C.uint8_t)(unsafe.Pointer(&txPtrs[0])),
		(*C.size_t)(unsafe.Pointer(&txLens[0])),
		C.size_t(txCount),
		&resultPtr,
	)

	if status != 0 {
		return nil, errors.New("block execution failed")
	}

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}

	defer C.qmdb_free_block_result(resultPtr)

	// Convert C result to Go
	result := &BlockResult{
		Successful:        uint64(resultPtr.successful),
		Failed:            uint64(resultPtr.failed),
		TotalComputeUnits: uint64(resultPtr.total_compute_units),
		TotalFees:         uint64(resultPtr.total_fees),
		ExecutionTimeUs:   uint64(resultPtr.execution_time_us),
		BlockHeight:       uint64(resultPtr.block_height),
	}

	// Copy merkle root
	for i := 0; i < 32; i++ {
		result.MerkleRoot[i] = byte(resultPtr.merkle_root[i])
	}

	// Convert individual results
	if resultPtr.results != nil && resultPtr.results_len > 0 {
		results := unsafe.Slice(resultPtr.results, resultPtr.results_len)
		for _, r := range results {
			execResult := &ExecutionResult{
				Success:          r.success != 0,
				ComputeUnitsUsed: uint64(r.compute_units_used),
				Fee:              uint64(r.fee),
			}

			if r.error_msg != nil {
				execResult.Error = C.GoString(r.error_msg)
			}

			// Copy logs
			if r.logs != nil && r.logs_len > 0 {
				logs := unsafe.Slice(r.logs, r.logs_len)
				for _, log := range logs {
					if log != nil {
						execResult.Logs = append(execResult.Logs, C.GoString(log))
					}
				}
			}

			result.Results = append(result.Results, execResult)
		}
	}

	return result, nil
}

// GetMerkleRoot returns the current state merkle root
func (q *QMDBExecutor) GetMerkleRoot() ([32]byte, error) {
	if q.handle == nil {
		return [32]byte{}, errors.New("executor is closed")
	}

	var root [32]byte
	status := C.qmdb_get_merkle_root(q.handle, (*C.uint8_t)(unsafe.Pointer(&root[0])))

	if status != 0 {
		return [32]byte{}, errors.New("failed to get merkle root")
	}

	return root, nil
}

// LoadAccount loads an account into the QMDB state
func (q *QMDBExecutor) LoadAccount(pubkey [32]byte, lamports uint64, data []byte, owner [32]byte, executable bool) error {
	if q.handle == nil {
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

	status := C.qmdb_load_account(
		q.handle,
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

// GetAccount retrieves an account from the QMDB state
func (q *QMDBExecutor) GetAccount(pubkey [32]byte) (*StateChange, error) {
	if q.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var lamports C.uint64_t
	var dataPtr *C.uint8_t
	var dataLen C.size_t
	var owner [32]byte
	var executable C.int

	status := C.qmdb_get_account(
		q.handle,
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
		// Free the data allocated by Rust
		C.free(unsafe.Pointer(dataPtr))
	}

	return account, nil
}

// GetStats returns executor statistics
func (q *QMDBExecutor) GetStats() (*QMDBStats, error) {
	if q.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var totalTxs, totalBlocks, numThreads C.uint64_t

	status := C.qmdb_get_stats(
		q.handle,
		&totalTxs,
		&totalBlocks,
		&numThreads,
	)

	if status != 0 {
		return nil, errors.New("failed to get stats")
	}

	return &QMDBStats{
		TotalTxsProcessed:    uint64(totalTxs),
		TotalBlocksProcessed: uint64(totalBlocks),
		NumThreads:           uint64(numThreads),
	}, nil
}

// String returns a string representation of the block result
func (r *BlockResult) String() string {
	return fmt.Sprintf("BlockResult{height=%d, successful=%d, failed=%d, compute=%d, fees=%d, time=%dus, merkle=%x}",
		r.BlockHeight, r.Successful, r.Failed, r.TotalComputeUnits, r.TotalFees, r.ExecutionTimeUs, r.MerkleRoot[:8])
}
