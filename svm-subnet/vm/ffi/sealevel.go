package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lsvm_runtime -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../../runtime

#include <stdint.h>
#include <stdlib.h>

// Forward declarations for Sealevel parallel executor
typedef struct ParallelExecutorHandle ParallelExecutorHandle;

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
} SealevelResultFFI;

typedef struct {
    uint64_t successful;
    uint64_t failed;
    uint64_t total_compute_units;
    uint64_t total_fees;
    uint64_t execution_time_us;
    SealevelResultFFI* results;
    size_t results_len;
} BatchResultFFI;

// Sealevel lifecycle
extern ParallelExecutorHandle* sealevel_new(uint64_t num_threads);
extern void sealevel_free(ParallelExecutorHandle* handle);

// Block context
extern int sealevel_set_block(ParallelExecutorHandle* handle, uint64_t slot, int64_t timestamp, const uint8_t* blockhash);

// Batch execution
extern int sealevel_execute_batch(
    ParallelExecutorHandle* handle,
    const uint8_t** tx_data,
    const size_t* tx_lens,
    size_t tx_count,
    BatchResultFFI** result_out
);
extern void sealevel_free_batch_result(BatchResultFFI* result);

// Statistics
extern int sealevel_get_stats(
    ParallelExecutorHandle* handle,
    uint64_t* total_txs,
    uint64_t* total_batches,
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

// SealevelExecutor wraps the Rust parallel executor
type SealevelExecutor struct {
	handle *C.ParallelExecutorHandle
}

// BatchResult contains the result of batch execution
type BatchResult struct {
	Successful        uint64
	Failed            uint64
	TotalComputeUnits uint64
	TotalFees         uint64
	ExecutionTimeUs   uint64
	Results           []*ExecutionResult
}

// SealevelStats contains executor statistics
type SealevelStats struct {
	TotalTxsProcessed    uint64
	TotalBatchesProcessed uint64
	NumThreads           uint64
}

// NewSealevelExecutor creates a new parallel executor
// numThreads: 0 = use all available CPUs
func NewSealevelExecutor(numThreads uint64) (*SealevelExecutor, error) {
	if numThreads == 0 {
		numThreads = uint64(runtime.NumCPU())
	}

	handle := C.sealevel_new(C.uint64_t(numThreads))
	if handle == nil {
		return nil, errors.New("failed to create sealevel executor")
	}

	return &SealevelExecutor{handle: handle}, nil
}

// Close frees the executor resources
func (s *SealevelExecutor) Close() {
	if s.handle != nil {
		C.sealevel_free(s.handle)
		s.handle = nil
	}
}

// SetBlockContext sets the current block context for execution
func (s *SealevelExecutor) SetBlockContext(slot uint64, timestamp int64, blockhash []byte) error {
	if s.handle == nil {
		return errors.New("executor is closed")
	}

	if len(blockhash) != 32 {
		return errors.New("blockhash must be 32 bytes")
	}

	result := C.sealevel_set_block(
		s.handle,
		C.uint64_t(slot),
		C.int64_t(timestamp),
		(*C.uint8_t)(unsafe.Pointer(&blockhash[0])),
	)

	if result != 0 {
		return errors.New("failed to set block context")
	}

	return nil
}

// ExecuteBatch executes multiple transactions in parallel
func (s *SealevelExecutor) ExecuteBatch(txDataList [][]byte) (*BatchResult, error) {
	if s.handle == nil {
		return nil, errors.New("executor is closed")
	}

	if len(txDataList) == 0 {
		return &BatchResult{}, nil
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

	var resultPtr *C.BatchResultFFI

	status := C.sealevel_execute_batch(
		s.handle,
		(**C.uint8_t)(unsafe.Pointer(&txPtrs[0])),
		(*C.size_t)(unsafe.Pointer(&txLens[0])),
		C.size_t(txCount),
		&resultPtr,
	)

	if status != 0 {
		return nil, errors.New("batch execution failed")
	}

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}

	defer C.sealevel_free_batch_result(resultPtr)

	// Convert C result to Go
	result := &BatchResult{
		Successful:        uint64(resultPtr.successful),
		Failed:            uint64(resultPtr.failed),
		TotalComputeUnits: uint64(resultPtr.total_compute_units),
		TotalFees:         uint64(resultPtr.total_fees),
		ExecutionTimeUs:   uint64(resultPtr.execution_time_us),
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

// GetStats returns executor statistics
func (s *SealevelExecutor) GetStats() (*SealevelStats, error) {
	if s.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var totalTxs, totalBatches, numThreads C.uint64_t

	status := C.sealevel_get_stats(
		s.handle,
		&totalTxs,
		&totalBatches,
		&numThreads,
	)

	if status != 0 {
		return nil, errors.New("failed to get stats")
	}

	return &SealevelStats{
		TotalTxsProcessed:    uint64(totalTxs),
		TotalBatchesProcessed: uint64(totalBatches),
		NumThreads:           uint64(numThreads),
	}, nil
}

// String returns a string representation of the batch result
func (r *BatchResult) String() string {
	return fmt.Sprintf("BatchResult{successful=%d, failed=%d, compute=%d, fees=%d, time=%dus}",
		r.Successful, r.Failed, r.TotalComputeUnits, r.TotalFees, r.ExecutionTimeUs)
}
