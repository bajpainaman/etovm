package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lsvm_runtime -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../../runtime

#include <stdint.h>
#include <stdlib.h>

// Forward declarations matching Rust FFI exports
typedef struct RuntimeHandle RuntimeHandle;

typedef struct {
    uint8_t pubkey[32];
    uint8_t* account_data;
    size_t account_data_len;
    uint64_t lamports;
    uint8_t owner[32];
    int executable;
    uint64_t rent_epoch;
} StateChangeFFI;

typedef struct {
    int success;
    uint64_t compute_units_used;
    uint64_t fee;
    char* error_msg;
    char** logs;
    size_t logs_len;
    StateChangeFFI* state_changes;
    size_t state_changes_len;
} ExecutionResultFFI;

// Runtime lifecycle
extern RuntimeHandle* svm_runtime_new();
extern void svm_runtime_free(RuntimeHandle* handle);

// Block context
extern int svm_set_block_context(RuntimeHandle* handle, uint64_t slot, int64_t timestamp, const uint8_t* blockhash);

// Transaction execution
extern int svm_execute_transaction(RuntimeHandle* handle, const uint8_t* tx_data, size_t tx_len, ExecutionResultFFI** result_out);
extern void svm_free_result(ExecutionResultFFI* result);

// State queries
extern uint64_t svm_get_slot(RuntimeHandle* handle);
extern int svm_get_blockhash(RuntimeHandle* handle, uint8_t* blockhash_out);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Runtime wraps the Rust SVM runtime
type Runtime struct {
	handle *C.RuntimeHandle
}

// ExecutionResult contains the result of transaction execution
type ExecutionResult struct {
	Success           bool
	ComputeUnitsUsed  uint64
	Fee               uint64
	Error             string
	Logs              []string
	StateChanges      []StateChange
}

// StateChange represents a change to an account
type StateChange struct {
	Pubkey     [32]byte
	Lamports   uint64
	Data       []byte
	Owner      [32]byte
	Executable bool
	RentEpoch  uint64
}

// NewRuntime creates a new SVM runtime instance
func NewRuntime() (*Runtime, error) {
	handle := C.svm_runtime_new()
	if handle == nil {
		return nil, errors.New("failed to create runtime")
	}

	return &Runtime{handle: handle}, nil
}

// Close frees the runtime resources
func (r *Runtime) Close() {
	if r.handle != nil {
		C.svm_runtime_free(r.handle)
		r.handle = nil
	}
}

// SetBlockContext sets the current block context for execution
func (r *Runtime) SetBlockContext(slot uint64, timestamp int64, blockhash []byte) error {
	if r.handle == nil {
		return errors.New("runtime is closed")
	}

	if len(blockhash) != 32 {
		return errors.New("blockhash must be 32 bytes")
	}

	result := C.svm_set_block_context(
		r.handle,
		C.uint64_t(slot),
		C.int64_t(timestamp),
		(*C.uint8_t)(unsafe.Pointer(&blockhash[0])),
	)

	if result != 0 {
		return errors.New("failed to set block context")
	}

	return nil
}

// ExecuteTransaction executes a serialized transaction
func (r *Runtime) ExecuteTransaction(txData []byte) (*ExecutionResult, error) {
	if r.handle == nil {
		return nil, errors.New("runtime is closed")
	}

	if len(txData) == 0 {
		return nil, errors.New("empty transaction data")
	}

	var resultPtr *C.ExecutionResultFFI

	status := C.svm_execute_transaction(
		r.handle,
		(*C.uint8_t)(unsafe.Pointer(&txData[0])),
		C.size_t(len(txData)),
		&resultPtr,
	)

	if status != 0 {
		return nil, errors.New("transaction execution failed")
	}

	if resultPtr == nil {
		return nil, errors.New("null result from runtime")
	}

	defer C.svm_free_result(resultPtr)

	// Convert C result to Go
	result := &ExecutionResult{
		Success:          resultPtr.success != 0,
		ComputeUnitsUsed: uint64(resultPtr.compute_units_used),
		Fee:              uint64(resultPtr.fee),
	}

	// Copy error message
	if resultPtr.error_msg != nil {
		result.Error = C.GoString(resultPtr.error_msg)
	}

	// Copy logs
	if resultPtr.logs != nil && resultPtr.logs_len > 0 {
		logs := unsafe.Slice(resultPtr.logs, resultPtr.logs_len)
		for _, log := range logs {
			if log != nil {
				result.Logs = append(result.Logs, C.GoString(log))
			}
		}
	}

	// Copy state changes
	if resultPtr.state_changes != nil && resultPtr.state_changes_len > 0 {
		changes := unsafe.Slice(resultPtr.state_changes, resultPtr.state_changes_len)
		for _, change := range changes {
			sc := StateChange{
				Lamports:   uint64(change.lamports),
				Executable: change.executable != 0,
				RentEpoch:  uint64(change.rent_epoch),
			}

			// Copy pubkey
			for i := 0; i < 32; i++ {
				sc.Pubkey[i] = byte(change.pubkey[i])
			}

			// Copy owner
			for i := 0; i < 32; i++ {
				sc.Owner[i] = byte(change.owner[i])
			}

			// Copy account data
			if change.account_data != nil && change.account_data_len > 0 {
				sc.Data = C.GoBytes(unsafe.Pointer(change.account_data), C.int(change.account_data_len))
			}

			result.StateChanges = append(result.StateChanges, sc)
		}
	}

	return result, nil
}

// GetSlot returns the current slot
func (r *Runtime) GetSlot() (uint64, error) {
	if r.handle == nil {
		return 0, errors.New("runtime is closed")
	}

	return uint64(C.svm_get_slot(r.handle)), nil
}

// GetBlockhash returns the current blockhash
func (r *Runtime) GetBlockhash() ([32]byte, error) {
	if r.handle == nil {
		return [32]byte{}, errors.New("runtime is closed")
	}

	var hash [32]byte
	result := C.svm_get_blockhash(r.handle, (*C.uint8_t)(unsafe.Pointer(&hash[0])))

	if result != 0 {
		return [32]byte{}, errors.New("failed to get blockhash")
	}

	return hash, nil
}

// String returns a string representation of the execution result
func (r *ExecutionResult) String() string {
	status := "failed"
	if r.Success {
		status = "success"
	}

	return fmt.Sprintf("ExecutionResult{status=%s, compute=%d, fee=%d, error=%q, stateChanges=%d}",
		status, r.ComputeUnitsUsed, r.Fee, r.Error, len(r.StateChanges))
}
