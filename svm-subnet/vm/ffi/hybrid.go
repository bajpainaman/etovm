package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../runtime/target/release -lsvm_runtime -ldl -lpthread -lm
#cgo CFLAGS: -I${SRCDIR}/../../runtime

#include <stdint.h>
#include <stdlib.h>

// Hybrid executor handle
typedef struct HybridHandle HybridHandle;

// State change struct
typedef struct {
    uint8_t pubkey[32];
    uint8_t* account_data;
    size_t account_data_len;
    uint64_t lamports;
    uint8_t owner[32];
    int executable;
    uint64_t rent_epoch;
} StateChangeFFI;

// Execution result struct
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

// Result cleanup
extern void svm_free_result(ExecutionResultFFI* result);

// Hybrid lifecycle
extern HybridHandle* hybrid_new(uint64_t chain_id);
extern void hybrid_free(HybridHandle* handle);

// Block context
extern int hybrid_set_block(HybridHandle* handle, uint64_t slot, int64_t timestamp);

// Account management
extern int hybrid_load_account(
    HybridHandle* handle,
    const uint8_t* pubkey,
    uint64_t lamports,
    const uint8_t* data,
    size_t data_len,
    const uint8_t* owner,
    int executable
);

extern int hybrid_get_account(
    HybridHandle* handle,
    const uint8_t* pubkey,
    uint64_t* lamports_out,
    uint8_t** data_out,
    size_t* data_len_out,
    uint8_t* owner_out,
    int* executable_out
);

// Execution
extern int hybrid_execute_evm(
    HybridHandle* handle,
    const uint8_t* caller,
    const uint8_t* to,
    uint64_t value_lo,
    uint64_t value_hi,
    const uint8_t* data,
    size_t data_len,
    uint64_t gas_limit,
    ExecutionResultFFI** result_out
);

extern int hybrid_execute_bpf(
    HybridHandle* handle,
    const uint8_t* program_id,
    const uint8_t* bytecode,
    size_t bytecode_len,
    const uint8_t* input_data,
    size_t input_len,
    ExecutionResultFFI** result_out
);

// Cross-VM calls
extern int hybrid_svm_call_evm(
    HybridHandle* handle,
    const uint8_t* caller_pubkey,
    const uint8_t* evm_contract,
    const uint8_t* calldata,
    size_t calldata_len,
    uint64_t value_lo,
    uint64_t value_hi,
    uint8_t** result_out,
    size_t* result_len_out
);
*/
import "C"

import (
	"errors"
	"math/big"
	"unsafe"
)

// HybridExecutor wraps the Rust hybrid SVM+EVM executor
type HybridExecutor struct {
	handle *C.HybridHandle
}

// NewHybridExecutor creates a new hybrid executor
func NewHybridExecutor(chainID uint64) (*HybridExecutor, error) {
	handle := C.hybrid_new(C.uint64_t(chainID))
	if handle == nil {
		return nil, errors.New("failed to create hybrid executor")
	}
	return &HybridExecutor{handle: handle}, nil
}

// Close frees the hybrid executor resources
func (h *HybridExecutor) Close() {
	if h.handle != nil {
		C.hybrid_free(h.handle)
		h.handle = nil
	}
}

// SetBlock sets the current block context
func (h *HybridExecutor) SetBlock(slot uint64, timestamp int64) error {
	if h.handle == nil {
		return errors.New("executor is closed")
	}
	result := C.hybrid_set_block(h.handle, C.uint64_t(slot), C.int64_t(timestamp))
	if result != 0 {
		return errors.New("failed to set block context")
	}
	return nil
}

// LoadAccount loads an account into the executor
func (h *HybridExecutor) LoadAccount(pubkey [32]byte, lamports uint64, data []byte, owner [32]byte, executable bool) error {
	if h.handle == nil {
		return errors.New("executor is closed")
	}

	var dataPtr *C.uint8_t
	var dataLen C.size_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
		dataLen = C.size_t(len(data))
	}

	execFlag := 0
	if executable {
		execFlag = 1
	}

	result := C.hybrid_load_account(
		h.handle,
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		C.uint64_t(lamports),
		dataPtr,
		dataLen,
		(*C.uint8_t)(unsafe.Pointer(&owner[0])),
		C.int(execFlag),
	)

	if result != 0 {
		return errors.New("failed to load account")
	}
	return nil
}

// GetAccount retrieves an account from the executor
func (h *HybridExecutor) GetAccount(pubkey [32]byte) (*StateChange, error) {
	if h.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var lamports C.uint64_t
	var dataPtr *C.uint8_t
	var dataLen C.size_t
	var owner [32]C.uint8_t
	var executable C.int

	result := C.hybrid_get_account(
		h.handle,
		(*C.uint8_t)(unsafe.Pointer(&pubkey[0])),
		&lamports,
		&dataPtr,
		&dataLen,
		(*C.uint8_t)(unsafe.Pointer(&owner[0])),
		&executable,
	)

	if result != 0 {
		return nil, errors.New("account not found")
	}

	account := &StateChange{
		Pubkey:     pubkey,
		Lamports:   uint64(lamports),
		Executable: executable != 0,
	}

	for i := 0; i < 32; i++ {
		account.Owner[i] = byte(owner[i])
	}

	if dataPtr != nil && dataLen > 0 {
		account.Data = C.GoBytes(unsafe.Pointer(dataPtr), C.int(dataLen))
		C.free(unsafe.Pointer(dataPtr))
	}

	return account, nil
}

// ExecuteEVM executes an EVM transaction
func (h *HybridExecutor) ExecuteEVM(caller [20]byte, to *[20]byte, value *big.Int, data []byte, gasLimit uint64) (*ExecutionResult, error) {
	if h.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var toPtr *C.uint8_t
	if to != nil {
		toPtr = (*C.uint8_t)(unsafe.Pointer(&(*to)[0]))
	}

	var valueLo, valueHi uint64
	if value != nil {
		// Split 128-bit value into two 64-bit parts
		valueLo = value.Uint64()
		shifted := new(big.Int).Rsh(value, 64)
		valueHi = shifted.Uint64()
	}

	var dataPtr *C.uint8_t
	var dataLen C.size_t
	if len(data) > 0 {
		dataPtr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
		dataLen = C.size_t(len(data))
	}

	var resultPtr *C.ExecutionResultFFI
	status := C.hybrid_execute_evm(
		h.handle,
		(*C.uint8_t)(unsafe.Pointer(&caller[0])),
		toPtr,
		C.uint64_t(valueLo),
		C.uint64_t(valueHi),
		dataPtr,
		dataLen,
		C.uint64_t(gasLimit),
		&resultPtr,
	)

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}
	defer C.svm_free_result(resultPtr)

	result := &ExecutionResult{
		Success:          resultPtr.success != 0,
		ComputeUnitsUsed: uint64(resultPtr.compute_units_used),
		Fee:              uint64(resultPtr.fee),
	}

	if resultPtr.error_msg != nil {
		result.Error = C.GoString(resultPtr.error_msg)
	}

	if status != 0 && !result.Success {
		return result, errors.New(result.Error)
	}

	return result, nil
}

// ExecuteBPF executes BPF bytecode
func (h *HybridExecutor) ExecuteBPF(programID [32]byte, bytecode []byte, inputData []byte) (*ExecutionResult, error) {
	if h.handle == nil {
		return nil, errors.New("executor is closed")
	}

	if len(bytecode) == 0 {
		return nil, errors.New("empty bytecode")
	}

	var inputPtr *C.uint8_t
	var inputLen C.size_t
	if len(inputData) > 0 {
		inputPtr = (*C.uint8_t)(unsafe.Pointer(&inputData[0]))
		inputLen = C.size_t(len(inputData))
	}

	var resultPtr *C.ExecutionResultFFI
	returnValue := C.hybrid_execute_bpf(
		h.handle,
		(*C.uint8_t)(unsafe.Pointer(&programID[0])),
		(*C.uint8_t)(unsafe.Pointer(&bytecode[0])),
		C.size_t(len(bytecode)),
		inputPtr,
		inputLen,
		&resultPtr,
	)

	if resultPtr == nil {
		return nil, errors.New("null result from executor")
	}
	defer C.svm_free_result(resultPtr)

	result := &ExecutionResult{
		Success:          resultPtr.success != 0,
		ComputeUnitsUsed: uint64(resultPtr.compute_units_used),
		Fee:              uint64(resultPtr.fee),
	}

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

	if returnValue < 0 && !result.Success {
		return result, errors.New(result.Error)
	}

	return result, nil
}

// SVMCallEVM performs a cross-VM call from SVM to EVM
func (h *HybridExecutor) SVMCallEVM(callerPubkey [32]byte, evmContract [20]byte, calldata []byte, value *big.Int) ([]byte, error) {
	if h.handle == nil {
		return nil, errors.New("executor is closed")
	}

	var valueLo, valueHi uint64
	if value != nil {
		valueLo = value.Uint64()
		shifted := new(big.Int).Rsh(value, 64)
		valueHi = shifted.Uint64()
	}

	var calldataPtr *C.uint8_t
	var calldataLen C.size_t
	if len(calldata) > 0 {
		calldataPtr = (*C.uint8_t)(unsafe.Pointer(&calldata[0]))
		calldataLen = C.size_t(len(calldata))
	}

	var resultPtr *C.uint8_t
	var resultLen C.size_t

	status := C.hybrid_svm_call_evm(
		h.handle,
		(*C.uint8_t)(unsafe.Pointer(&callerPubkey[0])),
		(*C.uint8_t)(unsafe.Pointer(&evmContract[0])),
		calldataPtr,
		calldataLen,
		C.uint64_t(valueLo),
		C.uint64_t(valueHi),
		&resultPtr,
		&resultLen,
	)

	if status != 0 {
		return nil, errors.New("cross-VM call failed")
	}

	if resultPtr == nil || resultLen == 0 {
		return []byte{}, nil
	}

	result := C.GoBytes(unsafe.Pointer(resultPtr), C.int(resultLen))
	C.free(unsafe.Pointer(resultPtr))

	return result, nil
}
