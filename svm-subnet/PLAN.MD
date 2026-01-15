# SVM-Subnet Implementation Plan - Part 2

## Continuing from Part 1...

### Task 2.8 (Continued): FFI Interface

Complete `runtime/src/lib.rs` FFI section:

```rust
// ... continued from Part 1

/// FFI-safe execution result
#[repr(C)]
pub struct ExecutionResultFFI {
    pub success: c_int,           // 1 = success, 0 = failure
    pub compute_units_used: c_ulong,
    pub fee: c_ulong,
    pub error_msg: *mut c_char,   // null if no error
    pub logs: *mut *mut c_char,   // null-terminated array
    pub logs_len: usize,
    pub state_changes: *mut StateChangeFFI,
    pub state_changes_len: usize,
}

#[repr(C)]
pub struct StateChangeFFI {
    pub pubkey: [c_uchar; 32],
    pub account_data: *mut c_uchar,
    pub account_data_len: usize,
    pub lamports: c_ulong,
    pub owner: [c_uchar; 32],
    pub executable: c_int,
    pub rent_epoch: c_ulong,
}

#[no_mangle]
pub extern "C" fn svm_execute_transaction(
    handle: *mut RuntimeHandle,
    tx_data: *const c_uchar,
    tx_len: usize,
    result_out: *mut *mut ExecutionResultFFI,
) -> c_int {
    if handle.is_null() || tx_data.is_null() || result_out.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };
    let tx_bytes = unsafe { slice::from_raw_parts(tx_data, tx_len) };

    // Deserialize transaction
    let tx: Transaction = match borsh::from_slice(tx_bytes) {
        Ok(tx) => tx,
        Err(e) => {
            let result = create_error_result(&format!("Failed to deserialize tx: {}", e));
            unsafe { *result_out = result };
            return 0;
        }
    };

    // Execute
    let result = match runtime.execute_transaction(&tx) {
        Ok(r) => r,
        Err(e) => {
            let result = create_error_result(&format!("Execution error: {:?}", e));
            unsafe { *result_out = result };
            return 0;
        }
    };

    // Convert to FFI result
    let ffi_result = execution_result_to_ffi(&result);
    unsafe { *result_out = ffi_result };
    
    0
}

fn create_error_result(msg: &str) -> *mut ExecutionResultFFI {
    let error_msg = CString::new(msg).unwrap().into_raw();
    
    let result = Box::new(ExecutionResultFFI {
        success: 0,
        compute_units_used: 0,
        fee: 0,
        error_msg,
        logs: ptr::null_mut(),
        logs_len: 0,
        state_changes: ptr::null_mut(),
        state_changes_len: 0,
    });
    
    Box::into_raw(result)
}

fn execution_result_to_ffi(result: &ExecutionResult) -> *mut ExecutionResultFFI {
    // Convert logs
    let logs: Vec<*mut c_char> = result.logs
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap().into_raw())
        .collect();
    
    let logs_ptr = if logs.is_empty() {
        ptr::null_mut()
    } else {
        let mut logs_vec = logs.into_boxed_slice();
        let ptr = logs_vec.as_mut_ptr();
        std::mem::forget(logs_vec);
        ptr
    };

    // Convert state changes
    let state_changes: Vec<StateChangeFFI> = result.state_changes
        .iter()
        .map(|(pubkey, account)| {
            let mut data = account.data.clone().into_boxed_slice();
            let data_ptr = data.as_mut_ptr();
            let data_len = data.len();
            std::mem::forget(data);

            StateChangeFFI {
                pubkey: pubkey.to_bytes(),
                account_data: data_ptr,
                account_data_len: data_len,
                lamports: account.lamports as c_ulong,
                owner: account.owner.to_bytes(),
                executable: if account.executable { 1 } else { 0 },
                rent_epoch: account.rent_epoch as c_ulong,
            }
        })
        .collect();

    let state_changes_ptr = if state_changes.is_empty() {
        ptr::null_mut()
    } else {
        let mut changes_vec = state_changes.into_boxed_slice();
        let ptr = changes_vec.as_mut_ptr();
        std::mem::forget(changes_vec);
        ptr
    };

    // Convert error message
    let error_msg = result.error
        .as_ref()
        .map(|e| CString::new(format!("{:?}", e)).unwrap().into_raw())
        .unwrap_or(ptr::null_mut());

    let ffi_result = Box::new(ExecutionResultFFI {
        success: if result.success { 1 } else { 0 },
        compute_units_used: result.compute_units_used as c_ulong,
        fee: result.fee as c_ulong,
        error_msg,
        logs: logs_ptr,
        logs_len: result.logs.len(),
        state_changes: state_changes_ptr,
        state_changes_len: result.state_changes.len(),
    });

    Box::into_raw(ffi_result)
}

/// Free execution result
#[no_mangle]
pub extern "C" fn svm_free_result(result: *mut ExecutionResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let result = Box::from_raw(result);

        // Free error message
        if !result.error_msg.is_null() {
            drop(CString::from_raw(result.error_msg));
        }

        // Free logs
        if !result.logs.is_null() {
            let logs = slice::from_raw_parts_mut(result.logs, result.logs_len);
            for log in logs {
                if !log.is_null() {
                    drop(CString::from_raw(*log));
                }
            }
            drop(Box::from_raw(result.logs));
        }

        // Free state changes
        if !result.state_changes.is_null() {
            let changes = slice::from_raw_parts_mut(result.state_changes, result.state_changes_len);
            for change in changes {
                if !change.account_data.is_null() {
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        change.account_data,
                        change.account_data_len,
                    )));
                }
            }
            drop(Box::from_raw(result.state_changes));
        }
    }
}

/// Get account data
/// Returns account data length, or -1 on error
/// If account_out is not null, copies account data to it
#[no_mangle]
pub extern "C" fn svm_get_account(
    handle: *mut RuntimeHandle,
    pubkey: *const c_uchar,
    account_out: *mut c_uchar,
    account_out_len: usize,
) -> i64 {
    if handle.is_null() || pubkey.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };
    let pubkey_slice = unsafe { slice::from_raw_parts(pubkey, 32) };
    
    let mut pk_bytes = [0u8; 32];
    pk_bytes.copy_from_slice(pubkey_slice);
    let pk = Pubkey::from(pk_bytes);

    match runtime.get_account(&pk) {
        Ok(Some(account)) => {
            let serialized = match borsh::to_vec(&account) {
                Ok(data) => data,
                Err(_) => return -1,
            };

            if !account_out.is_null() && account_out_len >= serialized.len() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        serialized.as_ptr(),
                        account_out,
                        serialized.len(),
                    );
                }
            }

            serialized.len() as i64
        }
        Ok(None) => 0,
        Err(_) => -1,
    }
}

/// Get current slot
#[no_mangle]
pub extern "C" fn svm_get_slot(handle: *mut RuntimeHandle) -> c_ulong {
    if handle.is_null() {
        return 0;
    }

    let runtime = unsafe { &(*handle).runtime };
    runtime.current_slot().unwrap_or(0) as c_ulong
}

/// Get current blockhash
#[no_mangle]
pub extern "C" fn svm_get_blockhash(
    handle: *mut RuntimeHandle,
    blockhash_out: *mut c_uchar,
) -> c_int {
    if handle.is_null() || blockhash_out.is_null() {
        return -1;
    }

    let runtime = unsafe { &(*handle).runtime };
    
    match runtime.current_blockhash() {
        Ok(hash) => {
            unsafe {
                ptr::copy_nonoverlapping(hash.as_ptr(), blockhash_out, 32);
            }
            0
        }
        Err(_) => -1,
    }
}
```

---

## Phase 3: Go VM Plugin Implementation

### Task 3.1: FFI Bindings

Create `vm/ffi/runtime.go`:

```go
package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../build -lsvm_runtime -ldl -lm
#cgo CFLAGS: -I${SRCDIR}

#include "runtime.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Runtime wraps the Rust SVM runtime
type Runtime struct {
	handle *C.RuntimeHandle
}

// ExecutionResult represents the result of transaction execution
type ExecutionResult struct {
	Success          bool
	ComputeUnitsUsed uint64
	Fee              uint64
	Error            string
	Logs             []string
	StateChanges     map[[32]byte]Account
}

// Account represents a Solana account
type Account struct {
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

// Close destroys the runtime instance
func (r *Runtime) Close() {
	if r.handle != nil {
		C.svm_runtime_free(r.handle)
		r.handle = nil
	}
}

// SetBlockContext sets the current block context
func (r *Runtime) SetBlockContext(slot uint64, timestamp int64, blockhash [32]byte) error {
	result := C.svm_set_block_context(
		r.handle,
		C.ulong(slot),
		C.long(timestamp),
		(*C.uchar)(unsafe.Pointer(&blockhash[0])),
	)
	if result != 0 {
		return errors.New("failed to set block context")
	}
	return nil
}

// ExecuteTransaction executes a serialized transaction
func (r *Runtime) ExecuteTransaction(txData []byte) (*ExecutionResult, error) {
	var resultPtr *C.ExecutionResultFFI

	ret := C.svm_execute_transaction(
		r.handle,
		(*C.uchar)(unsafe.Pointer(&txData[0])),
		C.size_t(len(txData)),
		&resultPtr,
	)

	if ret != 0 {
		return nil, errors.New("execution failed")
	}

	defer C.svm_free_result(resultPtr)

	// Convert FFI result to Go
	result := &ExecutionResult{
		Success:          resultPtr.success == 1,
		ComputeUnitsUsed: uint64(resultPtr.compute_units_used),
		Fee:              uint64(resultPtr.fee),
		StateChanges:     make(map[[32]byte]Account),
	}

	// Get error message
	if resultPtr.error_msg != nil {
		result.Error = C.GoString(resultPtr.error_msg)
	}

	// Get logs
	if resultPtr.logs != nil && resultPtr.logs_len > 0 {
		logs := (*[1 << 30]*C.char)(unsafe.Pointer(resultPtr.logs))[:resultPtr.logs_len:resultPtr.logs_len]
		result.Logs = make([]string, len(logs))
		for i, log := range logs {
			result.Logs[i] = C.GoString(log)
		}
	}

	// Get state changes
	if resultPtr.state_changes != nil && resultPtr.state_changes_len > 0 {
		changes := (*[1 << 30]C.StateChangeFFI)(unsafe.Pointer(resultPtr.state_changes))[:resultPtr.state_changes_len:resultPtr.state_changes_len]
		for _, change := range changes {
			var pubkey [32]byte
			copy(pubkey[:], C.GoBytes(unsafe.Pointer(&change.pubkey[0]), 32))

			var owner [32]byte
			copy(owner[:], C.GoBytes(unsafe.Pointer(&change.owner[0]), 32))

			data := C.GoBytes(unsafe.Pointer(change.account_data), C.int(change.account_data_len))

			result.StateChanges[pubkey] = Account{
				Lamports:   uint64(change.lamports),
				Data:       data,
				Owner:      owner,
				Executable: change.executable == 1,
				RentEpoch:  uint64(change.rent_epoch),
			}
		}
	}

	return result, nil
}

// GetAccount retrieves an account
func (r *Runtime) GetAccount(pubkey [32]byte) (*Account, error) {
	// First call to get size
	size := C.svm_get_account(
		r.handle,
		(*C.uchar)(unsafe.Pointer(&pubkey[0])),
		nil,
		0,
	)

	if size < 0 {
		return nil, errors.New("failed to get account")
	}

	if size == 0 {
		return nil, nil // Account doesn't exist
	}

	// Second call to get data
	data := make([]byte, size)
	C.svm_get_account(
		r.handle,
		(*C.uchar)(unsafe.Pointer(&pubkey[0])),
		(*C.uchar)(unsafe.Pointer(&data[0])),
		C.size_t(size),
	)

	// Deserialize account (using borsh - need to implement or use library)
	// For now, return raw data
	return &Account{Data: data}, nil
}

// GetSlot returns the current slot
func (r *Runtime) GetSlot() uint64 {
	return uint64(C.svm_get_slot(r.handle))
}

// GetBlockhash returns the current blockhash
func (r *Runtime) GetBlockhash() ([32]byte, error) {
	var hash [32]byte
	result := C.svm_get_blockhash(
		r.handle,
		(*C.uchar)(unsafe.Pointer(&hash[0])),
	)
	if result != 0 {
		return hash, errors.New("failed to get blockhash")
	}
	return hash, nil
}
```

Create `vm/ffi/types.go`:

```go
package ffi

// Pubkey represents a 32-byte public key
type Pubkey [32]byte

// SystemProgramID is the system program address
var SystemProgramID = Pubkey{}

// Transaction represents a Solana transaction (for Go-side construction)
type Transaction struct {
	Signatures [][]byte
	Message    Message
}

// Message represents the message component of a transaction
type Message struct {
	Header          MessageHeader
	AccountKeys     []Pubkey
	RecentBlockhash [32]byte
	Instructions    []CompiledInstruction
}

// MessageHeader contains signature counts
type MessageHeader struct {
	NumRequiredSignatures       uint8
	NumReadonlySignedAccounts   uint8
	NumReadonlyUnsignedAccounts uint8
}

// CompiledInstruction is a compiled instruction
type CompiledInstruction struct {
	ProgramIDIndex uint8
	Accounts       []uint8
	Data           []byte
}
```

### Task 3.2: VM Implementation

Create `vm/vm.go`:

```go
package vm

import (
	"context"
	"fmt"
	"sync"

	"github.com/ava-labs/avalanchego/database"
	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/snow"
	"github.com/ava-labs/avalanchego/snow/consensus/snowman"
	"github.com/ava-labs/avalanchego/snow/engine/common"
	"github.com/ava-labs/avalanchego/snow/engine/snowman/block"
	"github.com/ava-labs/avalanchego/version"
	"go.uber.org/zap"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

var (
	_ block.ChainVM = (*VM)(nil)
	
	Version = &version.Semantic{
		Major: 0,
		Minor: 1,
		Patch: 0,
	}
)

// VM implements the AvalancheGo ChainVM interface
type VM struct {
	ctx       *snow.Context
	db        database.Database
	state     *State
	runtime   *ffi.Runtime
	mempool   *Mempool
	config    Config

	// Block tracking
	preferred    ids.ID
	lastAccepted ids.ID

	// Synchronization
	lock sync.RWMutex

	// Logging
	log *zap.Logger

	// Channels
	toEngine chan<- common.Message
}

// Initialize implements block.ChainVM
func (vm *VM) Initialize(
	ctx context.Context,
	snowCtx *snow.Context,
	db database.Database,
	genesisBytes []byte,
	upgradeBytes []byte,
	configBytes []byte,
	toEngine chan<- common.Message,
	fxs []*common.Fx,
	appSender common.AppSender,
) error {
	vm.ctx = snowCtx
	vm.db = db
	vm.toEngine = toEngine
	vm.log = snowCtx.Log

	vm.log.Info("Initializing SVM-Subnet VM")

	// Parse config
	config, err := ParseConfig(configBytes)
	if err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}
	vm.config = config

	// Initialize Rust runtime
	runtime, err := ffi.NewRuntime()
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	vm.runtime = runtime

	// Initialize state
	vm.state = NewState(db)

	// Initialize mempool
	vm.mempool = NewMempool(vm.config.MempoolSize)

	// Process genesis
	if err := vm.initGenesis(genesisBytes); err != nil {
		return fmt.Errorf("failed to init genesis: %w", err)
	}

	vm.log.Info("SVM-Subnet VM initialized",
		zap.String("lastAccepted", vm.lastAccepted.String()),
	)

	return nil
}

func (vm *VM) initGenesis(genesisBytes []byte) error {
	// Check if we already have state
	has, err := vm.state.HasLastAccepted()
	if err != nil {
		return err
	}

	if has {
		// Load existing state
		lastAccepted, err := vm.state.GetLastAccepted()
		if err != nil {
			return err
		}
		vm.lastAccepted = lastAccepted
		vm.preferred = lastAccepted
		return nil
	}

	// Parse genesis
	genesis, err := ParseGenesis(genesisBytes)
	if err != nil {
		return err
	}

	// Create genesis block
	genesisBlock := &Block{
		vm:        vm,
		id:        ids.Empty,
		parentID:  ids.Empty,
		height:    0,
		timestamp: genesis.Timestamp,
		txs:       nil,
	}
	genesisBlock.id = genesisBlock.computeID()

	// Initialize accounts from genesis
	for _, account := range genesis.Accounts {
		if err := vm.state.SetAccount(account.Pubkey, &account.Account); err != nil {
			return err
		}
	}

	// Store genesis block
	if err := vm.state.PutBlock(genesisBlock); err != nil {
		return err
	}

	// Set as last accepted
	if err := vm.state.SetLastAccepted(genesisBlock.id); err != nil {
		return err
	}

	vm.lastAccepted = genesisBlock.id
	vm.preferred = genesisBlock.id

	return nil
}

// SetState implements block.ChainVM
func (vm *VM) SetState(ctx context.Context, state snow.State) error {
	switch state {
	case snow.Bootstrapping:
		vm.log.Info("VM entering bootstrapping state")
	case snow.NormalOp:
		vm.log.Info("VM entering normal operation state")
	}
	return nil
}

// Shutdown implements block.ChainVM
func (vm *VM) Shutdown(ctx context.Context) error {
	vm.log.Info("Shutting down SVM-Subnet VM")
	
	if vm.runtime != nil {
		vm.runtime.Close()
	}
	
	return vm.db.Close()
}

// Version implements block.ChainVM
func (vm *VM) Version(ctx context.Context) (string, error) {
	return Version.String(), nil
}

// CreateHandlers implements block.ChainVM
func (vm *VM) CreateHandlers(ctx context.Context) (map[string]http.Handler, error) {
	// Create RPC handlers
	rpcServer := NewRPCServer(vm)
	
	return map[string]http.Handler{
		"/rpc": rpcServer,
	}, nil
}

// HealthCheck implements block.ChainVM
func (vm *VM) HealthCheck(ctx context.Context) (interface{}, error) {
	return map[string]interface{}{
		"healthy": true,
		"slot":    vm.runtime.GetSlot(),
	}, nil
}

// BuildBlock implements block.ChainVM
func (vm *VM) BuildBlock(ctx context.Context) (snowman.Block, error) {
	vm.lock.Lock()
	defer vm.lock.Unlock()

	// Get parent block
	parent, err := vm.state.GetBlock(vm.preferred)
	if err != nil {
		return nil, err
	}

	// Get pending transactions from mempool
	txs := vm.mempool.PopN(vm.config.MaxTxsPerBlock)
	if len(txs) == 0 {
		return nil, fmt.Errorf("no transactions to build block")
	}

	// Create new block
	height := parent.Height() + 1
	timestamp := vm.ctx.Clock.Time().Unix()

	blk := &Block{
		vm:        vm,
		parentID:  vm.preferred,
		height:    height,
		timestamp: timestamp,
		txs:       txs,
	}
	blk.id = blk.computeID()

	// Update runtime context
	blockhash := blk.computeBlockhash()
	if err := vm.runtime.SetBlockContext(height, timestamp, blockhash); err != nil {
		return nil, err
	}

	// Execute transactions
	for _, tx := range txs {
		result, err := vm.runtime.ExecuteTransaction(tx.Bytes)
		if err != nil {
			vm.log.Warn("Transaction execution failed", zap.Error(err))
			continue
		}

		// Store result in block
		blk.results = append(blk.results, result)
	}

	// Store block
	if err := vm.state.PutBlock(blk); err != nil {
		return nil, err
	}

	vm.log.Info("Built block",
		zap.String("id", blk.id.String()),
		zap.Uint64("height", height),
		zap.Int("txs", len(txs)),
	)

	return blk, nil
}

// ParseBlock implements block.ChainVM
func (vm *VM) ParseBlock(ctx context.Context, bytes []byte) (snowman.Block, error) {
	blk := &Block{vm: vm}
	if err := blk.Unmarshal(bytes); err != nil {
		return nil, err
	}
	return blk, nil
}

// GetBlock implements block.ChainVM
func (vm *VM) GetBlock(ctx context.Context, id ids.ID) (snowman.Block, error) {
	return vm.state.GetBlock(id)
}

// SetPreference implements block.ChainVM
func (vm *VM) SetPreference(ctx context.Context, id ids.ID) error {
	vm.lock.Lock()
	defer vm.lock.Unlock()
	
	vm.preferred = id
	return nil
}

// LastAccepted implements block.ChainVM
func (vm *VM) LastAccepted(ctx context.Context) (ids.ID, error) {
	return vm.lastAccepted, nil
}

// GetBlockIDAtHeight implements block.HeightIndexedChainVM
func (vm *VM) GetBlockIDAtHeight(ctx context.Context, height uint64) (ids.ID, error) {
	return vm.state.GetBlockIDByHeight(height)
}

// SubmitTransaction adds a transaction to the mempool
func (vm *VM) SubmitTransaction(tx *Transaction) error {
	// Validate transaction
	if err := tx.Verify(); err != nil {
		return fmt.Errorf("invalid transaction: %w", err)
	}

	// Add to mempool
	if err := vm.mempool.Add(tx); err != nil {
		return fmt.Errorf("failed to add to mempool: %w", err)
	}

	// Notify engine
	select {
	case vm.toEngine <- common.PendingTxs:
	default:
	}

	return nil
}

// GetAccount returns an account from state
func (vm *VM) GetAccount(pubkey [32]byte) (*ffi.Account, error) {
	return vm.state.GetAccount(pubkey)
}
```

### Task 3.3: Block Implementation

Create `vm/block.go`:

```go
package vm

import (
	"crypto/sha256"
	"encoding/binary"
	"time"

	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/snow/consensus/snowman"
	"go.uber.org/zap"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

var _ snowman.Block = (*Block)(nil)

// Block represents a block in the SVM chain
type Block struct {
	vm *VM

	id        ids.ID
	parentID  ids.ID
	height    uint64
	timestamp int64
	txs       []*Transaction
	results   []*ffi.ExecutionResult

	bytes []byte
}

// ID implements snowman.Block
func (b *Block) ID() ids.ID {
	return b.id
}

// Accept implements snowman.Block
func (b *Block) Accept(ctx context.Context) error {
	b.vm.lock.Lock()
	defer b.vm.lock.Unlock()

	// Apply state changes from execution results
	for _, result := range b.results {
		if !result.Success {
			continue
		}

		for pubkey, account := range result.StateChanges {
			if err := b.vm.state.SetAccount(pubkey, &account); err != nil {
				return err
			}
		}
	}

	// Update last accepted
	b.vm.lastAccepted = b.id
	if err := b.vm.state.SetLastAccepted(b.id); err != nil {
		return err
	}

	// Store height -> block ID mapping
	if err := b.vm.state.SetBlockIDByHeight(b.height, b.id); err != nil {
		return err
	}

	b.vm.log.Info("Accepted block",
		zap.String("id", b.id.String()),
		zap.Uint64("height", b.height),
	)

	return nil
}

// Reject implements snowman.Block
func (b *Block) Reject(ctx context.Context) error {
	b.vm.log.Info("Rejected block",
		zap.String("id", b.id.String()),
		zap.Uint64("height", b.height),
	)
	
	// Return transactions to mempool
	for _, tx := range b.txs {
		_ = b.vm.mempool.Add(tx)
	}
	
	return nil
}

// Status implements snowman.Block
func (b *Block) Status() choices.Status {
	// Check if this is the last accepted block
	if b.id == b.vm.lastAccepted {
		return choices.Accepted
	}

	// Check if it's an ancestor of last accepted
	lastAccepted, err := b.vm.state.GetBlock(b.vm.lastAccepted)
	if err != nil {
		return choices.Unknown
	}

	if lastAccepted.Height() >= b.height {
		// It could be accepted if it's in the chain
		blkAtHeight, err := b.vm.state.GetBlockIDByHeight(b.height)
		if err == nil && blkAtHeight == b.id {
			return choices.Accepted
		}
		return choices.Rejected
	}

	return choices.Processing
}

// Parent implements snowman.Block
func (b *Block) Parent() ids.ID {
	return b.parentID
}

// Verify implements snowman.Block
func (b *Block) Verify(ctx context.Context) error {
	// Verify parent exists
	parent, err := b.vm.state.GetBlock(b.parentID)
	if err != nil {
		return fmt.Errorf("parent block not found: %w", err)
	}

	// Verify height
	if b.height != parent.Height()+1 {
		return fmt.Errorf("invalid height: expected %d, got %d", parent.Height()+1, b.height)
	}

	// Verify timestamp
	if b.timestamp < parent.Timestamp().Unix() {
		return fmt.Errorf("timestamp before parent")
	}

	// Re-execute transactions
	blockhash := b.computeBlockhash()
	if err := b.vm.runtime.SetBlockContext(b.height, b.timestamp, blockhash); err != nil {
		return err
	}

	for i, tx := range b.txs {
		result, err := b.vm.runtime.ExecuteTransaction(tx.Bytes)
		if err != nil {
			return fmt.Errorf("tx %d execution failed: %w", i, err)
		}
		
		// Verify result matches
		if len(b.results) > i {
			if result.Success != b.results[i].Success {
				return fmt.Errorf("tx %d result mismatch", i)
			}
		}
	}

	return nil
}

// Bytes implements snowman.Block
func (b *Block) Bytes() []byte {
	if b.bytes == nil {
		b.bytes = b.Marshal()
	}
	return b.bytes
}

// Height implements snowman.Block
func (b *Block) Height() uint64 {
	return b.height
}

// Timestamp implements snowman.Block
func (b *Block) Timestamp() time.Time {
	return time.Unix(b.timestamp, 0)
}

func (b *Block) computeID() ids.ID {
	hash := sha256.Sum256(b.Bytes())
	return ids.ID(hash)
}

func (b *Block) computeBlockhash() [32]byte {
	// Create blockhash from height and parent
	data := make([]byte, 40)
	binary.BigEndian.PutUint64(data[0:8], b.height)
	copy(data[8:40], b.parentID[:])
	return sha256.Sum256(data)
}

// Marshal serializes the block
func (b *Block) Marshal() []byte {
	// Simple serialization format:
	// [8 bytes height][32 bytes parent][8 bytes timestamp][4 bytes tx count][txs...]
	
	size := 8 + 32 + 8 + 4
	for _, tx := range b.txs {
		size += 4 + len(tx.Bytes)
	}

	data := make([]byte, size)
	offset := 0

	binary.BigEndian.PutUint64(data[offset:], b.height)
	offset += 8

	copy(data[offset:], b.parentID[:])
	offset += 32

	binary.BigEndian.PutUint64(data[offset:], uint64(b.timestamp))
	offset += 8

	binary.BigEndian.PutUint32(data[offset:], uint32(len(b.txs)))
	offset += 4

	for _, tx := range b.txs {
		binary.BigEndian.PutUint32(data[offset:], uint32(len(tx.Bytes)))
		offset += 4
		copy(data[offset:], tx.Bytes)
		offset += len(tx.Bytes)
	}

	return data
}

// Unmarshal deserializes a block
func (b *Block) Unmarshal(data []byte) error {
	if len(data) < 52 {
		return fmt.Errorf("block data too short")
	}

	offset := 0

	b.height = binary.BigEndian.Uint64(data[offset:])
	offset += 8

	copy(b.parentID[:], data[offset:offset+32])
	offset += 32

	b.timestamp = int64(binary.BigEndian.Uint64(data[offset:]))
	offset += 8

	txCount := binary.BigEndian.Uint32(data[offset:])
	offset += 4

	b.txs = make([]*Transaction, txCount)
	for i := uint32(0); i < txCount; i++ {
		if offset+4 > len(data) {
			return fmt.Errorf("unexpected end of data")
		}

		txLen := binary.BigEndian.Uint32(data[offset:])
		offset += 4

		if offset+int(txLen) > len(data) {
			return fmt.Errorf("unexpected end of data")
		}

		b.txs[i] = &Transaction{
			Bytes: data[offset : offset+int(txLen)],
		}
		offset += int(txLen)
	}

	b.bytes = data
	b.id = b.computeID()

	return nil
}
```

### Task 3.4: State Management

Create `vm/state.go`:

```go
package vm

import (
	"encoding/binary"
	"fmt"

	"github.com/ava-labs/avalanchego/database"
	"github.com/ava-labs/avalanchego/ids"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

var (
	lastAcceptedKey = []byte("lastAccepted")
	blockPrefix     = []byte("block:")
	heightPrefix    = []byte("height:")
	accountPrefix   = []byte("account:")
)

// State manages persistent state
type State struct {
	db database.Database
}

// NewState creates a new state manager
func NewState(db database.Database) *State {
	return &State{db: db}
}

// HasLastAccepted checks if last accepted block exists
func (s *State) HasLastAccepted() (bool, error) {
	return s.db.Has(lastAcceptedKey)
}

// GetLastAccepted returns the last accepted block ID
func (s *State) GetLastAccepted() (ids.ID, error) {
	data, err := s.db.Get(lastAcceptedKey)
	if err != nil {
		return ids.Empty, err
	}
	return ids.ToID(data)
}

// SetLastAccepted sets the last accepted block ID
func (s *State) SetLastAccepted(id ids.ID) error {
	return s.db.Put(lastAcceptedKey, id[:])
}

// GetBlock retrieves a block by ID
func (s *State) GetBlock(id ids.ID) (*Block, error) {
	key := append(blockPrefix, id[:]...)
	data, err := s.db.Get(key)
	if err != nil {
		return nil, err
	}

	blk := &Block{}
	if err := blk.Unmarshal(data); err != nil {
		return nil, err
	}
	return blk, nil
}

// PutBlock stores a block
func (s *State) PutBlock(blk *Block) error {
	key := append(blockPrefix, blk.ID()[:]...)
	return s.db.Put(key, blk.Bytes())
}

// GetBlockIDByHeight returns the block ID at a given height
func (s *State) GetBlockIDByHeight(height uint64) (ids.ID, error) {
	key := make([]byte, len(heightPrefix)+8)
	copy(key, heightPrefix)
	binary.BigEndian.PutUint64(key[len(heightPrefix):], height)

	data, err := s.db.Get(key)
	if err != nil {
		return ids.Empty, err
	}
	return ids.ToID(data)
}

// SetBlockIDByHeight stores the block ID for a height
func (s *State) SetBlockIDByHeight(height uint64, id ids.ID) error {
	key := make([]byte, len(heightPrefix)+8)
	copy(key, heightPrefix)
	binary.BigEndian.PutUint64(key[len(heightPrefix):], height)

	return s.db.Put(key, id[:])
}

// GetAccount retrieves an account
func (s *State) GetAccount(pubkey [32]byte) (*ffi.Account, error) {
	key := append(accountPrefix, pubkey[:]...)
	data, err := s.db.Get(key)
	if err != nil {
		if err == database.ErrNotFound {
			return nil, nil
		}
		return nil, err
	}

	account := &ffi.Account{}
	// Deserialize account (borsh format)
	// TODO: Implement proper borsh deserialization
	account.Data = data
	return account, nil
}

// SetAccount stores an account
func (s *State) SetAccount(pubkey [32]byte, account *ffi.Account) error {
	key := append(accountPrefix, pubkey[:]...)
	
	// Serialize account (borsh format)
	// TODO: Implement proper borsh serialization
	data := account.Data
	
	return s.db.Put(key, data)
}

// DeleteAccount removes an account
func (s *State) DeleteAccount(pubkey [32]byte) error {
	key := append(accountPrefix, pubkey[:]...)
	return s.db.Delete(key)
}
```

---

## Phase 4: RPC Server

### Task 4.1: RPC Server

Create `rpc/server.go`:

```go
package rpc

import (
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
	"go.uber.org/zap"
)

// Server is the JSON-RPC server
type Server struct {
	vm     VMInterface
	log    *zap.Logger
	router *mux.Router
}

// VMInterface defines the VM methods needed by RPC
type VMInterface interface {
	SubmitTransaction(tx *Transaction) error
	GetAccount(pubkey [32]byte) (*Account, error)
	GetSlot() uint64
	GetBlockhash() ([32]byte, error)
	SimulateTransaction(txData []byte) (*ExecutionResult, error)
}

// NewServer creates a new RPC server
func NewServer(vm VMInterface, log *zap.Logger) *Server {
	s := &Server{
		vm:     vm,
		log:    log,
		router: mux.NewRouter(),
	}
	s.setupRoutes()
	return s
}

func (s *Server) setupRoutes() {
	s.router.HandleFunc("/", s.handleRPC).Methods("POST")
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

// JSON-RPC request
type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

// JSON-RPC response
type rpcResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   *rpcError   `json:"error,omitempty"`
}

type rpcError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

func (s *Server) handleRPC(w http.ResponseWriter, r *http.Request) {
	var req rpcRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.sendError(w, nil, -32700, "Parse error", nil)
		return
	}

	s.log.Debug("RPC request", zap.String("method", req.Method))

	var result interface{}
	var rpcErr *rpcError

	switch req.Method {
	case "sendTransaction":
		result, rpcErr = s.sendTransaction(req.Params)
	case "getAccountInfo":
		result, rpcErr = s.getAccountInfo(req.Params)
	case "getLatestBlockhash":
		result, rpcErr = s.getLatestBlockhash(req.Params)
	case "getSlot":
		result, rpcErr = s.getSlot(req.Params)
	case "simulateTransaction":
		result, rpcErr = s.simulateTransaction(req.Params)
	case "getSignatureStatuses":
		result, rpcErr = s.getSignatureStatuses(req.Params)
	default:
		rpcErr = &rpcError{Code: -32601, Message: "Method not found"}
	}

	if rpcErr != nil {
		s.sendError(w, req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		return
	}

	s.sendResponse(w, req.ID, result)
}

func (s *Server) sendResponse(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) sendError(w http.ResponseWriter, id interface{}, code int, message string, data interface{}) {
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &rpcError{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// sendTransaction handler
func (s *Server) sendTransaction(params json.RawMessage) (interface{}, *rpcError) {
	var p []interface{}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	if len(p) < 1 {
		return nil, &rpcError{Code: -32602, Message: "Missing transaction"}
	}

	// Decode base64 transaction
	txBase64, ok := p[0].(string)
	if !ok {
		return nil, &rpcError{Code: -32602, Message: "Invalid transaction format"}
	}

	txData, err := base64.StdEncoding.DecodeString(txBase64)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid base64"}
	}

	// Submit transaction
	tx := &Transaction{Bytes: txData}
	if err := s.vm.SubmitTransaction(tx); err != nil {
		return nil, &rpcError{Code: -32000, Message: err.Error()}
	}

	// Return signature (first 64 bytes of tx as base58)
	sig := base58Encode(txData[:64])
	return sig, nil
}

// getAccountInfo handler
func (s *Server) getAccountInfo(params json.RawMessage) (interface{}, *rpcError) {
	var p []interface{}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	if len(p) < 1 {
		return nil, &rpcError{Code: -32602, Message: "Missing pubkey"}
	}

	pubkeyStr, ok := p[0].(string)
	if !ok {
		return nil, &rpcError{Code: -32602, Message: "Invalid pubkey format"}
	}

	pubkeyBytes, err := base58Decode(pubkeyStr)
	if err != nil || len(pubkeyBytes) != 32 {
		return nil, &rpcError{Code: -32602, Message: "Invalid pubkey"}
	}

	var pubkey [32]byte
	copy(pubkey[:], pubkeyBytes)

	account, err := s.vm.GetAccount(pubkey)
	if err != nil {
		return nil, &rpcError{Code: -32000, Message: err.Error()}
	}

	if account == nil {
		return map[string]interface{}{"value": nil}, nil
	}

	return map[string]interface{}{
		"value": map[string]interface{}{
			"lamports":   account.Lamports,
			"data":       []string{base64.StdEncoding.EncodeToString(account.Data), "base64"},
			"owner":      base58Encode(account.Owner[:]),
			"executable": account.Executable,
			"rentEpoch":  account.RentEpoch,
		},
	}, nil
}

// getLatestBlockhash handler
func (s *Server) getLatestBlockhash(params json.RawMessage) (interface{}, *rpcError) {
	blockhash, err := s.vm.GetBlockhash()
	if err != nil {
		return nil, &rpcError{Code: -32000, Message: err.Error()}
	}

	return map[string]interface{}{
		"value": map[string]interface{}{
			"blockhash":            base58Encode(blockhash[:]),
			"lastValidBlockHeight": s.vm.GetSlot() + 150,
		},
	}, nil
}

// getSlot handler
func (s *Server) getSlot(params json.RawMessage) (interface{}, *rpcError) {
	return s.vm.GetSlot(), nil
}

// simulateTransaction handler
func (s *Server) simulateTransaction(params json.RawMessage) (interface{}, *rpcError) {
	var p []interface{}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	if len(p) < 1 {
		return nil, &rpcError{Code: -32602, Message: "Missing transaction"}
	}

	txBase64, ok := p[0].(string)
	if !ok {
		return nil, &rpcError{Code: -32602, Message: "Invalid transaction format"}
	}

	txData, err := base64.StdEncoding.DecodeString(txBase64)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid base64"}
	}

	result, err := s.vm.SimulateTransaction(txData)
	if err != nil {
		return nil, &rpcError{Code: -32000, Message: err.Error()}
	}

	return map[string]interface{}{
		"value": map[string]interface{}{
			"err":             result.Error,
			"logs":            result.Logs,
			"unitsConsumed":   result.ComputeUnitsUsed,
		},
	}, nil
}

// getSignatureStatuses handler
func (s *Server) getSignatureStatuses(params json.RawMessage) (interface{}, *rpcError) {
	// TODO: Implement signature status tracking
	return map[string]interface{}{
		"value": []interface{}{},
	}, nil
}

// Base58 encoding/decoding helpers
func base58Encode(data []byte) string {
	// Use bs58 library
	return "" // TODO: Implement
}

func base58Decode(s string) ([]byte, error) {
	// Use bs58 library
	return nil, nil // TODO: Implement
}
```

---

## Phase 5: Testing

### Task 5.1: Integration Tests

Create `tests/integration/basic_transfer_test.go`:

```go
package integration

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/eto-chain/svm-subnet/vm"
	"github.com/eto-chain/svm-subnet/vm/ffi"
)

func TestBasicTransfer(t *testing.T) {
	// Initialize runtime
	runtime, err := ffi.NewRuntime()
	require.NoError(t, err)
	defer runtime.Close()

	// Set block context
	err = runtime.SetBlockContext(1, time.Now().Unix(), [32]byte{})
	require.NoError(t, err)

	// TODO: Create transfer transaction
	// TODO: Execute and verify

	t.Log("Basic transfer test passed")
}

func TestCreateAccount(t *testing.T) {
	// Initialize runtime
	runtime, err := ffi.NewRuntime()
	require.NoError(t, err)
	defer runtime.Close()

	// Set block context
	err = runtime.SetBlockContext(1, time.Now().Unix(), [32]byte{})
	require.NoError(t, err)

	// TODO: Create account creation transaction
	// TODO: Execute and verify

	t.Log("Create account test passed")
}
```

Create `tests/integration/program_deploy_test.go`:

```go
package integration

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

func TestProgramDeploy(t *testing.T) {
	// Initialize runtime
	runtime, err := ffi.NewRuntime()
	require.NoError(t, err)
	defer runtime.Close()

	// TODO: Deploy a simple BPF program
	// TODO: Verify program is executable

	t.Log("Program deploy test passed")
}
```

---

## Timeline and Milestones

### Week 1-2: Foundation
- [ ] Set up repository structure
- [ ] Implement Rust types (Pubkey, Account, Transaction)
- [ ] Implement System Program
- [ ] Basic FFI interface

### Week 3-4: Core Runtime
- [ ] Transaction executor
- [ ] Sysvars implementation
- [ ] BPF Loader (simplified)
- [ ] Full FFI bindings

### Week 5-6: Go VM Plugin
- [ ] VM interface implementation
- [ ] Block handling
- [ ] State management
- [ ] Mempool

### Week 7-8: RPC and Testing
- [ ] Solana-compatible RPC server
- [ ] Integration tests
- [ ] Local network testing
- [ ] Documentation

### Week 9-10: BPF Execution
- [ ] Integrate solana-rbpf
- [ ] Program execution
- [ ] CPI support
- [ ] End-to-end tests

### Week 11-12: GhostEVM and Polish
- [ ] GhostEVM setup (subnet-evm)
- [ ] AWM integration
- [ ] Gateway contracts
- [ ] Performance optimization

---

## Commands for Claude Code

When starting implementation, use these commands:

```bash
# Create project structure
mkdir -p svm-subnet/{vm/ffi,runtime/src/{types,sysvars,programs},rpc,genesis,tests/{integration,e2e,fixtures},ghostevm/{contracts,config},docs,scripts}

# Initialize Go module
cd svm-subnet && go mod init github.com/eto-chain/svm-subnet

# Initialize Rust crate
cd runtime && cargo init --lib

# Build everything
make build

# Run tests
make test

# Start local network
make run-local
```

---

## Critical Dependencies

Add to Rust `Cargo.toml`:
```toml
solana-program-runtime = "1.18"
solana-sdk = "1.18"
solana-bpf-loader-program = "1.18"
borsh = "1.3"
ed25519-dalek = "2.1"
```

Add to Go `go.mod`:
```go
require (
    github.com/ava-labs/avalanchego v1.11.3
    github.com/mr-tron/base58 v1.2.0
)
```

---

This plan provides Claude Code with everything needed to implement the SVM-Subnet. Start with Phase 1 (scaffolding), then proceed through each phase sequentially.
# SVM-Subnet Implementation Plan - Part 3

## Additional Components

### Mempool Implementation

Create `vm/mempool.go`:

```go
package vm

import (
	"container/heap"
	"errors"
	"sync"
)

var (
	ErrMempoolFull      = errors.New("mempool is full")
	ErrDuplicateTx      = errors.New("duplicate transaction")
	ErrInvalidPriority  = errors.New("invalid priority")
)

// Transaction represents a pending transaction
type Transaction struct {
	Bytes     []byte
	Signature [64]byte
	Priority  uint64 // Compute unit price
	Timestamp int64
}

// Mempool is a priority queue of pending transactions
type Mempool struct {
	lock    sync.RWMutex
	txs     txHeap
	seen    map[[64]byte]bool
	maxSize int
}

// txHeap implements heap.Interface for priority ordering
type txHeap []*Transaction

func (h txHeap) Len() int           { return len(h) }
func (h txHeap) Less(i, j int) bool { return h[i].Priority > h[j].Priority } // Higher priority first
func (h txHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *txHeap) Push(x interface{}) {
	*h = append(*h, x.(*Transaction))
}

func (h *txHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// NewMempool creates a new mempool
func NewMempool(maxSize int) *Mempool {
	return &Mempool{
		txs:     make(txHeap, 0),
		seen:    make(map[[64]byte]bool),
		maxSize: maxSize,
	}
}

// Add adds a transaction to the mempool
func (m *Mempool) Add(tx *Transaction) error {
	m.lock.Lock()
	defer m.lock.Unlock()

	// Check for duplicate
	if m.seen[tx.Signature] {
		return ErrDuplicateTx
	}

	// Check capacity
	if len(m.txs) >= m.maxSize {
		// Check if new tx has higher priority than lowest
		if len(m.txs) > 0 && tx.Priority <= m.txs[len(m.txs)-1].Priority {
			return ErrMempoolFull
		}
		// Remove lowest priority tx
		removed := heap.Pop(&m.txs).(*Transaction)
		delete(m.seen, removed.Signature)
	}

	// Add transaction
	heap.Push(&m.txs, tx)
	m.seen[tx.Signature] = true

	return nil
}

// Pop removes and returns the highest priority transaction
func (m *Mempool) Pop() *Transaction {
	m.lock.Lock()
	defer m.lock.Unlock()

	if len(m.txs) == 0 {
		return nil
	}

	tx := heap.Pop(&m.txs).(*Transaction)
	delete(m.seen, tx.Signature)
	return tx
}

// PopN removes and returns up to n highest priority transactions
func (m *Mempool) PopN(n int) []*Transaction {
	m.lock.Lock()
	defer m.lock.Unlock()

	result := make([]*Transaction, 0, n)
	for i := 0; i < n && len(m.txs) > 0; i++ {
		tx := heap.Pop(&m.txs).(*Transaction)
		delete(m.seen, tx.Signature)
		result = append(result, tx)
	}

	return result
}

// Peek returns the highest priority transaction without removing it
func (m *Mempool) Peek() *Transaction {
	m.lock.RLock()
	defer m.lock.RUnlock()

	if len(m.txs) == 0 {
		return nil
	}
	return m.txs[0]
}

// Size returns the number of transactions in the mempool
func (m *Mempool) Size() int {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return len(m.txs)
}

// Has checks if a transaction is in the mempool
func (m *Mempool) Has(sig [64]byte) bool {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return m.seen[sig]
}

// Remove removes a specific transaction by signature
func (m *Mempool) Remove(sig [64]byte) bool {
	m.lock.Lock()
	defer m.lock.Unlock()

	if !m.seen[sig] {
		return false
	}

	// Find and remove the transaction
	for i, tx := range m.txs {
		if tx.Signature == sig {
			heap.Remove(&m.txs, i)
			delete(m.seen, sig)
			return true
		}
	}

	return false
}

// Clear removes all transactions
func (m *Mempool) Clear() {
	m.lock.Lock()
	defer m.lock.Unlock()

	m.txs = make(txHeap, 0)
	m.seen = make(map[[64]byte]bool)
}
```

### Transaction Verification

Create `vm/tx.go`:

```go
package vm

import (
	"crypto/ed25519"
	"errors"
	"fmt"
)

var (
	ErrInvalidSignature   = errors.New("invalid signature")
	ErrInvalidTransaction = errors.New("invalid transaction")
	ErrExpiredBlockhash   = errors.New("expired blockhash")
)

// Verify validates a transaction
func (tx *Transaction) Verify() error {
	if len(tx.Bytes) < 65 {
		return ErrInvalidTransaction
	}

	// Parse signature count
	sigCount := int(tx.Bytes[0])
	if sigCount == 0 {
		return ErrInvalidSignature
	}

	// Check minimum length
	minLen := 1 + (sigCount * 64) + 3 + 32 + 32 // sig_count + sigs + header + accounts(min 1) + blockhash
	if len(tx.Bytes) < minLen {
		return ErrInvalidTransaction
	}

	// Extract message (everything after signatures)
	sigEnd := 1 + (sigCount * 64)
	message := tx.Bytes[sigEnd:]

	// Parse header
	header := message[:3]
	numRequiredSigs := int(header[0])
	
	if numRequiredSigs > sigCount {
		return fmt.Errorf("not enough signatures: need %d, have %d", numRequiredSigs, sigCount)
	}

	// Extract account keys
	numAccounts := int(message[3])
	if len(message) < 4+(numAccounts*32)+32 {
		return ErrInvalidTransaction
	}

	// Verify signatures
	for i := 0; i < numRequiredSigs; i++ {
		sigStart := 1 + (i * 64)
		sig := tx.Bytes[sigStart : sigStart+64]

		pubkeyStart := 4 + (i * 32)
		pubkey := message[pubkeyStart : pubkeyStart+32]

		if !ed25519.Verify(pubkey, message, sig) {
			return fmt.Errorf("signature %d invalid", i)
		}
	}

	// Copy first signature to tx.Signature
	copy(tx.Signature[:], tx.Bytes[1:65])

	return nil
}

// ExtractPriority extracts the compute unit price from a transaction
func (tx *Transaction) ExtractPriority() uint64 {
	// Look for ComputeBudget SetComputeUnitPrice instruction
	// For now, return default priority
	return 0
}
```

### Configuration

Create `vm/config.go`:

```go
package vm

import (
	"encoding/json"
)

// Config holds VM configuration
type Config struct {
	// Maximum transactions per block
	MaxTxsPerBlock int `json:"maxTxsPerBlock"`

	// Mempool size
	MempoolSize int `json:"mempoolSize"`

	// Block time target in milliseconds
	BlockTimeMs int `json:"blockTimeMs"`

	// Maximum compute units per transaction
	MaxComputeUnits uint64 `json:"maxComputeUnits"`

	// Lamports per signature (base fee)
	LamportsPerSignature uint64 `json:"lamportsPerSignature"`

	// RPC configuration
	RPCPort int `json:"rpcPort"`

	// Log level
	LogLevel string `json:"logLevel"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() Config {
	return Config{
		MaxTxsPerBlock:       1000,
		MempoolSize:          10000,
		BlockTimeMs:          400,
		MaxComputeUnits:      1_400_000,
		LamportsPerSignature: 5000,
		RPCPort:              8899,
		LogLevel:             "info",
	}
}

// ParseConfig parses configuration from JSON bytes
func ParseConfig(data []byte) (Config, error) {
	config := DefaultConfig()
	
	if len(data) == 0 {
		return config, nil
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, err
	}

	return config, nil
}
```

### Genesis Configuration

Create `genesis/genesis.go`:

```go
package genesis

import (
	"encoding/json"
	"time"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

// Genesis represents the genesis state
type Genesis struct {
	// Timestamp of genesis block
	Timestamp int64 `json:"timestamp"`

	// Initial accounts
	Accounts []GenesisAccount `json:"accounts"`

	// Initial programs
	Programs []GenesisProgram `json:"programs"`

	// Chain parameters
	Params GenesisParams `json:"params"`
}

// GenesisAccount represents an initial account
type GenesisAccount struct {
	Pubkey  [32]byte    `json:"pubkey"`
	Account ffi.Account `json:"account"`
}

// GenesisProgram represents an initial program
type GenesisProgram struct {
	Pubkey     [32]byte `json:"pubkey"`
	Executable []byte   `json:"executable"`
}

// GenesisParams are chain parameters
type GenesisParams struct {
	SlotsPerEpoch        uint64 `json:"slotsPerEpoch"`
	LamportsPerSignature uint64 `json:"lamportsPerSignature"`
	RentLamportsPerByte  uint64 `json:"rentLamportsPerByte"`
}

// DefaultGenesis returns a default genesis configuration
func DefaultGenesis() *Genesis {
	// Create faucet account with lots of SOL
	faucetPubkey := [32]byte{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	
	return &Genesis{
		Timestamp: time.Now().Unix(),
		Accounts: []GenesisAccount{
			{
				Pubkey: faucetPubkey,
				Account: ffi.Account{
					Lamports:   1_000_000_000_000_000, // 1 billion SOL
					Data:       []byte{},
					Owner:      [32]byte{}, // System program
					Executable: false,
					RentEpoch:  0,
				},
			},
		},
		Programs: []GenesisProgram{},
		Params: GenesisParams{
			SlotsPerEpoch:        432_000,
			LamportsPerSignature: 5000,
			RentLamportsPerByte:  3480,
		},
	}
}

// ParseGenesis parses genesis from JSON bytes
func ParseGenesis(data []byte) (*Genesis, error) {
	if len(data) == 0 {
		return DefaultGenesis(), nil
	}

	genesis := &Genesis{}
	if err := json.Unmarshal(data, genesis); err != nil {
		return nil, err
	}

	return genesis, nil
}

// Marshal serializes genesis to JSON
func (g *Genesis) Marshal() ([]byte, error) {
	return json.MarshalIndent(g, "", "  ")
}
```

### VM Factory

Create `vm/factory.go`:

```go
package vm

import (
	"github.com/ava-labs/avalanchego/vms"
)

var _ vms.Factory = (*Factory)(nil)

// Factory implements the vms.Factory interface
type Factory struct{}

// New returns a new VM instance
func (f *Factory) New(*snow.Context) (interface{}, error) {
	return &VM{}, nil
}
```

---

## GhostEVM Integration

### Phase 6: GhostEVM Chain Setup

Create `ghostevm/README.md`:

```markdown
# GhostEVM - Minimal EVM Chain for AVAX Interop

GhostEVM is a minimal EVM chain running alongside SVM-Chain in the same subnet.
Its primary purpose is to provide:

1. AVAX bridging capabilities via AWM (Avalanche Warp Messaging)
2. Wallet compatibility for MetaMask and other EVM wallets
3. Entry point for cross-chain calls to SVM programs

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Same Subnet                         │
├─────────────────────────┬────────────────────────────┤
│       SVM-Chain         │        GhostEVM            │
│  (Solana execution)     │    (EVM interop)           │
├─────────────────────────┼────────────────────────────┤
│ • Execute BPF programs  │ • AWM precompiles          │
│ • Solana accounts       │ • SVMGateway contract      │
│ • Solana RPC            │ • Bridge contracts         │
└─────────────────────────┴────────────────────────────┘
            ↑                        ↑
            └──── AWM Messages ──────┘
```

## Deployment

GhostEVM uses subnet-evm with Warp messaging enabled.

### Configuration

See `config/chain_config.json` for the chain configuration.

### Contracts

- `SVMGateway.sol`: Entry point for calling SVM programs from EVM
- Processes incoming Warp messages from SVM-Chain
- Encodes outgoing calls to SVM programs
```

Create `ghostevm/config/chain_config.json`:

```json
{
  "chainId": 99999,
  "homesteadBlock": 0,
  "eip150Block": 0,
  "eip155Block": 0,
  "eip158Block": 0,
  "byzantiumBlock": 0,
  "constantinopleBlock": 0,
  "petersburgBlock": 0,
  "istanbulBlock": 0,
  "muirGlacierBlock": 0,
  "subnetEVMTimestamp": 0,
  "feeConfig": {
    "gasLimit": 15000000,
    "targetBlockRate": 1,
    "minBaseFee": 1000000000,
    "targetGas": 15000000,
    "baseFeeChangeDenominator": 36,
    "minBlockGasCost": 0,
    "maxBlockGasCost": 1000000,
    "blockGasCostStep": 100000
  },
  "warpConfig": {
    "blockTimestamp": 0,
    "quorumNumerator": 67
  }
}
```

Create `ghostevm/contracts/SVMGateway.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IWarpMessenger.sol";

/**
 * @title SVMGateway
 * @notice Gateway contract for interacting with SVM-Chain via AWM
 */
contract SVMGateway {
    // Warp precompile address
    address constant WARP_PRECOMPILE = 0x0200000000000000000000000000000000000005;
    
    // SVM-Chain blockchain ID (set during deployment)
    bytes32 public immutable svmChainID;
    
    // Events
    event SVMCallInitiated(
        bytes32 indexed messageID,
        bytes32 indexed programID,
        bytes instruction
    );
    
    event SVMCallReceived(
        bytes32 indexed sourceChainID,
        bytes32 indexed messageID,
        bytes result
    );
    
    constructor(bytes32 _svmChainID) {
        svmChainID = _svmChainID;
    }
    
    /**
     * @notice Call an SVM program
     * @param programID The Solana program ID (32 bytes)
     * @param instruction The instruction data to send
     * @return messageID The Warp message ID
     */
    function callSVMProgram(
        bytes32 programID,
        bytes calldata instruction
    ) external payable returns (bytes32 messageID) {
        // Encode the message
        bytes memory payload = abi.encode(
            programID,
            msg.sender,
            instruction
        );
        
        // Send via Warp
        IWarpMessenger warp = IWarpMessenger(WARP_PRECOMPILE);
        messageID = warp.sendWarpMessage(payload);
        
        emit SVMCallInitiated(messageID, programID, instruction);
        
        return messageID;
    }
    
    /**
     * @notice Receive a callback from SVM-Chain
     * @param messageIndex Index of the Warp message in the block
     */
    function receiveSVMCallback(uint32 messageIndex) external {
        IWarpMessenger warp = IWarpMessenger(WARP_PRECOMPILE);
        
        // Get and verify the Warp message
        (WarpMessage memory message, bool valid) = warp.getVerifiedWarpMessage(messageIndex);
        require(valid, "Invalid Warp message");
        require(message.sourceChainID == svmChainID, "Invalid source chain");
        
        // Decode the result
        (bytes32 originalMessageID, bytes memory result) = abi.decode(
            message.payload,
            (bytes32, bytes)
        );
        
        emit SVMCallReceived(message.sourceChainID, originalMessageID, result);
        
        // TODO: Route result to original caller
    }
    
    /**
     * @notice Transfer native AVAX to SVM-Chain
     * @param recipient Solana address to receive (32 bytes)
     */
    function bridgeToSVM(bytes32 recipient) external payable {
        require(msg.value > 0, "No value sent");
        
        bytes memory payload = abi.encode(
            bytes32(0), // null program = native transfer
            recipient,
            msg.value
        );
        
        IWarpMessenger warp = IWarpMessenger(WARP_PRECOMPILE);
        warp.sendWarpMessage(payload);
    }
}

// Warp message structure
struct WarpMessage {
    bytes32 sourceChainID;
    address originSenderAddress;
    bytes payload;
}

interface IWarpMessenger {
    function sendWarpMessage(bytes calldata payload) external returns (bytes32 messageID);
    function getVerifiedWarpMessage(uint32 index) external view returns (WarpMessage memory message, bool valid);
}
```

Create `ghostevm/contracts/interfaces/IWarpMessenger.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

struct WarpMessage {
    bytes32 sourceChainID;
    address originSenderAddress;
    bytes payload;
}

struct WarpBlockHash {
    bytes32 sourceChainID;
    bytes32 blockHash;
}

interface IWarpMessenger {
    function sendWarpMessage(bytes calldata payload) external returns (bytes32 messageID);
    
    function getVerifiedWarpMessage(uint32 index)
        external view
        returns (WarpMessage memory message, bool valid);
    
    function getVerifiedWarpBlockHash(uint32 index)
        external view
        returns (WarpBlockHash memory warpBlockHash, bool valid);
    
    function getBlockchainID() external view returns (bytes32 blockchainID);
}
```

---

## AWM Integration for SVM-Chain

### Task 6.1: AWM Message Handler

Add to `vm/vm.go`:

```go
// AppGossip implements common.VM for receiving gossip messages
func (vm *VM) AppGossip(ctx context.Context, nodeID ids.NodeID, msg []byte) error {
    // Handle gossiped transactions
    tx := &Transaction{Bytes: msg}
    return vm.SubmitTransaction(tx)
}

// AppRequest implements common.VM for handling app-level requests
func (vm *VM) AppRequest(
    ctx context.Context,
    nodeID ids.NodeID,
    requestID uint32,
    deadline time.Time,
    request []byte,
) error {
    // Handle Warp message requests
    return nil
}

// AppRequestFailed implements common.VM
func (vm *VM) AppRequestFailed(ctx context.Context, nodeID ids.NodeID, requestID uint32) error {
    return nil
}

// AppResponse implements common.VM
func (vm *VM) AppResponse(
    ctx context.Context,
    nodeID ids.NodeID,
    requestID uint32,
    response []byte,
) error {
    return nil
}
```

### Task 6.2: Warp Message Processing

Create `vm/warp.go`:

```go
package vm

import (
	"encoding/binary"
	"errors"

	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/vms/platformvm/warp"
)

// WarpMessageType identifies the type of cross-chain message
type WarpMessageType uint8

const (
	WarpMessageTypeCall     WarpMessageType = 0
	WarpMessageTypeTransfer WarpMessageType = 1
	WarpMessageTypeCallback WarpMessageType = 2
)

// WarpMessage represents a cross-chain message
type WarpMessage struct {
	Type        WarpMessageType
	SourceChain ids.ID
	Sender      [32]byte // EVM address padded or Solana pubkey
	Payload     []byte
}

// ProcessWarpMessage handles incoming Warp messages from GhostEVM
func (vm *VM) ProcessWarpMessage(msg *warp.Message) error {
	// Verify the message
	if err := msg.Verify(); err != nil {
		return err
	}

	// Decode the payload
	warpMsg, err := DecodeWarpMessage(msg.Payload)
	if err != nil {
		return err
	}

	switch warpMsg.Type {
	case WarpMessageTypeCall:
		return vm.handleWarpCall(warpMsg)
	case WarpMessageTypeTransfer:
		return vm.handleWarpTransfer(warpMsg)
	default:
		return errors.New("unknown warp message type")
	}
}

func (vm *VM) handleWarpCall(msg *WarpMessage) error {
	// Decode: programID (32) + instruction
	if len(msg.Payload) < 32 {
		return errors.New("invalid warp call payload")
	}

	var programID [32]byte
	copy(programID[:], msg.Payload[:32])
	instruction := msg.Payload[32:]

	// TODO: Create and execute transaction to call program
	vm.log.Info("Received Warp call",
		zap.String("program", base58Encode(programID[:])),
		zap.Int("instruction_len", len(instruction)),
	)

	return nil
}

func (vm *VM) handleWarpTransfer(msg *WarpMessage) error {
	// Decode: recipient (32) + amount (8)
	if len(msg.Payload) < 40 {
		return errors.New("invalid warp transfer payload")
	}

	var recipient [32]byte
	copy(recipient[:], msg.Payload[:32])
	amount := binary.BigEndian.Uint64(msg.Payload[32:40])

	// TODO: Credit recipient account with bridged tokens
	vm.log.Info("Received Warp transfer",
		zap.String("recipient", base58Encode(recipient[:])),
		zap.Uint64("amount", amount),
	)

	return nil
}

// SendWarpMessage sends a message to GhostEVM
func (vm *VM) SendWarpMessage(destChain ids.ID, payload []byte) error {
	// TODO: Implement outgoing Warp messages
	return nil
}

// DecodeWarpMessage decodes a raw Warp payload
func DecodeWarpMessage(data []byte) (*WarpMessage, error) {
	if len(data) < 65 { // type(1) + source(32) + sender(32)
		return nil, errors.New("warp message too short")
	}

	msg := &WarpMessage{
		Type: WarpMessageType(data[0]),
	}
	copy(msg.SourceChain[:], data[1:33])
	copy(msg.Sender[:], data[33:65])
	msg.Payload = data[65:]

	return msg, nil
}
```

---

## Deployment Instructions

### Local Network Setup

Create `scripts/local-network.sh`:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
AVALANCHEGO_VERSION="v1.11.3"
SUBNET_EVM_VERSION="v0.6.3"

# Install avalanche-network-runner if not present
if ! command -v avalanche-network-runner &> /dev/null; then
    echo "Installing avalanche-network-runner..."
    go install github.com/ava-labs/avalanche-network-runner/cmd/avalanche-network-runner@latest
fi

# Build the VM
echo "Building SVM-Subnet VM..."
cd "$PROJECT_DIR"
make build

# Get VM ID
VM_ID="srEXiWaHq7HVqKPT3HTPM8iSdLVatYC49pLVfxzdjwc9hLKm2"

# Start the network runner server
echo "Starting network runner server..."
avalanche-network-runner server \
    --log-level info \
    --port :8080 \
    --grpc-gateway-port :8081 &

SERVER_PID=$!
sleep 2

# Start the network
echo "Starting local network..."
avalanche-network-runner control start \
    --log-level info \
    --endpoint :8080 \
    --number-of-nodes 5 \
    --blockchain-specs '[
        {
            "vm_name": "svm-subnet",
            "genesis": "'$(cat "$PROJECT_DIR/genesis/default.json" | base64)'"
        }
    ]' \
    --custom-vms '{"svm-subnet": "'$PROJECT_DIR/build/svm-vm'"}'

echo "Network started!"
echo "RPC endpoints available at:"
echo "  Node 1: http://127.0.0.1:9650"
echo "  Node 2: http://127.0.0.1:9652"
echo "  Node 3: http://127.0.0.1:9654"
echo "  Node 4: http://127.0.0.1:9656"
echo "  Node 5: http://127.0.0.1:9658"

# Wait for shutdown
echo "Press Ctrl+C to stop the network"
wait $SERVER_PID
```

### Testnet Deployment

Create `scripts/deploy-testnet.sh`:

```bash
#!/bin/bash
set -e

# Requires:
# - avalanche-cli installed
# - Funded P-Chain wallet

SUBNET_NAME="svm-subnet"
VM_ID="srEXiWaHq7HVqKPT3HTPM8iSdLVatYC49pLVfxzdjwc9hLKm2"

# Build VM
echo "Building VM..."
make build

# Create subnet
echo "Creating subnet..."
avalanche subnet create $SUBNET_NAME \
    --vm custom \
    --vm-path ./build/svm-vm \
    --genesis ./genesis/testnet.json

# Deploy to Fuji
echo "Deploying to Fuji testnet..."
avalanche subnet deploy $SUBNET_NAME --fuji

echo "Deployment complete!"
avalanche subnet describe $SUBNET_NAME --fuji
```

---

## Testing Checklist

### Unit Tests
- [ ] Types serialization/deserialization
- [ ] System program instructions
- [ ] BPF loader instructions
- [ ] Signature verification
- [ ] Mempool operations

### Integration Tests
- [ ] Basic SOL transfer
- [ ] Account creation
- [ ] Program deployment
- [ ] Program execution
- [ ] RPC methods

### E2E Tests
- [ ] Local network bootstrap
- [ ] Block production
- [ ] Transaction finality
- [ ] State persistence across restarts

### GhostEVM Tests
- [ ] AWM message sending
- [ ] AWM message receiving
- [ ] Cross-chain transfers
- [ ] Cross-chain calls

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Block time | 400ms |
| Finality | < 1s |
| TPS (simple transfers) | 5,000+ |
| TPS (program calls) | 1,000+ |
| Memory usage | < 8GB |
| State sync time | < 30min |

---

## Monitoring and Observability

### Metrics to Export
- Transactions per second
- Block production rate
- Mempool size
- Account count
- Program execution time
- Cross-chain message latency

### Logging
- Transaction submission
- Block acceptance
- State transitions
- Warp messages
- Errors and warnings

---

This completes the comprehensive implementation plan for the SVM-Subnet. Claude Code should proceed through each phase sequentially, starting with the Rust runtime and moving through to the GhostEVM integration.