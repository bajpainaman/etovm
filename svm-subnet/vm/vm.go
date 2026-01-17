package vm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"sync"
	"time"

	"github.com/ava-labs/avalanchego/api/health"
	"github.com/ava-labs/avalanchego/database"
	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/snow"
	"github.com/ava-labs/avalanchego/snow/consensus/snowman"
	"github.com/ava-labs/avalanchego/snow/engine/common"
	"github.com/ava-labs/avalanchego/snow/engine/snowman/block"
	"github.com/ava-labs/avalanchego/utils/logging"
	"github.com/ava-labs/avalanchego/version"
	"github.com/eto-chain/svm-subnet/vm/ffi"
)

var (
	_ block.ChainVM                      = (*VM)(nil)
	_ block.BuildBlockWithContextChainVM = (*VM)(nil)
	_ health.Checker                     = (*VM)(nil)
)

const (
	Name        = "svm"
	VersionStr  = "0.1.0"
)

// VM implements the Snowman ChainVM interface for SVM execution
type VM struct {
	ctx    *snow.Context
	db     database.Database
	config *Config
	log    logging.Logger

	// Runtime (legacy - for single tx execution)
	runtime *ffi.Runtime

	// Sealevel parallel executor (for batch execution)
	sealevel *ffi.SealevelExecutor

	// QMDB parallel executor (block execution with merkle roots)
	qmdb *ffi.QMDBExecutor

	// Turbo executor (11M+ TPS delta mode)
	turbo *ffi.TurboExecutor

	// Hybrid executor (SVM + EVM)
	hybrid *ffi.HybridExecutor

	// State
	state     *State
	preferred ids.ID
	lastBlock *Block

	// Mempool
	mempool *Mempool

	// App sender for engine notifications
	appSender common.AppSender

	// Notification channel for pending transactions
	pendingTxs chan struct{}

	// Sync
	mu sync.RWMutex
}

// NewVM creates a new VM instance
func NewVM() *VM {
	return &VM{}
}

// Initialize implements block.ChainVM
func (vm *VM) Initialize(
	ctx context.Context,
	snowCtx *snow.Context,
	db database.Database,
	genesisBytes []byte,
	upgradeBytes []byte,
	configBytes []byte,
	_ []*common.Fx,
	appSender common.AppSender,
) error {
	vm.ctx = snowCtx
	vm.db = db
	vm.appSender = appSender
	vm.log = snowCtx.Log

	// Debug: very early log to confirm Initialize is called
	vm.log.Info("=== SVM VM Initialize started ===")
	vm.log.Info(fmt.Sprintf("genesisBytes len=%d, configBytes len=%d", len(genesisBytes), len(configBytes)))

	// Parse config
	config := DefaultConfig()
	if len(configBytes) > 0 {
		if err := json.Unmarshal(configBytes, config); err != nil {
			return fmt.Errorf("failed to parse config: %w", err)
		}
	}
	vm.config = config

	// Initialize runtime (legacy - for single tx execution)
	vm.log.Info("initializing SVM runtime...")
	runtime, err := ffi.NewRuntime()
	if err != nil {
		vm.log.Error(fmt.Sprintf("SVM runtime init failed: %v", err))
		return fmt.Errorf("failed to initialize SVM runtime: %w", err)
	}
	vm.runtime = runtime
	vm.log.Info("SVM runtime initialized successfully")

	// Initialize Sealevel parallel executor
	vm.log.Info("initializing Sealevel parallel executor...")
	sealevel, err := ffi.NewSealevelExecutor(0) // 0 = use all CPUs
	if err != nil {
		vm.log.Error(fmt.Sprintf("Sealevel init failed: %v", err))
		return fmt.Errorf("failed to initialize Sealevel executor: %w", err)
	}
	vm.sealevel = sealevel
	sealevelStats, _ := sealevel.GetStats()
	if sealevelStats != nil {
		vm.log.Info(fmt.Sprintf("Sealevel initialized with %d threads", sealevelStats.NumThreads))
	}

	// Initialize QMDB parallel executor (for block-level execution with merkle roots)
	vm.log.Info("initializing QMDB parallel executor...")
	qmdb, err := ffi.NewQMDBExecutor(0) // 0 = use all CPUs
	if err != nil {
		vm.log.Error(fmt.Sprintf("QMDB init failed: %v", err))
		return fmt.Errorf("failed to initialize QMDB executor: %w", err)
	}
	vm.qmdb = qmdb
	qmdbStats, _ := qmdb.GetStats()
	if qmdbStats != nil {
		vm.log.Info(fmt.Sprintf("QMDB initialized with %d threads", qmdbStats.NumThreads))
	}

	// Initialize Turbo executor (11M+ TPS delta mode)
	vm.log.Info("initializing Turbo executor (delta mode)...")
	turbo, err := ffi.NewTurboExecutor(0, config.VerifySignatures) // 0 = use all CPUs
	if err != nil {
		vm.log.Error(fmt.Sprintf("Turbo init failed: %v", err))
		return fmt.Errorf("failed to initialize Turbo executor: %w", err)
	}
	vm.turbo = turbo
	turboStats, _ := turbo.GetStats()
	if turboStats != nil {
		vm.log.Info(fmt.Sprintf("Turbo executor initialized with %d threads (11M+ TPS delta mode)", turboStats.NumThreads))
	}

	// Initialize hybrid executor (SVM + EVM)
	vm.log.Info("initializing hybrid executor...")
	hybrid, err := ffi.NewHybridExecutor(config.ChainID)
	if err != nil {
		vm.log.Error(fmt.Sprintf("hybrid executor init failed: %v", err))
		return fmt.Errorf("failed to initialize hybrid executor: %w", err)
	}
	vm.hybrid = hybrid
	vm.log.Info("hybrid executor initialized successfully")

	// Initialize state
	vm.state = NewState(db)

	// Initialize mempool
	vm.mempool = NewMempool(config.MempoolSize)

	// Initialize notification channel (buffered to avoid blocking)
	vm.pendingTxs = make(chan struct{}, 1)

	// Parse genesis
	var genesis Genesis
	if err := json.Unmarshal(genesisBytes, &genesis); err != nil {
		return fmt.Errorf("failed to parse genesis: %w", err)
	}
	vm.log.Info(fmt.Sprintf("parsed genesis: timestamp=%d, accounts=%d, chainId=%d", genesis.Timestamp, len(genesis.Accounts), genesis.ChainID))
	vm.log.Info(fmt.Sprintf("raw genesis bytes length: %d", len(genesisBytes)))
	if len(genesisBytes) < 500 {
		vm.log.Info(fmt.Sprintf("genesis bytes: %s", string(genesisBytes)))
	}

	// Check if we have existing state
	lastAccepted, err := vm.state.GetLastAccepted()
	if err == database.ErrNotFound {
		// Initialize from genesis
		genesisBlock, err := vm.initFromGenesis(&genesis)
		if err != nil {
			return fmt.Errorf("failed to initialize from genesis: %w", err)
		}
		vm.lastBlock = genesisBlock
		vm.preferred = genesisBlock.ID()
		vm.log.Info(fmt.Sprintf("initialized from genesis blockID=%s", genesisBlock.ID().String()))
	} else if err != nil {
		return fmt.Errorf("failed to get last accepted: %w", err)
	} else {
		// Load existing state
		lastBlock, err := vm.state.GetBlock(lastAccepted)
		if err != nil {
			return fmt.Errorf("failed to get last block: %w", err)
		}
		vm.lastBlock = lastBlock
		vm.preferred = lastAccepted
		vm.log.Info(fmt.Sprintf("loaded existing state blockID=%s height=%d", lastAccepted.String(), lastBlock.Height()))
	}

	// Final debug log
	vm.log.Info(fmt.Sprintf("=== SVM VM Initialize completed successfully, lastBlock timestamp=%s (unix=%d) ===",
		vm.lastBlock.timestamp.String(), vm.lastBlock.timestamp.Unix()))

	return nil
}

func (vm *VM) initFromGenesis(genesis *Genesis) (*Block, error) {
	// Ensure genesis timestamp is valid (non-zero)
	// Protobuf requires a non-zero timestamp
	genesisTime := time.Unix(genesis.Timestamp, 0)
	if genesis.Timestamp <= 0 {
		// Use a reasonable default if not specified
		genesisTime = time.Now()
	}

	vm.log.Info(fmt.Sprintf("creating genesis block with timestamp=%d (%s), isZero=%v", genesis.Timestamp, genesisTime.String(), genesisTime.IsZero()))

	// Create genesis block
	genesisBlock := &Block{
		vm:           vm,
		id:           ids.Empty,
		parentID:     ids.Empty,
		height:       0,
		timestamp:    genesisTime,
		transactions: nil,
		stateRoot:    [32]byte{},
	}

	// Initialize genesis accounts
	for _, account := range genesis.Accounts {
		if err := vm.state.SetAccount(account.Pubkey, &AccountState{
			Lamports:   account.Lamports,
			Data:       account.Data,
			Owner:      account.Owner,
			Executable: account.Executable,
		}); err != nil {
			return nil, fmt.Errorf("failed to set genesis account: %w", err)
		}

		// Also load into hybrid executor
		if vm.hybrid != nil {
			if err := vm.hybrid.LoadAccount(account.Pubkey, account.Lamports, account.Data, account.Owner, account.Executable); err != nil {
				vm.log.Warn(fmt.Sprintf("failed to load account into hybrid executor: %v", err))
			}
		}

		// Also load into turbo executor (for 11M+ TPS delta mode)
		if vm.turbo != nil {
			if err := vm.turbo.LoadAccount(account.Pubkey, account.Lamports, account.Data, account.Owner, account.Executable); err != nil {
				vm.log.Warn(fmt.Sprintf("failed to load account into turbo executor: %v", err))
			}
		}
	}

	// Calculate genesis block ID
	genesisBlock.id = genesisBlock.calculateID()

	vm.log.Info(fmt.Sprintf("genesis block created: id=%s, height=%d, timestamp=%s, tsUnix=%d",
		genesisBlock.id.String(), genesisBlock.height, genesisBlock.timestamp.String(), genesisBlock.timestamp.Unix()))

	// Save genesis block
	if err := vm.state.PutBlock(genesisBlock); err != nil {
		return nil, fmt.Errorf("failed to save genesis block: %w", err)
	}

	if err := vm.state.SetLastAccepted(genesisBlock.ID()); err != nil {
		return nil, fmt.Errorf("failed to set last accepted: %w", err)
	}

	return genesisBlock, nil
}

// Shutdown implements block.ChainVM
func (vm *VM) Shutdown(ctx context.Context) error {
	vm.log.Info("shutting down VM")

	if vm.runtime != nil {
		vm.runtime.Close()
	}

	if vm.sealevel != nil {
		vm.sealevel.Close()
	}

	if vm.qmdb != nil {
		vm.qmdb.Close()
	}

	if vm.turbo != nil {
		vm.turbo.Close()
	}

	if vm.hybrid != nil {
		vm.hybrid.Close()
	}

	return vm.db.Close()
}

// SetState implements block.ChainVM
func (vm *VM) SetState(ctx context.Context, state snow.State) error {
	switch state {
	case snow.Bootstrapping:
		vm.log.Info("entering bootstrapping state")
	case snow.NormalOp:
		vm.log.Info("entering normal operation state")
	}
	return nil
}

// CreateHandlers implements common.VM
func (vm *VM) CreateHandlers(ctx context.Context) (map[string]http.Handler, error) {
	// Create unified RPC handler for extensions
	handler := vm.createRPCHandler()
	return map[string]http.Handler{
		"/rpc":    handler,
		"/solana": handler,
		"/evm":    handler,
	}, nil
}

// NewHTTPHandler implements common.VM
func (vm *VM) NewHTTPHandler(ctx context.Context) (http.Handler, error) {
	return vm.createRPCHandler(), nil
}

// createRPCHandler creates the unified JSON-RPC handler
func (vm *VM) createRPCHandler() http.Handler {
	return &rpcHandler{vm: vm}
}

// rpcHandler implements http.Handler for JSON-RPC
type rpcHandler struct {
	vm *VM
}

// Version implements common.VM
func (vm *VM) Version(ctx context.Context) (string, error) {
	return VersionStr, nil
}

// WaitForEvent implements common.VM
func (vm *VM) WaitForEvent(ctx context.Context) (common.Message, error) {
	// Check if we already have pending transactions
	vm.mu.RLock()
	hasPending := vm.mempool.Len() > 0
	vm.mu.RUnlock()

	if hasPending {
		return common.PendingTxs, nil
	}

	// Wait for new transactions or context cancellation
	select {
	case <-vm.pendingTxs:
		return common.PendingTxs, nil
	case <-ctx.Done():
		return 0, ctx.Err()
	}
}

// BuildBlock implements block.ChainVM
func (vm *VM) BuildBlock(ctx context.Context) (snowman.Block, error) {
	return vm.BuildBlockWithContext(ctx, nil)
}

// BuildBlockWithContext implements block.BuildBlockWithContextChainVM
func (vm *VM) BuildBlockWithContext(ctx context.Context, blockCtx *block.Context) (snowman.Block, error) {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	// Get transactions from mempool
	txs := vm.mempool.PopAll(vm.config.MaxTxsPerBlock)
	if len(txs) == 0 {
		return nil, errors.New("no transactions to include in block")
	}

	parent, err := vm.state.GetBlock(vm.preferred)
	if err != nil {
		return nil, fmt.Errorf("failed to get parent block: %w", err)
	}

	// Create new block
	newBlock := &Block{
		vm:           vm,
		parentID:     parent.ID(),
		height:       parent.Height() + 1,
		timestamp:    time.Now(),
		transactions: txs,
	}

	// Prepare transaction data for block execution
	txDataList := make([][]byte, len(txs))
	for i, tx := range txs {
		txDataList[i] = tx.Bytes()
	}

	blockhash := parent.ID()

	// Use Turbo executor for 11M+ TPS delta execution (default for Avalanche L1)
	if vm.config.UseTurboMode && vm.turbo != nil {
		// Set block context in Turbo executor
		if err := vm.turbo.SetBlockContext(newBlock.height, newBlock.timestamp.Unix(), blockhash[:]); err != nil {
			return nil, fmt.Errorf("failed to set turbo block context: %w", err)
		}

		// Execute block using Turbo delta mode (11M+ TPS)
		turboResult, err := vm.turbo.ExecuteBlockDelta(newBlock.height, txDataList)
		if err != nil {
			vm.log.Warn(fmt.Sprintf("turbo block execution failed: %v", err))
			return nil, fmt.Errorf("turbo block execution failed: %w", err)
		}

		// In delta mode, all parsed transactions succeed (invalid ones fail silently)
		// Mark successful transactions
		executedTxs := make([]*Transaction, 0, turboResult.Successful)
		successCount := int(turboResult.Successful)
		for i := 0; i < len(txs) && i < successCount; i++ {
			txs[i].Result = &ffi.ExecutionResult{
				Success:          true,
				ComputeUnitsUsed: turboResult.TotalComputeUnits / uint64(successCount),
				Fee:              turboResult.TotalFees / uint64(successCount),
			}
			executedTxs = append(executedTxs, txs[i])
		}

		if len(executedTxs) == 0 {
			return nil, errors.New("no successful transactions in turbo mode")
		}

		newBlock.transactions = executedTxs
		newBlock.stateRoot = turboResult.MerkleRoot
		newBlock.id = newBlock.calculateID()

		tps := turboResult.TPS()
		vm.log.Info(fmt.Sprintf("TURBO built block id=%s height=%d txCount=%d successful=%d failed=%d verify_fail=%d time=%dus (%.0f TPS) stateRoot=%x",
			newBlock.ID().String(), newBlock.Height(), len(txs),
			turboResult.Successful, turboResult.Failed, turboResult.VerificationFailures,
			turboResult.ExecutionTimeUs, tps, turboResult.MerkleRoot[:8]))
		vm.log.Info(fmt.Sprintf("  Timing: verify=%dus analyze=%dus schedule=%dus execute=%dus commit=%dus merkle=%dus",
			turboResult.VerifyUs, turboResult.AnalyzeUs, turboResult.ScheduleUs,
			turboResult.ExecuteUs, turboResult.CommitUs, turboResult.MerkleUs))

		return newBlock, nil
	}

	// Fallback to QMDB executor (standard mode)
	if err := vm.qmdb.SetBlockContext(newBlock.height, newBlock.timestamp.Unix(), blockhash[:]); err != nil {
		return nil, fmt.Errorf("failed to set block context: %w", err)
	}

	// Execute block using QMDB (parallel execution with atomic state commit)
	blockResult, err := vm.qmdb.ExecuteBlock(newBlock.height, txDataList)
	if err != nil {
		vm.log.Warn(fmt.Sprintf("block execution failed: %v", err))
		return nil, fmt.Errorf("block execution failed: %w", err)
	}

	// Collect successful transactions
	var executedTxs []*Transaction
	for i, result := range blockResult.Results {
		if i < len(txs) && result.Success {
			txs[i].Result = result
			executedTxs = append(executedTxs, txs[i])
		}
	}

	if len(executedTxs) == 0 {
		return nil, errors.New("no successful transactions")
	}

	newBlock.transactions = executedTxs
	newBlock.stateRoot = blockResult.MerkleRoot // Store state root in block
	newBlock.id = newBlock.calculateID()

	vm.log.Info(fmt.Sprintf("built block id=%s height=%d txCount=%d successful=%d failed=%d time=%dus stateRoot=%x",
		newBlock.ID().String(), newBlock.Height(), len(txs),
		blockResult.Successful, blockResult.Failed, blockResult.ExecutionTimeUs,
		blockResult.MerkleRoot[:8]))

	return newBlock, nil
}

// ParseBlock implements block.ChainVM
func (vm *VM) ParseBlock(ctx context.Context, bytes []byte) (snowman.Block, error) {
	block, err := ParseBlock(vm, bytes)
	if err != nil {
		return nil, err
	}
	return block, nil
}

// GetBlock implements block.ChainVM
func (vm *VM) GetBlock(ctx context.Context, id ids.ID) (snowman.Block, error) {
	block, err := vm.state.GetBlock(id)
	if err != nil {
		vm.log.Error(fmt.Sprintf("GetBlock failed for id=%s: %v", id.String(), err))
		return nil, err
	}
	vm.log.Info(fmt.Sprintf("GetBlock id=%s height=%d timestamp=%s (unix=%d)",
		id.String(), block.Height(), block.Timestamp().String(), block.Timestamp().Unix()))
	return block, nil
}

// SetPreference implements block.ChainVM
func (vm *VM) SetPreference(ctx context.Context, id ids.ID) error {
	vm.mu.Lock()
	defer vm.mu.Unlock()

	vm.preferred = id
	return nil
}

// LastAccepted implements block.ChainVM
func (vm *VM) LastAccepted(ctx context.Context) (ids.ID, error) {
	id, err := vm.state.GetLastAccepted()
	if err != nil {
		vm.log.Error(fmt.Sprintf("LastAccepted failed: %v", err))
		return ids.Empty, err
	}
	vm.log.Info(fmt.Sprintf("LastAccepted called, returning id=%s", id.String()))
	return id, nil
}

// Health implements health.Checker
func (vm *VM) HealthCheck(ctx context.Context) (interface{}, error) {
	return map[string]interface{}{
		"healthy": true,
	}, nil
}

// Connected implements block.ChainVM
func (vm *VM) Connected(ctx context.Context, nodeID ids.NodeID, nodeVersion *version.Application) error {
	return nil
}

// Disconnected implements block.ChainVM
func (vm *VM) Disconnected(ctx context.Context, nodeID ids.NodeID) error {
	return nil
}

// CrossChainAppRequest implements block.ChainVM
func (vm *VM) CrossChainAppRequest(ctx context.Context, chainID ids.ID, requestID uint32, deadline time.Time, request []byte) error {
	return nil
}

// CrossChainAppRequestFailed implements block.ChainVM
func (vm *VM) CrossChainAppRequestFailed(ctx context.Context, chainID ids.ID, requestID uint32, appErr *common.AppError) error {
	return nil
}

// CrossChainAppResponse implements block.ChainVM
func (vm *VM) CrossChainAppResponse(ctx context.Context, chainID ids.ID, requestID uint32, response []byte) error {
	return nil
}

// AppGossip implements common.VM (optional)
func (vm *VM) AppGossip(ctx context.Context, nodeID ids.NodeID, msg []byte) error {
	return nil
}

// AppRequest implements common.VM (optional)
func (vm *VM) AppRequest(ctx context.Context, nodeID ids.NodeID, requestID uint32, deadline time.Time, request []byte) error {
	return nil
}

// AppResponse implements common.VM (optional)
func (vm *VM) AppResponse(ctx context.Context, nodeID ids.NodeID, requestID uint32, response []byte) error {
	return nil
}

// AppRequestFailed implements common.VM (optional)
func (vm *VM) AppRequestFailed(ctx context.Context, nodeID ids.NodeID, requestID uint32, appErr *common.AppError) error {
	return nil
}

// SubmitTransaction adds a transaction to the mempool
func (vm *VM) SubmitTransaction(tx *Transaction) error {
	if err := vm.mempool.Add(tx); err != nil {
		return err
	}

	// Notify engine that we have pending transactions
	select {
	case vm.pendingTxs <- struct{}{}:
		// Notification sent
	default:
		// Channel already has notification pending, no need to send another
	}

	return nil
}

// GetTransaction retrieves a transaction by signature
func (vm *VM) GetTransaction(signature []byte) (*Transaction, error) {
	return vm.state.GetTransaction(signature)
}

// GetAccount retrieves account state
func (vm *VM) GetAccount(pubkey [32]byte) (*AccountState, error) {
	return vm.state.GetAccount(pubkey)
}

// SetAccount sets account state (used by faucet/airdrop)
func (vm *VM) SetAccount(pubkey [32]byte, account *AccountState) error {
	return vm.state.SetAccount(pubkey, account)
}

// GetSlot returns the current slot (block height)
func (vm *VM) GetSlot() uint64 {
	vm.mu.RLock()
	defer vm.mu.RUnlock()

	if vm.lastBlock != nil {
		return vm.lastBlock.Height()
	}
	return 0
}

// GetBlockhash returns the hash of a given slot
func (vm *VM) GetBlockhash(slot uint64) (ids.ID, error) {
	// Search for block at given height
	// This is inefficient - in production would use an index
	lastAccepted, err := vm.state.GetLastAccepted()
	if err != nil {
		return ids.Empty, err
	}

	current, err := vm.state.GetBlock(lastAccepted)
	if err != nil {
		return ids.Empty, err
	}

	for current.Height() > slot && current.parentID != ids.Empty {
		current, err = vm.state.GetBlock(current.parentID)
		if err != nil {
			return ids.Empty, err
		}
	}

	if current.Height() == slot {
		return current.ID(), nil
	}

	return ids.Empty, fmt.Errorf("block at slot %d not found", slot)
}

// GetBlockIDAtHeight implements block.ChainVM
func (vm *VM) GetBlockIDAtHeight(ctx context.Context, height uint64) (ids.ID, error) {
	return vm.GetBlockhash(height)
}

// ExecuteBPF executes BPF bytecode via hybrid executor
func (vm *VM) ExecuteBPF(programID [32]byte, bytecode []byte, inputData []byte) (*ffi.ExecutionResult, error) {
	if vm.hybrid == nil {
		return nil, errors.New("hybrid executor not initialized")
	}

	// Set block context
	slot := vm.GetSlot()
	_ = vm.hybrid.SetBlock(slot, time.Now().Unix())

	return vm.hybrid.ExecuteBPF(programID, bytecode, inputData)
}

// ExecuteEVM executes an EVM transaction via hybrid executor
func (vm *VM) ExecuteEVM(caller [20]byte, to *[20]byte, value []byte, data []byte, gasLimit uint64) (*ffi.ExecutionResult, error) {
	if vm.hybrid == nil {
		return nil, errors.New("hybrid executor not initialized")
	}

	// Set block context
	slot := vm.GetSlot()
	vm.hybrid.SetBlock(slot, time.Now().Unix())

	// Convert value bytes to big.Int
	var valueBig *big.Int
	if len(value) > 0 {
		valueBig = new(big.Int).SetBytes(value)
	}

	return vm.hybrid.ExecuteEVM(caller, to, valueBig, data, gasLimit)
}

// GetHybridAccount retrieves account from hybrid executor
func (vm *VM) GetHybridAccount(pubkey [32]byte) (*ffi.StateChange, error) {
	if vm.hybrid == nil {
		return nil, errors.New("hybrid executor not initialized")
	}
	return vm.hybrid.GetAccount(pubkey)
}

// LoadHybridAccount loads account into hybrid executor
func (vm *VM) LoadHybridAccount(pubkey [32]byte, lamports uint64, data []byte, owner [32]byte, executable bool) error {
	if vm.hybrid == nil {
		return errors.New("hybrid executor not initialized")
	}
	return vm.hybrid.LoadAccount(pubkey, lamports, data, owner, executable)
}
