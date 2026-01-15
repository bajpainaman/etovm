package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/snow/choices"
	"github.com/ava-labs/avalanchego/snow/consensus/snowman"
	"github.com/ava-labs/avalanchego/utils/hashing"
)

var _ snowman.Block = (*Block)(nil)

// Block implements the snowman.Block interface
type Block struct {
	vm *VM

	id           ids.ID
	parentID     ids.ID
	height       uint64
	timestamp    time.Time
	transactions []*Transaction
	stateRoot    [32]byte

	status choices.Status
	bytes  []byte
}

// BlockData is the serializable representation of a block
type BlockData struct {
	ParentID     ids.ID             `json:"parentID"`
	Height       uint64             `json:"height"`
	Timestamp    int64              `json:"timestamp"`
	Transactions []*TransactionData `json:"transactions"`
	StateRoot    [32]byte           `json:"stateRoot"`
}

// ID implements snowman.Block
func (b *Block) ID() ids.ID {
	return b.id
}

// Parent implements snowman.Block
func (b *Block) Parent() ids.ID {
	return b.parentID
}

// Height implements snowman.Block
func (b *Block) Height() uint64 {
	return b.height
}

// Timestamp implements snowman.Block
func (b *Block) Timestamp() time.Time {
	// Debug: ensure timestamp is never zero
	if b.timestamp.IsZero() {
		// Return a safe default
		return time.Unix(1, 0)
	}
	return b.timestamp
}

// Status implements snowman.Block
func (b *Block) Status() choices.Status {
	return b.status
}

// Bytes implements snowman.Block
func (b *Block) Bytes() []byte {
	if b.bytes == nil {
		b.bytes = b.serialize()
	}
	return b.bytes
}

// Verify implements snowman.Block
func (b *Block) Verify(ctx context.Context) error {
	// Verify parent exists
	parent, err := b.vm.state.GetBlock(b.parentID)
	if err != nil {
		return fmt.Errorf("parent block not found: %w", err)
	}

	// Verify height is consecutive
	if b.height != parent.Height()+1 {
		return fmt.Errorf("invalid height: expected %d, got %d", parent.Height()+1, b.height)
	}

	// Verify timestamp is after parent
	if !b.timestamp.After(parent.Timestamp()) {
		return fmt.Errorf("timestamp must be after parent")
	}

	// Verify transactions
	for _, tx := range b.transactions {
		if err := tx.Verify(); err != nil {
			return fmt.Errorf("invalid transaction: %w", err)
		}
	}

	return nil
}

// Accept implements snowman.Block
func (b *Block) Accept(ctx context.Context) error {
	b.vm.mu.Lock()
	defer b.vm.mu.Unlock()

	b.status = choices.Accepted

	// NOTE: We do NOT re-execute transactions here.
	// Transactions were already executed in BuildBlockWithContext and their
	// results are cached in tx.Result. This eliminates double execution
	// which was a major performance bottleneck.

	// Apply cached state changes from execution results
	for _, tx := range b.transactions {
		if tx.Result == nil {
			b.vm.log.Warn("transaction has no cached result, skipping state changes")
			continue
		}

		// Apply state changes from cached result
		for _, change := range tx.Result.StateChanges {
			if err := b.vm.state.SetAccount(change.Pubkey, &AccountState{
				Lamports:   change.Lamports,
				Data:       change.Data,
				Owner:      change.Owner,
				Executable: change.Executable,
			}); err != nil {
				return fmt.Errorf("failed to apply state change: %w", err)
			}
		}

		// Store transaction
		if err := b.vm.state.PutTransaction(tx); err != nil {
			return fmt.Errorf("failed to store transaction: %w", err)
		}
	}

	// Save block
	if err := b.vm.state.PutBlock(b); err != nil {
		return fmt.Errorf("failed to save block: %w", err)
	}

	// Update last accepted
	if err := b.vm.state.SetLastAccepted(b.id); err != nil {
		return fmt.Errorf("failed to set last accepted: %w", err)
	}

	b.vm.lastBlock = b

	b.vm.log.Info(fmt.Sprintf("accepted block id=%s height=%d txCount=%d", b.id.String(), b.height, len(b.transactions)))

	return nil
}

// Reject implements snowman.Block
func (b *Block) Reject(ctx context.Context) error {
	b.status = choices.Rejected

	// Return transactions to mempool
	for _, tx := range b.transactions {
		_ = b.vm.mempool.Add(tx)
	}

	b.vm.log.Info(fmt.Sprintf("rejected block id=%s", b.id.String()))

	return nil
}

func (b *Block) serialize() []byte {
	data := BlockData{
		ParentID:  b.parentID,
		Height:    b.height,
		Timestamp: b.timestamp.Unix(),
		StateRoot: b.stateRoot,
	}

	for _, tx := range b.transactions {
		data.Transactions = append(data.Transactions, tx.ToData())
	}

	bytes, _ := json.Marshal(data)
	return bytes
}

func (b *Block) calculateID() ids.ID {
	return hashing.ComputeHash256Array(b.Bytes())
}

// ParseBlock deserializes a block from bytes
func ParseBlock(vm *VM, bytes []byte) (*Block, error) {
	var data BlockData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal block: %w", err)
	}

	// Ensure timestamp is valid (protobuf requires non-zero)
	blockTime := time.Unix(data.Timestamp, 0)
	if data.Timestamp <= 0 {
		blockTime = time.Unix(1, 0)
	}

	block := &Block{
		vm:        vm,
		parentID:  data.ParentID,
		height:    data.Height,
		timestamp: blockTime,
		stateRoot: data.StateRoot,
		bytes:     bytes,
		status:    choices.Processing,
	}

	for _, txData := range data.Transactions {
		tx, err := TransactionFromData(txData)
		if err != nil {
			return nil, fmt.Errorf("failed to parse transaction: %w", err)
		}
		block.transactions = append(block.transactions, tx)
	}

	block.id = block.calculateID()

	return block, nil
}
