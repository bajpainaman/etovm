package vm

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/ava-labs/avalanchego/database"
	"github.com/ava-labs/avalanchego/ids"
)

var (
	blockPrefix       = []byte("block:")
	accountPrefix     = []byte("account:")
	transactionPrefix = []byte("tx:")
	lastAcceptedKey   = []byte("lastAccepted")
)

// AccountState represents the state of a Solana-style account
type AccountState struct {
	Lamports   uint64   `json:"lamports"`
	Data       []byte   `json:"data"`
	Owner      [32]byte `json:"owner"`
	Executable bool     `json:"executable"`
	RentEpoch  uint64   `json:"rentEpoch"`
}

// State manages the VM's persistent state
type State struct {
	db database.Database
}

// NewState creates a new State instance
func NewState(db database.Database) *State {
	return &State{db: db}
}

// GetLastAccepted returns the ID of the last accepted block
func (s *State) GetLastAccepted() (ids.ID, error) {
	bytes, err := s.db.Get(lastAcceptedKey)
	if err != nil {
		return ids.Empty, err
	}

	var id ids.ID
	copy(id[:], bytes)
	return id, nil
}

// SetLastAccepted sets the ID of the last accepted block
func (s *State) SetLastAccepted(id ids.ID) error {
	return s.db.Put(lastAcceptedKey, id[:])
}

// GetBlock retrieves a block by ID
func (s *State) GetBlock(id ids.ID) (*Block, error) {
	key := append(blockPrefix, id[:]...)
	bytes, err := s.db.Get(key)
	if err != nil {
		return nil, err
	}

	// Note: We need a reference to the VM to fully reconstruct the block
	// This is handled by the caller providing the VM reference
	var data BlockData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, err
	}

	// Ensure timestamp is valid (protobuf requires non-zero)
	blockTime := time.Unix(data.Timestamp, 0)
	if data.Timestamp <= 0 {
		// Default to Unix epoch + 1 second if missing
		blockTime = time.Unix(1, 0)
	}

	block := &Block{
		id:        id,
		parentID:  data.ParentID,
		height:    data.Height,
		timestamp: blockTime,
		stateRoot: data.StateRoot,
		bytes:     bytes,
	}

	for _, txData := range data.Transactions {
		tx, err := TransactionFromData(txData)
		if err != nil {
			return nil, err
		}
		block.transactions = append(block.transactions, tx)
	}

	return block, nil
}

// PutBlock stores a block
func (s *State) PutBlock(block *Block) error {
	id := block.ID()
	key := append(blockPrefix, id[:]...)
	return s.db.Put(key, block.Bytes())
}

// GetAccount retrieves account state by pubkey
func (s *State) GetAccount(pubkey [32]byte) (*AccountState, error) {
	key := append(accountPrefix, pubkey[:]...)
	bytes, err := s.db.Get(key)
	if err != nil {
		return nil, err
	}

	var account AccountState
	if err := json.Unmarshal(bytes, &account); err != nil {
		return nil, err
	}

	return &account, nil
}

// SetAccount stores account state
func (s *State) SetAccount(pubkey [32]byte, account *AccountState) error {
	key := append(accountPrefix, pubkey[:]...)
	bytes, err := json.Marshal(account)
	if err != nil {
		return err
	}
	return s.db.Put(key, bytes)
}

// DeleteAccount removes an account
func (s *State) DeleteAccount(pubkey [32]byte) error {
	key := append(accountPrefix, pubkey[:]...)
	return s.db.Delete(key)
}

// GetTransaction retrieves a transaction by signature
func (s *State) GetTransaction(signature []byte) (*Transaction, error) {
	key := append(transactionPrefix, signature...)
	bytes, err := s.db.Get(key)
	if err != nil {
		return nil, err
	}

	var data TransactionData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, err
	}

	return TransactionFromData(&data)
}

// PutTransaction stores a transaction
func (s *State) PutTransaction(tx *Transaction) error {
	if len(tx.Signatures) == 0 {
		return fmt.Errorf("transaction has no signature")
	}

	key := append(transactionPrefix, tx.Signatures[0]...)
	bytes, err := json.Marshal(tx.ToData())
	if err != nil {
		return err
	}
	return s.db.Put(key, bytes)
}

// HasBlock checks if a block exists
func (s *State) HasBlock(id ids.ID) (bool, error) {
	key := append(blockPrefix, id[:]...)
	return s.db.Has(key)
}

// Commit commits pending changes to the database
func (s *State) Commit() error {
	// For simple database implementations, this is a no-op
	// For batched writes, this would commit the batch
	return nil
}

// Close closes the state database
func (s *State) Close() error {
	return s.db.Close()
}
