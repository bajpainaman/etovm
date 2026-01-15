package vm

import (
	"encoding/json"
	"fmt"

	"github.com/eto-chain/svm-subnet/vm/ffi"
)

// Transaction represents a Solana-compatible transaction
type Transaction struct {
	// Signatures for this transaction (first is primary/fee payer)
	Signatures [][]byte `json:"signatures"`

	// Message containing the instructions
	Message *TransactionMessage `json:"message"`

	// Cached serialized form
	bytes []byte

	// Execution result (populated after execution)
	Result *ffi.ExecutionResult
}

// TransactionMessage contains the transaction payload
type TransactionMessage struct {
	// Header with signature counts
	Header TransactionHeader `json:"header"`

	// All account pubkeys referenced by this transaction
	AccountKeys [][32]byte `json:"accountKeys"`

	// Recent blockhash for replay protection
	RecentBlockhash [32]byte `json:"recentBlockhash"`

	// Compiled instructions
	Instructions []CompiledInstruction `json:"instructions"`
}

// TransactionHeader contains metadata about signers
type TransactionHeader struct {
	NumRequiredSignatures       uint8 `json:"numRequiredSignatures"`
	NumReadonlySignedAccounts   uint8 `json:"numReadonlySignedAccounts"`
	NumReadonlyUnsignedAccounts uint8 `json:"numReadonlyUnsignedAccounts"`
}

// CompiledInstruction is a compact instruction representation
type CompiledInstruction struct {
	// Index into AccountKeys for the program
	ProgramIDIndex uint8 `json:"programIdIndex"`

	// Indices into AccountKeys for accounts this instruction uses
	Accounts []uint8 `json:"accounts"`

	// Instruction data
	Data []byte `json:"data"`
}

// TransactionData is the serializable form
type TransactionData struct {
	Signatures      [][]byte             `json:"signatures"`
	Message         *TransactionMessage  `json:"message"`
	ExecutionResult *ffi.ExecutionResult `json:"result,omitempty"`
}

// NewTransaction creates a new unsigned transaction
func NewTransaction(message *TransactionMessage) *Transaction {
	return &Transaction{
		Signatures: make([][]byte, message.Header.NumRequiredSignatures),
		Message:    message,
	}
}

// Bytes returns the serialized transaction (for submission to runtime)
func (tx *Transaction) Bytes() []byte {
	if tx.bytes == nil {
		// Serialize in borsh format matching Rust expectations
		tx.bytes = tx.toBorsh()
	}
	return tx.bytes
}

// toBorsh serializes the transaction in borsh format
func (tx *Transaction) toBorsh() []byte {
	// This is a simplified borsh serialization
	// In production, use a proper borsh library
	data, _ := json.Marshal(tx.ToData())
	return data
}

// Signature returns the primary signature (first signature)
func (tx *Transaction) Signature() []byte {
	if len(tx.Signatures) > 0 {
		return tx.Signatures[0]
	}
	return nil
}

// FeePayer returns the fee payer pubkey (first account key)
func (tx *Transaction) FeePayer() [32]byte {
	if tx.Message != nil && len(tx.Message.AccountKeys) > 0 {
		return tx.Message.AccountKeys[0]
	}
	return [32]byte{}
}

// Verify validates the transaction structure (not signatures)
func (tx *Transaction) Verify() error {
	if tx.Message == nil {
		return fmt.Errorf("transaction message is nil")
	}

	if len(tx.Signatures) != int(tx.Message.Header.NumRequiredSignatures) {
		return fmt.Errorf("signature count mismatch: expected %d, got %d",
			tx.Message.Header.NumRequiredSignatures, len(tx.Signatures))
	}

	if len(tx.Message.AccountKeys) == 0 {
		return fmt.Errorf("no account keys")
	}

	for i, ix := range tx.Message.Instructions {
		if int(ix.ProgramIDIndex) >= len(tx.Message.AccountKeys) {
			return fmt.Errorf("instruction %d: program ID index out of range", i)
		}

		for j, accountIdx := range ix.Accounts {
			if int(accountIdx) >= len(tx.Message.AccountKeys) {
				return fmt.Errorf("instruction %d, account %d: index out of range", i, j)
			}
		}
	}

	return nil
}

// ToData converts to serializable form
func (tx *Transaction) ToData() *TransactionData {
	return &TransactionData{
		Signatures:      tx.Signatures,
		Message:         tx.Message,
		ExecutionResult: tx.Result,
	}
}

// TransactionFromData reconstructs a transaction from data
func TransactionFromData(data *TransactionData) (*Transaction, error) {
	if data == nil {
		return nil, fmt.Errorf("nil transaction data")
	}

	return &Transaction{
		Signatures: data.Signatures,
		Message:    data.Message,
		Result:     data.ExecutionResult,
	}, nil
}

// TransactionFromBytes deserializes a transaction
func TransactionFromBytes(bytes []byte) (*Transaction, error) {
	var data TransactionData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal transaction: %w", err)
	}
	return TransactionFromData(&data)
}
