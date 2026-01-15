package rpc

import (
	"crypto/sha256"
	"encoding/hex"
	"math/big"
	"sync"

	"github.com/eto-chain/svm-subnet/vm"
)

// EVMBackend provides EVM-compatible state access
// This bridges EVM addresses to Solana-style accounts
type EVMBackend struct {
	svm     *vm.VM
	chainID uint64
	mu      sync.RWMutex

	// Pending transactions
	pendingNonces map[string]uint64

	// Block state
	baseFee *big.Int
}

// NewEVMBackend creates a new EVM backend
func NewEVMBackend(svm *vm.VM, chainID uint64) *EVMBackend {
	return &EVMBackend{
		svm:           svm,
		chainID:       chainID,
		pendingNonces: make(map[string]uint64),
		baseFee:       big.NewInt(1000000000), // 1 gwei
	}
}

// Address conversion utilities

// EVMAddressToSolana converts a 20-byte EVM address to 32-byte Solana pubkey
// Uses a deterministic derivation: sha256("evm:" || address)[0:32]
func EVMAddressToSolana(evmAddr [20]byte) [32]byte {
	data := append([]byte("evm:"), evmAddr[:]...)
	hash := sha256.Sum256(data)
	return hash
}

// SolanaToEVMAddress extracts EVM address from account data if present
// Returns the first 20 bytes of the pubkey as a fallback
func SolanaToEVMAddress(pubkey [32]byte) [20]byte {
	var addr [20]byte
	copy(addr[:], pubkey[12:32]) // Take last 20 bytes like Ethereum
	return addr
}

// HexToEVMAddress parses a hex address string
func HexToEVMAddress(hexAddr string) ([20]byte, error) {
	var addr [20]byte
	if len(hexAddr) >= 2 && hexAddr[:2] == "0x" {
		hexAddr = hexAddr[2:]
	}
	bytes, err := hex.DecodeString(hexAddr)
	if err != nil {
		return addr, err
	}
	if len(bytes) != 20 {
		return addr, nil
	}
	copy(addr[:], bytes)
	return addr, nil
}

// EVMAddressToHex converts address to 0x-prefixed hex
func EVMAddressToHex(addr [20]byte) string {
	return "0x" + hex.EncodeToString(addr[:])
}

// ========== Balance & Nonce ==========

// GetBalance returns the balance for an EVM address in wei
// Converts lamports to wei (1 lamport = 1e9 wei for compatibility)
func (e *EVMBackend) GetBalance(addr [20]byte) *big.Int {
	solanaPubkey := EVMAddressToSolana(addr)
	account, err := e.svm.GetAccount(solanaPubkey)
	if err != nil {
		return big.NewInt(0)
	}

	// Convert lamports to wei (multiply by 1e9 for gas compatibility)
	balance := new(big.Int).SetUint64(account.Lamports)
	return balance.Mul(balance, big.NewInt(1e9))
}

// GetNonce returns the transaction count for an address
func (e *EVMBackend) GetNonce(addr [20]byte) uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()

	hexAddr := EVMAddressToHex(addr)
	if nonce, ok := e.pendingNonces[hexAddr]; ok {
		return nonce
	}

	// TODO: Read actual nonce from account state
	return 0
}

// IncrementNonce increments the nonce for an address
func (e *EVMBackend) IncrementNonce(addr [20]byte) {
	e.mu.Lock()
	defer e.mu.Unlock()

	hexAddr := EVMAddressToHex(addr)
	e.pendingNonces[hexAddr]++
}

// ========== Code & Storage ==========

// GetCode returns the contract code at an address
func (e *EVMBackend) GetCode(addr [20]byte) []byte {
	solanaPubkey := EVMAddressToSolana(addr)
	account, err := e.svm.GetAccount(solanaPubkey)
	if err != nil {
		return nil
	}

	if !account.Executable {
		return nil
	}

	return account.Data
}

// GetStorageAt returns storage value at a position
func (e *EVMBackend) GetStorageAt(addr [20]byte, position [32]byte) [32]byte {
	// TODO: Implement storage slot mapping
	return [32]byte{}
}

// ========== Block Info ==========

// BlockNumber returns the current block number
func (e *EVMBackend) BlockNumber() uint64 {
	return e.svm.GetSlot()
}

// BlockHash returns the hash of a block
func (e *EVMBackend) BlockHash(number uint64) [32]byte {
	hash, err := e.svm.GetBlockhash(number)
	if err != nil {
		return [32]byte{}
	}
	return hash
}

// ChainID returns the chain ID
func (e *EVMBackend) ChainID() uint64 {
	return e.chainID
}

// BaseFee returns the current base fee
func (e *EVMBackend) BaseFee() *big.Int {
	return new(big.Int).Set(e.baseFee)
}

// GasPrice returns the current gas price
func (e *EVMBackend) GasPrice() *big.Int {
	return new(big.Int).Set(e.baseFee)
}

// ========== Transaction Types ==========

// EVMTransaction represents an Ethereum transaction
type EVMTransaction struct {
	Type     uint8    `json:"type"`
	ChainID  uint64   `json:"chainId"`
	Nonce    uint64   `json:"nonce"`
	GasPrice *big.Int `json:"gasPrice,omitempty"`
	GasTip   *big.Int `json:"maxPriorityFeePerGas,omitempty"`
	GasCap   *big.Int `json:"maxFeePerGas,omitempty"`
	Gas      uint64   `json:"gas"`
	To       *[20]byte `json:"to"`
	Value    *big.Int `json:"value"`
	Data     []byte   `json:"input"`
	V        *big.Int `json:"v"`
	R        *big.Int `json:"r"`
	S        *big.Int `json:"s"`
}

// EVMReceipt represents a transaction receipt
type EVMReceipt struct {
	TransactionHash   [32]byte  `json:"transactionHash"`
	TransactionIndex  uint64    `json:"transactionIndex"`
	BlockHash         [32]byte  `json:"blockHash"`
	BlockNumber       uint64    `json:"blockNumber"`
	From              [20]byte  `json:"from"`
	To                *[20]byte `json:"to"`
	CumulativeGasUsed uint64    `json:"cumulativeGasUsed"`
	GasUsed           uint64    `json:"gasUsed"`
	ContractAddress   *[20]byte `json:"contractAddress"`
	Logs              []EVMLog  `json:"logs"`
	Status            uint64    `json:"status"`
	EffectiveGasPrice *big.Int  `json:"effectiveGasPrice"`
}

// EVMLog represents an event log
type EVMLog struct {
	Address     [20]byte   `json:"address"`
	Topics      [][32]byte `json:"topics"`
	Data        []byte     `json:"data"`
	BlockNumber uint64     `json:"blockNumber"`
	TxHash      [32]byte   `json:"transactionHash"`
	TxIndex     uint64     `json:"transactionIndex"`
	BlockHash   [32]byte   `json:"blockHash"`
	LogIndex    uint64     `json:"logIndex"`
	Removed     bool       `json:"removed"`
}

// EVMBlock represents a block
type EVMBlock struct {
	Number           uint64        `json:"number"`
	Hash             [32]byte      `json:"hash"`
	ParentHash       [32]byte      `json:"parentHash"`
	Nonce            uint64        `json:"nonce"`
	MixHash          [32]byte      `json:"mixHash"`
	Sha3Uncles       [32]byte      `json:"sha3Uncles"`
	LogsBloom        [256]byte     `json:"logsBloom"`
	TransactionsRoot [32]byte      `json:"transactionsRoot"`
	StateRoot        [32]byte      `json:"stateRoot"`
	ReceiptsRoot     [32]byte      `json:"receiptsRoot"`
	Miner            [20]byte      `json:"miner"`
	Difficulty       *big.Int      `json:"difficulty"`
	TotalDifficulty  *big.Int      `json:"totalDifficulty"`
	ExtraData        []byte        `json:"extraData"`
	Size             uint64        `json:"size"`
	GasLimit         uint64        `json:"gasLimit"`
	GasUsed          uint64        `json:"gasUsed"`
	Timestamp        uint64        `json:"timestamp"`
	Transactions     []interface{} `json:"transactions"`
	Uncles           [][32]byte    `json:"uncles"`
	BaseFeePerGas    *big.Int      `json:"baseFeePerGas"`
}
