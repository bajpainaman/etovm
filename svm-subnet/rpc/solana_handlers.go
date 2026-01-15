package rpc

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/eto-chain/svm-subnet/vm"
	"github.com/mr-tron/base58"
)

// dispatchSolana routes Solana RPC methods
func (s *Server) dispatchSolana(ctx context.Context, method string, params json.RawMessage) (interface{}, error) {
	switch method {
	// Account methods
	case "getAccountInfo":
		return s.getAccountInfo(ctx, params)
	case "getBalance":
		return s.getBalance(ctx, params)
	case "getMultipleAccounts":
		return s.getMultipleAccounts(ctx, params)

	// Block methods
	case "getBlock":
		return s.getBlock(ctx, params)
	case "getBlockHeight":
		return s.getBlockHeight(ctx, params)
	case "getBlockTime":
		return s.getBlockTime(ctx, params)

	// Slot methods
	case "getSlot":
		return s.getSlot(ctx, params)

	// Transaction methods
	case "getTransaction":
		return s.getTransaction(ctx, params)
	case "sendTransaction":
		return s.sendTransaction(ctx, params)
	case "simulateTransaction":
		return s.simulateTransaction(ctx, params)

	// Blockhash methods
	case "getRecentBlockhash":
		return s.getRecentBlockhash(ctx, params)
	case "getLatestBlockhash":
		return s.getLatestBlockhash(ctx, params)
	case "isBlockhashValid":
		return s.isBlockhashValid(ctx, params)
	case "getFeeForMessage":
		return s.getFeeForMessage(ctx, params)

	// Cluster methods
	case "getHealth":
		return s.getHealth(ctx, params)
	case "getVersion":
		return s.getVersion(ctx, params)
	case "getGenesisHash":
		return s.getGenesisHash(ctx, params)

	// Rent
	case "getMinimumBalanceForRentExemption":
		return s.getMinimumBalanceForRentExemption(ctx, params)

	default:
		return nil, &RPCError{Code: MethodNotFound, Message: fmt.Sprintf("Method not found: %s", method)}
	}
}

// ========== Account Methods ==========

type GetAccountInfoParams struct {
	Pubkey string                 `json:"pubkey"`
	Config *AccountInfoConfig     `json:"config,omitempty"`
}

type AccountInfoConfig struct {
	Encoding   string `json:"encoding,omitempty"`
	Commitment string `json:"commitment,omitempty"`
}

type AccountInfoResult struct {
	Context *Context     `json:"context"`
	Value   *AccountInfo `json:"value"`
}

type Context struct {
	Slot uint64 `json:"slot"`
}

type AccountInfo struct {
	Lamports   uint64   `json:"lamports"`
	Data       []string `json:"data"`
	Owner      string   `json:"owner"`
	Executable bool     `json:"executable"`
	RentEpoch  uint64   `json:"rentEpoch"`
}

func (s *Server) getAccountInfo(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing pubkey"}
	}

	pubkeyStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid pubkey"}
	}

	pubkeyBytes, err := base58.Decode(pubkeyStr)
	if err != nil || len(pubkeyBytes) != 32 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid pubkey format"}
	}

	var pubkey [32]byte
	copy(pubkey[:], pubkeyBytes)

	account, err := s.vm.GetAccount(pubkey)
	if err != nil {
		return &AccountInfoResult{
			Context: &Context{Slot: s.vm.GetSlot()},
			Value:   nil,
		}, nil
	}

	encoding := "base64"
	if len(args) > 1 {
		if cfg, ok := args[1].(map[string]interface{}); ok {
			if enc, ok := cfg["encoding"].(string); ok {
				encoding = enc
			}
		}
	}

	var dataEncoded string
	switch encoding {
	case "base58":
		dataEncoded = base58.Encode(account.Data)
	case "base64":
		dataEncoded = base64.StdEncoding.EncodeToString(account.Data)
	default:
		dataEncoded = base64.StdEncoding.EncodeToString(account.Data)
		encoding = "base64"
	}

	return &AccountInfoResult{
		Context: &Context{Slot: s.vm.GetSlot()},
		Value: &AccountInfo{
			Lamports:   account.Lamports,
			Data:       []string{dataEncoded, encoding},
			Owner:      base58.Encode(account.Owner[:]),
			Executable: account.Executable,
			RentEpoch:  account.RentEpoch,
		},
	}, nil
}

type BalanceResult struct {
	Context *Context `json:"context"`
	Value   uint64   `json:"value"`
}

func (s *Server) getBalance(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing pubkey"}
	}

	pubkeyStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid pubkey"}
	}

	pubkeyBytes, err := base58.Decode(pubkeyStr)
	if err != nil || len(pubkeyBytes) != 32 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid pubkey format"}
	}

	var pubkey [32]byte
	copy(pubkey[:], pubkeyBytes)

	account, err := s.vm.GetAccount(pubkey)
	if err != nil {
		return &BalanceResult{
			Context: &Context{Slot: s.vm.GetSlot()},
			Value:   0,
		}, nil
	}

	return &BalanceResult{
		Context: &Context{Slot: s.vm.GetSlot()},
		Value:   account.Lamports,
	}, nil
}

func (s *Server) getMultipleAccounts(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing pubkeys"}
	}

	pubkeys, ok := args[0].([]interface{})
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid pubkeys"}
	}

	encoding := "base64"
	if len(args) > 1 {
		if cfg, ok := args[1].(map[string]interface{}); ok {
			if enc, ok := cfg["encoding"].(string); ok {
				encoding = enc
			}
		}
	}

	var accounts []*AccountInfo
	for _, pk := range pubkeys {
		pubkeyStr, ok := pk.(string)
		if !ok {
			accounts = append(accounts, nil)
			continue
		}

		pubkeyBytes, err := base58.Decode(pubkeyStr)
		if err != nil || len(pubkeyBytes) != 32 {
			accounts = append(accounts, nil)
			continue
		}

		var pubkey [32]byte
		copy(pubkey[:], pubkeyBytes)

		account, err := s.vm.GetAccount(pubkey)
		if err != nil {
			accounts = append(accounts, nil)
			continue
		}

		var dataEncoded string
		switch encoding {
		case "base58":
			dataEncoded = base58.Encode(account.Data)
		default:
			dataEncoded = base64.StdEncoding.EncodeToString(account.Data)
		}

		accounts = append(accounts, &AccountInfo{
			Lamports:   account.Lamports,
			Data:       []string{dataEncoded, encoding},
			Owner:      base58.Encode(account.Owner[:]),
			Executable: account.Executable,
			RentEpoch:  account.RentEpoch,
		})
	}

	return map[string]interface{}{
		"context": &Context{Slot: s.vm.GetSlot()},
		"value":   accounts,
	}, nil
}

// ========== Block Methods ==========

func (s *Server) getBlock(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing slot"}
	}

	slot, ok := args[0].(float64)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid slot"}
	}

	blockID, err := s.vm.GetBlockhash(uint64(slot))
	if err != nil {
		return nil, &RPCError{Code: BlockNotAvailable, Message: "Block not available"}
	}

	// Return simplified block info
	return map[string]interface{}{
		"blockhash":         base58.Encode(blockID[:]),
		"blockHeight":       uint64(slot),
		"blockTime":         nil, // Would need to fetch from state
		"parentSlot":        uint64(slot) - 1,
		"previousBlockhash": nil,
		"transactions":      []interface{}{},
	}, nil
}

func (s *Server) getBlockHeight(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return s.vm.GetSlot(), nil
}

func (s *Server) getBlockTime(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement block time lookup
	return nil, nil
}

// ========== Slot Methods ==========

func (s *Server) getSlot(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return s.vm.GetSlot(), nil
}

// ========== Transaction Methods ==========

func (s *Server) getTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing signature"}
	}

	sigStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid signature"}
	}

	sigBytes, err := base58.Decode(sigStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid signature format"}
	}

	tx, err := s.vm.GetTransaction(sigBytes)
	if err != nil {
		return nil, nil // Transaction not found returns null
	}

	return map[string]interface{}{
		"slot":        s.vm.GetSlot(),
		"transaction": tx.ToData(),
		"meta": map[string]interface{}{
			"err":                  nil,
			"fee":                  tx.Result.Fee,
			"computeUnitsConsumed": tx.Result.ComputeUnitsUsed,
			"logMessages":          tx.Result.Logs,
		},
	}, nil
}

func (s *Server) sendTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing transaction"}
	}

	txStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid transaction"}
	}

	// Determine encoding
	encoding := "base58"
	if len(args) > 1 {
		if cfg, ok := args[1].(map[string]interface{}); ok {
			if enc, ok := cfg["encoding"].(string); ok {
				encoding = enc
			}
		}
	}

	var txBytes []byte
	var err error
	switch encoding {
	case "base64":
		txBytes, err = base64.StdEncoding.DecodeString(txStr)
	default:
		txBytes, err = base58.Decode(txStr)
	}
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Failed to decode transaction"}
	}

	tx, err := vm.TransactionFromBytes(txBytes)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Failed to parse transaction"}
	}

	if err := s.vm.SubmitTransaction(tx); err != nil {
		return nil, &RPCError{Code: TransactionError, Message: err.Error()}
	}

	// Return signature
	return base58.Encode(tx.Signature()), nil
}

func (s *Server) simulateTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement transaction simulation
	return map[string]interface{}{
		"context": &Context{Slot: s.vm.GetSlot()},
		"value": map[string]interface{}{
			"err":                  nil,
			"logs":                 []string{},
			"unitsConsumed":        0,
			"returnData":           nil,
		},
	}, nil
}

// ========== Blockhash Methods ==========

type BlockhashResult struct {
	Context *Context       `json:"context"`
	Value   *BlockhashInfo `json:"value"`
}

type BlockhashInfo struct {
	Blockhash     string `json:"blockhash"`
	LastValidSlot uint64 `json:"lastValidBlockHeight"`
}

func (s *Server) getRecentBlockhash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return s.getLatestBlockhash(ctx, params)
}

func (s *Server) getLatestBlockhash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	slot := s.vm.GetSlot()
	blockhash, err := s.vm.GetBlockhash(slot)
	if err != nil {
		return nil, &RPCError{Code: InternalError, Message: "Failed to get blockhash"}
	}

	return &BlockhashResult{
		Context: &Context{Slot: slot},
		Value: &BlockhashInfo{
			Blockhash:     base58.Encode(blockhash[:]),
			LastValidSlot: slot + 150, // ~150 blocks validity
		},
	}, nil
}

func (s *Server) isBlockhashValid(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	if len(args) == 0 {
		return nil, &RPCError{Code: InvalidParams, Message: "Missing blockhash"}
	}

	blockhashStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid blockhash"}
	}

	blockhashBytes, err := base58.Decode(blockhashStr)
	if err != nil || len(blockhashBytes) != 32 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid blockhash format"}
	}

	// Check if blockhash is recent (within 150 blocks)
	currentSlot := s.vm.GetSlot()
	// For now, assume valid if we can decode it
	return map[string]interface{}{
		"context": &Context{Slot: currentSlot},
		"value":   true,
	}, nil
}

func (s *Server) getFeeForMessage(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// Return base fee
	return map[string]interface{}{
		"context": &Context{Slot: s.vm.GetSlot()},
		"value":   uint64(5000), // 5000 lamports per signature
	}, nil
}

// ========== Cluster Methods ==========

func (s *Server) getHealth(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return "ok", nil
}

func (s *Server) getVersion(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return map[string]interface{}{
		"solana-core": "1.18.0", // Compatibility version
		"feature-set": 0,
	}, nil
}

func (s *Server) getGenesisHash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	genesisHash, err := s.vm.GetBlockhash(0)
	if err != nil {
		return nil, &RPCError{Code: InternalError, Message: "Failed to get genesis hash"}
	}
	return base58.Encode(genesisHash[:]), nil
}

// ========== Rent Methods ==========

func (s *Server) getMinimumBalanceForRentExemption(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	dataLen := uint64(0)
	if len(args) > 0 {
		if dl, ok := args[0].(float64); ok {
			dataLen = uint64(dl)
		}
	}

	// Calculate rent-exempt balance: base + per-byte cost
	// Formula: (128 + dataLen) * lamports_per_byte_year * 2
	lamportsPerByteYear := uint64(3480)
	minBalance := (128 + dataLen) * lamportsPerByteYear * 2

	return minBalance, nil
}
