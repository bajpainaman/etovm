package vm

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strings"

	"github.com/mr-tron/base58"
)

// ServeHTTP implements http.Handler
func (h *rpcHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Handle CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req rpcRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.writeError(w, nil, -32700, "Parse error", nil)
		return
	}

	if req.JSONRPC != "2.0" {
		h.writeError(w, req.ID, -32600, "Invalid Request", nil)
		return
	}

	result, err := h.dispatch(r.Context(), req.Method, req.Params)
	if err != nil {
		if rpcErr, ok := err.(*rpcError); ok {
			h.writeError(w, req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		} else {
			h.writeError(w, req.ID, -32603, err.Error(), nil)
		}
		return
	}

	h.writeResult(w, req.ID, result)
}

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

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

func (e *rpcError) Error() string {
	return e.Message
}

func (h *rpcHandler) writeResult(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := rpcResponse{JSONRPC: "2.0", ID: id, Result: result}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *rpcHandler) writeError(w http.ResponseWriter, id interface{}, code int, msg string, data interface{}) {
	resp := rpcResponse{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: msg, Data: data}}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// dispatch routes methods to handlers
func (h *rpcHandler) dispatch(ctx context.Context, method string, params json.RawMessage) (interface{}, error) {
	// Ethereum methods
	if isEthMethod(method) {
		return h.dispatchEth(ctx, method, params)
	}
	// Solana methods
	return h.dispatchSolana(ctx, method, params)
}

func isEthMethod(method string) bool {
	prefixes := []string{"eth_", "net_", "web3_", "debug_", "txpool_"}
	for _, p := range prefixes {
		if strings.HasPrefix(method, p) {
			return true
		}
	}
	return false
}

// ========== Solana RPC Methods ==========

func (h *rpcHandler) dispatchSolana(ctx context.Context, method string, params json.RawMessage) (interface{}, error) {
	switch method {
	case "getAccountInfo":
		return h.getAccountInfo(ctx, params)
	case "getBalance":
		return h.getBalance(ctx, params)
	case "getSlot":
		return h.getSlot(ctx, params)
	case "getBlockHeight":
		return h.getBlockHeight(ctx, params)
	case "getLatestBlockhash", "getRecentBlockhash":
		return h.getLatestBlockhash(ctx, params)
	case "sendTransaction":
		return h.sendTransaction(ctx, params)
	case "getTransaction":
		return h.getTransaction(ctx, params)
	case "getHealth":
		return "ok", nil
	case "getVersion":
		return map[string]interface{}{"solana-core": "1.18.0", "feature-set": 0}, nil
	case "getMinimumBalanceForRentExemption":
		return h.getMinRentExemption(ctx, params)
	default:
		return nil, &rpcError{Code: -32601, Message: fmt.Sprintf("Method not found: %s", method)}
	}
}

func (h *rpcHandler) getAccountInfo(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	pubkeyStr, _ := args[0].(string)
	pubkeyBytes, err := base58.Decode(pubkeyStr)
	if err != nil || len(pubkeyBytes) != 32 {
		return nil, &rpcError{Code: -32602, Message: "Invalid pubkey"}
	}

	var pubkey [32]byte
	copy(pubkey[:], pubkeyBytes)

	account, err := h.vm.GetAccount(pubkey)
	if err != nil {
		return map[string]interface{}{"context": map[string]uint64{"slot": h.vm.GetSlot()}, "value": nil}, nil
	}

	return map[string]interface{}{
		"context": map[string]uint64{"slot": h.vm.GetSlot()},
		"value": map[string]interface{}{
			"lamports":   account.Lamports,
			"data":       []string{base64.StdEncoding.EncodeToString(account.Data), "base64"},
			"owner":      base58.Encode(account.Owner[:]),
			"executable": account.Executable,
			"rentEpoch":  account.RentEpoch,
		},
	}, nil
}

func (h *rpcHandler) getBalance(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	pubkeyStr, _ := args[0].(string)
	pubkeyBytes, err := base58.Decode(pubkeyStr)
	if err != nil || len(pubkeyBytes) != 32 {
		return nil, &rpcError{Code: -32602, Message: "Invalid pubkey"}
	}

	var pubkey [32]byte
	copy(pubkey[:], pubkeyBytes)

	account, err := h.vm.GetAccount(pubkey)
	balance := uint64(0)
	if err == nil {
		balance = account.Lamports
	}

	return map[string]interface{}{
		"context": map[string]uint64{"slot": h.vm.GetSlot()},
		"value":   balance,
	}, nil
}

func (h *rpcHandler) getSlot(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return h.vm.GetSlot(), nil
}

func (h *rpcHandler) getBlockHeight(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return h.vm.GetSlot(), nil
}

func (h *rpcHandler) getLatestBlockhash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	slot := h.vm.GetSlot()
	blockhash, err := h.vm.GetBlockhash(slot)
	if err != nil {
		return nil, &rpcError{Code: -32603, Message: "Failed to get blockhash"}
	}

	return map[string]interface{}{
		"context": map[string]uint64{"slot": slot},
		"value": map[string]interface{}{
			"blockhash":            base58.Encode(blockhash[:]),
			"lastValidBlockHeight": slot + 150,
		},
	}, nil
}

func (h *rpcHandler) sendTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	txStr, _ := args[0].(string)
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
	if encoding == "base64" {
		txBytes, err = base64.StdEncoding.DecodeString(txStr)
	} else {
		txBytes, err = base58.Decode(txStr)
	}
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Failed to decode transaction"}
	}

	tx, err := TransactionFromBytes(txBytes)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Failed to parse transaction"}
	}

	if err := h.vm.SubmitTransaction(tx); err != nil {
		return nil, &rpcError{Code: -32002, Message: err.Error()}
	}

	return base58.Encode(tx.Signature()), nil
}

func (h *rpcHandler) getTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	sigStr, _ := args[0].(string)
	sigBytes, err := base58.Decode(sigStr)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid signature"}
	}

	tx, err := h.vm.GetTransaction(sigBytes)
	if err != nil {
		return nil, nil
	}

	return map[string]interface{}{
		"slot":        h.vm.GetSlot(),
		"transaction": tx.ToData(),
		"meta": map[string]interface{}{
			"err": nil,
			"fee": tx.Result.Fee,
		},
	}, nil
}

func (h *rpcHandler) getMinRentExemption(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	json.Unmarshal(params, &args)

	dataLen := uint64(0)
	if len(args) > 0 {
		if dl, ok := args[0].(float64); ok {
			dataLen = uint64(dl)
		}
	}

	// Rent exempt formula: (128 + dataLen) * 3480 * 2
	return (128 + dataLen) * 3480 * 2, nil
}

// ========== Ethereum RPC Methods ==========

func (h *rpcHandler) dispatchEth(ctx context.Context, method string, params json.RawMessage) (interface{}, error) {
	switch method {
	case "eth_chainId":
		return h.ethChainId(ctx, params)
	case "eth_blockNumber":
		return h.ethBlockNumber(ctx, params)
	case "eth_getBalance":
		return h.ethGetBalance(ctx, params)
	case "eth_getTransactionCount":
		return h.ethGetTransactionCount(ctx, params)
	case "eth_getCode":
		return h.ethGetCode(ctx, params)
	case "eth_call":
		return h.ethCall(ctx, params)
	case "eth_estimateGas":
		return h.ethEstimateGas(ctx, params)
	case "eth_gasPrice":
		return "0x3b9aca00", nil // 1 gwei
	case "eth_maxPriorityFeePerGas":
		return "0x3b9aca00", nil
	case "eth_sendRawTransaction":
		return h.ethSendRawTransaction(ctx, params)
	case "eth_getTransactionByHash":
		return nil, nil
	case "eth_getTransactionReceipt":
		return nil, nil
	case "eth_getBlockByNumber":
		return h.ethGetBlockByNumber(ctx, params)
	case "eth_getBlockByHash":
		return nil, nil
	case "eth_getLogs":
		return []interface{}{}, nil
	case "eth_accounts":
		return []string{}, nil
	case "eth_syncing":
		return false, nil
	case "eth_coinbase":
		return "0x0000000000000000000000000000000000000000", nil
	case "eth_mining":
		return false, nil
	case "net_version":
		return fmt.Sprintf("%d", h.vm.config.ChainID), nil
	case "net_listening":
		return true, nil
	case "net_peerCount":
		return "0x1", nil
	case "web3_clientVersion":
		return "SVM-Subnet/v0.1.0", nil
	default:
		return nil, &rpcError{Code: -32601, Message: fmt.Sprintf("Method not found: %s", method)}
	}
}

func (h *rpcHandler) ethChainId(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return fmt.Sprintf("0x%x", h.vm.config.ChainID), nil
}

func (h *rpcHandler) ethBlockNumber(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return fmt.Sprintf("0x%x", h.vm.GetSlot()), nil
}

func (h *rpcHandler) ethGetBalance(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	addrStr, _ := args[0].(string)
	addrStr = strings.TrimPrefix(addrStr, "0x")
	addrBytes, err := hex.DecodeString(addrStr)
	if err != nil || len(addrBytes) != 20 {
		return nil, &rpcError{Code: -32602, Message: "Invalid address"}
	}

	// Convert EVM address to Solana pubkey
	var evmAddr [20]byte
	copy(evmAddr[:], addrBytes)
	solanaPubkey := evmToSolanaPubkey(evmAddr)

	account, err := h.vm.GetAccount(solanaPubkey)
	if err != nil {
		return "0x0", nil
	}

	// Convert lamports to wei (1 lamport = 1e9 wei)
	balance := new(big.Int).SetUint64(account.Lamports)
	balance.Mul(balance, big.NewInt(1e9))

	return "0x" + balance.Text(16), nil
}

func (h *rpcHandler) ethGetTransactionCount(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Track nonces
	return "0x0", nil
}

func (h *rpcHandler) ethGetCode(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	addrStr, _ := args[0].(string)
	addrStr = strings.TrimPrefix(addrStr, "0x")
	addrBytes, err := hex.DecodeString(addrStr)
	if err != nil || len(addrBytes) != 20 {
		return nil, &rpcError{Code: -32602, Message: "Invalid address"}
	}

	var evmAddr [20]byte
	copy(evmAddr[:], addrBytes)
	solanaPubkey := evmToSolanaPubkey(evmAddr)

	account, err := h.vm.GetAccount(solanaPubkey)
	if err != nil || !account.Executable {
		return "0x", nil
	}

	return "0x" + hex.EncodeToString(account.Data), nil
}

func (h *rpcHandler) ethSendRawTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	rawTxStr, _ := args[0].(string)
	rawTxStr = strings.TrimPrefix(rawTxStr, "0x")
	rawTx, err := hex.DecodeString(rawTxStr)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: "Invalid transaction hex"}
	}

	// Parse RLP transaction (simplified - real impl needs full RLP decoding)
	// For now, we'll just compute a hash and return it
	txHash := sha256Hash(rawTx)

	// TODO: Actually decode RLP and execute via hybrid executor
	// This requires implementing RLP decoding for legacy/EIP-1559/EIP-2930 txs

	return "0x" + hex.EncodeToString(txHash[:]), nil
}

func (h *rpcHandler) ethCall(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	callObj, ok := args[0].(map[string]interface{})
	if !ok {
		return nil, &rpcError{Code: -32602, Message: "Invalid call object"}
	}

	// Parse call parameters
	var caller [20]byte
	var to *[20]byte
	var data []byte
	var value []byte

	if from, ok := callObj["from"].(string); ok {
		fromBytes, _ := hex.DecodeString(strings.TrimPrefix(from, "0x"))
		if len(fromBytes) == 20 {
			copy(caller[:], fromBytes)
		}
	}

	if toStr, ok := callObj["to"].(string); ok && toStr != "" {
		toBytes, _ := hex.DecodeString(strings.TrimPrefix(toStr, "0x"))
		if len(toBytes) == 20 {
			var toAddr [20]byte
			copy(toAddr[:], toBytes)
			to = &toAddr
		}
	}

	if dataStr, ok := callObj["data"].(string); ok {
		data, _ = hex.DecodeString(strings.TrimPrefix(dataStr, "0x"))
	}

	if valueStr, ok := callObj["value"].(string); ok {
		value, _ = hex.DecodeString(strings.TrimPrefix(valueStr, "0x"))
	}

	// Execute via hybrid executor
	result, err := h.vm.ExecuteEVM(caller, to, value, data, 10_000_000)
	if err != nil {
		return nil, &rpcError{Code: -32000, Message: err.Error()}
	}

	if !result.Success {
		return nil, &rpcError{Code: -32000, Message: result.Error}
	}

	// Return empty result for now (real impl would return output bytes)
	return "0x", nil
}

func (h *rpcHandler) ethEstimateGas(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	callObj, ok := args[0].(map[string]interface{})
	if !ok {
		return nil, &rpcError{Code: -32602, Message: "Invalid call object"}
	}

	// Parse call parameters
	var caller [20]byte
	var to *[20]byte
	var data []byte
	var value []byte

	if from, ok := callObj["from"].(string); ok {
		fromBytes, _ := hex.DecodeString(strings.TrimPrefix(from, "0x"))
		if len(fromBytes) == 20 {
			copy(caller[:], fromBytes)
		}
	}

	if toStr, ok := callObj["to"].(string); ok && toStr != "" {
		toBytes, _ := hex.DecodeString(strings.TrimPrefix(toStr, "0x"))
		if len(toBytes) == 20 {
			var toAddr [20]byte
			copy(toAddr[:], toBytes)
			to = &toAddr
		}
	}

	if dataStr, ok := callObj["data"].(string); ok {
		data, _ = hex.DecodeString(strings.TrimPrefix(dataStr, "0x"))
	}

	if valueStr, ok := callObj["value"].(string); ok {
		value, _ = hex.DecodeString(strings.TrimPrefix(valueStr, "0x"))
	}

	// Execute to get gas used
	result, err := h.vm.ExecuteEVM(caller, to, value, data, 30_000_000)
	if err != nil {
		// Return default gas for simple transfer
		return "0x5208", nil // 21000
	}

	if !result.Success {
		// Return default gas
		return "0x5208", nil
	}

	// Add 20% buffer to actual gas used
	gasUsed := result.ComputeUnitsUsed
	gasWithBuffer := gasUsed + (gasUsed / 5)

	return fmt.Sprintf("0x%x", gasWithBuffer), nil
}

func sha256Hash(data []byte) [32]byte {
	h := sha256.Sum256(data)
	return h
}

func (h *rpcHandler) ethGetBlockByNumber(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) == 0 {
		return nil, &rpcError{Code: -32602, Message: "Invalid params"}
	}

	blockNumStr, _ := args[0].(string)
	var blockNum uint64

	if blockNumStr == "latest" || blockNumStr == "pending" {
		blockNum = h.vm.GetSlot()
	} else if blockNumStr == "earliest" {
		blockNum = 0
	} else {
		blockNumStr = strings.TrimPrefix(blockNumStr, "0x")
		fmt.Sscanf(blockNumStr, "%x", &blockNum)
	}

	blockHash, _ := h.vm.GetBlockhash(blockNum)
	parentHash := [32]byte{}
	if blockNum > 0 {
		parentHash, _ = h.vm.GetBlockhash(blockNum - 1)
	}

	return map[string]interface{}{
		"number":           fmt.Sprintf("0x%x", blockNum),
		"hash":             "0x" + hex.EncodeToString(blockHash[:]),
		"parentHash":       "0x" + hex.EncodeToString(parentHash[:]),
		"nonce":            "0x0000000000000000",
		"sha3Uncles":       "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
		"logsBloom":        "0x" + strings.Repeat("00", 256),
		"transactionsRoot": "0x" + strings.Repeat("00", 32),
		"stateRoot":        "0x" + strings.Repeat("00", 32),
		"receiptsRoot":     "0x" + strings.Repeat("00", 32),
		"miner":            "0x0000000000000000000000000000000000000000",
		"difficulty":       "0x0",
		"totalDifficulty":  "0x0",
		"extraData":        "0x",
		"size":             "0x0",
		"gasLimit":         "0x1c9c380", // 30M
		"gasUsed":          "0x0",
		"timestamp":        "0x0",
		"transactions":     []interface{}{},
		"uncles":           []interface{}{},
		"baseFeePerGas":    "0x3b9aca00",
	}, nil
}

// evmToSolanaPubkey converts 20-byte EVM address to 32-byte Solana pubkey
func evmToSolanaPubkey(addr [20]byte) [32]byte {
	var pubkey [32]byte
	// Pad EVM address to 32 bytes (left-pad with zeros)
	copy(pubkey[12:], addr[:])
	return pubkey
}
