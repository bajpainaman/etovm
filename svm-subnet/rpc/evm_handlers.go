package rpc

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
)

// dispatchEVM routes Ethereum RPC methods
func (s *Server) dispatchEVM(ctx context.Context, method string, params json.RawMessage) (interface{}, error) {
	switch method {
	// eth_ namespace
	case "eth_chainId":
		return s.ethChainId(ctx, params)
	case "eth_blockNumber":
		return s.ethBlockNumber(ctx, params)
	case "eth_getBalance":
		return s.ethGetBalance(ctx, params)
	case "eth_getTransactionCount":
		return s.ethGetTransactionCount(ctx, params)
	case "eth_getCode":
		return s.ethGetCode(ctx, params)
	case "eth_getStorageAt":
		return s.ethGetStorageAt(ctx, params)
	case "eth_call":
		return s.ethCall(ctx, params)
	case "eth_estimateGas":
		return s.ethEstimateGas(ctx, params)
	case "eth_gasPrice":
		return s.ethGasPrice(ctx, params)
	case "eth_maxPriorityFeePerGas":
		return s.ethMaxPriorityFeePerGas(ctx, params)
	case "eth_feeHistory":
		return s.ethFeeHistory(ctx, params)
	case "eth_sendRawTransaction":
		return s.ethSendRawTransaction(ctx, params)
	case "eth_getTransactionByHash":
		return s.ethGetTransactionByHash(ctx, params)
	case "eth_getTransactionReceipt":
		return s.ethGetTransactionReceipt(ctx, params)
	case "eth_getBlockByNumber":
		return s.ethGetBlockByNumber(ctx, params)
	case "eth_getBlockByHash":
		return s.ethGetBlockByHash(ctx, params)
	case "eth_getLogs":
		return s.ethGetLogs(ctx, params)
	case "eth_accounts":
		return s.ethAccounts(ctx, params)
	case "eth_syncing":
		return s.ethSyncing(ctx, params)
	case "eth_coinbase":
		return s.ethCoinbase(ctx, params)
	case "eth_mining":
		return s.ethMining(ctx, params)
	case "eth_hashrate":
		return s.ethHashrate(ctx, params)

	// net_ namespace
	case "net_version":
		return s.netVersion(ctx, params)
	case "net_listening":
		return s.netListening(ctx, params)
	case "net_peerCount":
		return s.netPeerCount(ctx, params)

	// web3_ namespace
	case "web3_clientVersion":
		return s.web3ClientVersion(ctx, params)
	case "web3_sha3":
		return s.web3Sha3(ctx, params)

	default:
		return nil, &RPCError{Code: MethodNotFound, Message: fmt.Sprintf("Method not found: %s", method)}
	}
}

// ========== Utility Functions ==========

func hexToUint64(s string) (uint64, error) {
	s = strings.TrimPrefix(s, "0x")
	var result uint64
	_, err := fmt.Sscanf(s, "%x", &result)
	return result, err
}

func uint64ToHex(n uint64) string {
	return fmt.Sprintf("0x%x", n)
}

func bigIntToHex(n *big.Int) string {
	if n == nil {
		return "0x0"
	}
	return "0x" + n.Text(16)
}

func hexToBigInt(s string) *big.Int {
	s = strings.TrimPrefix(s, "0x")
	n := new(big.Int)
	n.SetString(s, 16)
	return n
}

func bytesToHex(b []byte) string {
	return "0x" + hex.EncodeToString(b)
}

func hexToBytes(s string) ([]byte, error) {
	s = strings.TrimPrefix(s, "0x")
	return hex.DecodeString(s)
}

func hash32ToHex(h [32]byte) string {
	return "0x" + hex.EncodeToString(h[:])
}

func addr20ToHex(a [20]byte) string {
	return "0x" + hex.EncodeToString(a[:])
}

// ========== eth_ Methods ==========

func (s *Server) ethChainId(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return uint64ToHex(s.chainID), nil
}

func (s *Server) ethBlockNumber(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return uint64ToHex(s.evm.BlockNumber()), nil
}

func (s *Server) ethGetBalance(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	addrStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address"}
	}

	addr, err := HexToEVMAddress(addrStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address format"}
	}

	balance := s.evm.GetBalance(addr)
	return bigIntToHex(balance), nil
}

func (s *Server) ethGetTransactionCount(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	addrStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address"}
	}

	addr, err := HexToEVMAddress(addrStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address format"}
	}

	nonce := s.evm.GetNonce(addr)
	return uint64ToHex(nonce), nil
}

func (s *Server) ethGetCode(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	addrStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address"}
	}

	addr, err := HexToEVMAddress(addrStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address format"}
	}

	code := s.evm.GetCode(addr)
	if code == nil {
		return "0x", nil
	}
	return bytesToHex(code), nil
}

func (s *Server) ethGetStorageAt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 2 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	addrStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address"}
	}

	posStr, ok := args[1].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid position"}
	}

	addr, err := HexToEVMAddress(addrStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid address format"}
	}

	posBytes, err := hexToBytes(posStr)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid position format"}
	}

	var pos [32]byte
	copy(pos[32-len(posBytes):], posBytes)

	value := s.evm.GetStorageAt(addr, pos)
	return hash32ToHex(value), nil
}

func (s *Server) ethCall(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement EVM call execution
	// For now, return empty result
	return "0x", nil
}

func (s *Server) ethEstimateGas(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// Return standard gas estimate
	return uint64ToHex(21000), nil
}

func (s *Server) ethGasPrice(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return bigIntToHex(s.evm.GasPrice()), nil
}

func (s *Server) ethMaxPriorityFeePerGas(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// Return 1 gwei tip
	return "0x3b9aca00", nil
}

func (s *Server) ethFeeHistory(ctx context.Context, params json.RawMessage) (interface{}, error) {
	blockCount := uint64(1)
	currentBlock := s.evm.BlockNumber()

	baseFees := make([]string, blockCount+1)
	gasUsedRatios := make([]float64, blockCount)
	oldestBlock := currentBlock - blockCount + 1

	for i := uint64(0); i <= blockCount; i++ {
		baseFees[i] = bigIntToHex(s.evm.BaseFee())
	}

	for i := uint64(0); i < blockCount; i++ {
		gasUsedRatios[i] = 0.5 // 50% utilization
	}

	return map[string]interface{}{
		"oldestBlock":   uint64ToHex(oldestBlock),
		"baseFeePerGas": baseFees,
		"gasUsedRatio":  gasUsedRatios,
	}, nil
}

func (s *Server) ethSendRawTransaction(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	txHex, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid transaction"}
	}

	txBytes, err := hexToBytes(txHex)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid transaction hex"}
	}

	// TODO: Parse RLP-encoded transaction and submit to SVM
	// For now, return a mock tx hash
	_ = txBytes

	return "0x0000000000000000000000000000000000000000000000000000000000000000", nil
}

func (s *Server) ethGetTransactionByHash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement transaction lookup
	return nil, nil
}

func (s *Server) ethGetTransactionReceipt(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement receipt lookup
	return nil, nil
}

func (s *Server) ethGetBlockByNumber(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	blockNumStr, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid block number"}
	}

	var blockNum uint64
	if blockNumStr == "latest" || blockNumStr == "pending" {
		blockNum = s.evm.BlockNumber()
	} else if blockNumStr == "earliest" {
		blockNum = 0
	} else {
		var err error
		blockNum, err = hexToUint64(blockNumStr)
		if err != nil {
			return nil, &RPCError{Code: InvalidParams, Message: "Invalid block number format"}
		}
	}

	return s.buildBlockResponse(blockNum, false)
}

func (s *Server) ethGetBlockByHash(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement block by hash lookup
	return nil, nil
}

func (s *Server) buildBlockResponse(blockNum uint64, fullTx bool) (interface{}, error) {
	blockHash := s.evm.BlockHash(blockNum)
	parentHash := [32]byte{}
	if blockNum > 0 {
		parentHash = s.evm.BlockHash(blockNum - 1)
	}

	return map[string]interface{}{
		"number":           uint64ToHex(blockNum),
		"hash":             hash32ToHex(blockHash),
		"parentHash":       hash32ToHex(parentHash),
		"nonce":            "0x0000000000000000",
		"mixHash":          hash32ToHex([32]byte{}),
		"sha3Uncles":       "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
		"logsBloom":        "0x" + strings.Repeat("00", 256),
		"transactionsRoot": hash32ToHex([32]byte{}),
		"stateRoot":        hash32ToHex([32]byte{}),
		"receiptsRoot":     hash32ToHex([32]byte{}),
		"miner":            "0x0000000000000000000000000000000000000000",
		"difficulty":       "0x0",
		"totalDifficulty":  "0x0",
		"extraData":        "0x",
		"size":             "0x0",
		"gasLimit":         uint64ToHex(30000000),
		"gasUsed":          "0x0",
		"timestamp":        uint64ToHex(uint64(0)), // TODO: Get from block
		"transactions":     []interface{}{},
		"uncles":           []interface{}{},
		"baseFeePerGas":    bigIntToHex(s.evm.BaseFee()),
	}, nil
}

func (s *Server) ethGetLogs(ctx context.Context, params json.RawMessage) (interface{}, error) {
	// TODO: Implement log filtering
	return []interface{}{}, nil
}

func (s *Server) ethAccounts(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return []string{}, nil
}

func (s *Server) ethSyncing(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return false, nil
}

func (s *Server) ethCoinbase(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return "0x0000000000000000000000000000000000000000", nil
}

func (s *Server) ethMining(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return false, nil
}

func (s *Server) ethHashrate(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return "0x0", nil
}

// ========== net_ Methods ==========

func (s *Server) netVersion(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return fmt.Sprintf("%d", s.chainID), nil
}

func (s *Server) netListening(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return true, nil
}

func (s *Server) netPeerCount(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return "0x1", nil
}

// ========== web3_ Methods ==========

func (s *Server) web3ClientVersion(ctx context.Context, params json.RawMessage) (interface{}, error) {
	return "SVM-Subnet/v0.1.0", nil
}

func (s *Server) web3Sha3(ctx context.Context, params json.RawMessage) (interface{}, error) {
	var args []interface{}
	if err := json.Unmarshal(params, &args); err != nil || len(args) < 1 {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid params"}
	}

	dataHex, ok := args[0].(string)
	if !ok {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid data"}
	}

	data, err := hexToBytes(dataHex)
	if err != nil {
		return nil, &RPCError{Code: InvalidParams, Message: "Invalid data hex"}
	}

	// Use keccak256 (not sha3-256)
	// For now, use sha256 as placeholder
	hash := [32]byte{}
	copy(hash[:], data)

	return hash32ToHex(hash), nil
}
