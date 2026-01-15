package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Simple RPC test server that mimics the SVM RPC interface
// This demonstrates the API without the full AvalancheGo stack

func main() {
	http.HandleFunc("/", handleRPC)
	http.HandleFunc("/ext/bc/svm/rpc", handleRPC)

	fmt.Println("=================================")
	fmt.Println("SVM-Subnet RPC Test Server")
	fmt.Println("=================================")
	fmt.Println("Listening on http://localhost:9650")
	fmt.Println("")
	fmt.Println("Solana-compatible RPC:")
	fmt.Println("  curl -X POST http://localhost:9650 \\")
	fmt.Println("    -H 'Content-Type: application/json' \\")
	fmt.Println("    -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getHealth\"}'")
	fmt.Println("")
	fmt.Println("EVM-compatible RPC:")
	fmt.Println("  curl -X POST http://localhost:9650 \\")
	fmt.Println("    -H 'Content-Type: application/json' \\")
	fmt.Println("    -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_chainId\"}'")
	fmt.Println("")
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println("")

	if err := http.ListenAndServe(":9650", nil); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}

func handleRPC(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		writeError(w, nil, -32600, "Only POST method is supported")
		return
	}

	var req struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      interface{}     `json:"id"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, nil, -32700, "Parse error")
		return
	}

	result, err := handleMethod(req.Method, req.Params)
	if err != nil {
		writeError(w, req.ID, -32603, err.Error())
		return
	}

	writeResult(w, req.ID, result)
}

func handleMethod(method string, params json.RawMessage) (interface{}, error) {
	switch method {
	// Solana-compatible methods
	case "getHealth":
		return "ok", nil

	case "getVersion":
		return map[string]interface{}{
			"solana-core": "1.18.0-svm",
			"feature-set": 0,
		}, nil

	case "getSlot":
		return 1, nil

	case "getBlockHeight":
		return 1, nil

	case "getLatestBlockhash":
		return map[string]interface{}{
			"context": map[string]uint64{"slot": 1},
			"value": map[string]interface{}{
				"blockhash":            "11111111111111111111111111111111",
				"lastValidBlockHeight": uint64(150),
			},
		}, nil

	case "getBalance":
		return map[string]interface{}{
			"context": map[string]uint64{"slot": 1},
			"value":   uint64(1000000000),
		}, nil

	// EVM-compatible methods
	case "eth_chainId":
		return "0xa868", nil // 43112 (Avalanche local)

	case "eth_blockNumber":
		return "0x1", nil

	case "net_version":
		return "43112", nil

	case "web3_clientVersion":
		return "SVM-Subnet/v0.1.0", nil

	case "eth_gasPrice":
		return "0x5d21dba00", nil // 25 gwei

	case "eth_getBalance":
		return "0xde0b6b3a7640000", nil // 1 ETH

	default:
		return nil, fmt.Errorf("method %s not found", method)
	}
}

func writeResult(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"result":  result,
	}
	json.NewEncoder(w).Encode(resp)
}

func writeError(w http.ResponseWriter, id interface{}, code int, message string) {
	resp := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
		},
	}
	json.NewEncoder(w).Encode(resp)
}
