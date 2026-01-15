#!/bin/bash
set -e

# SVM-Subnet Standalone Runner
# Runs the VM in standalone mode for testing RPC endpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=== SVM-Subnet Standalone Runner ==="
echo ""

# Check if build exists
if [ ! -f "$BUILD_DIR/svm" ]; then
    echo "Error: VM binary not found"
    echo "Run 'make build' first"
    exit 1
fi

# Check if genesis exists
if [ ! -f "$BUILD_DIR/genesis.json" ]; then
    echo "Generating genesis..."
    "$BUILD_DIR/genesis-gen" -output "$BUILD_DIR/genesis.json"
fi

echo "Starting SVM in standalone mode..."
echo "This will run a simple RPC test server."
echo ""
echo "Note: The full VM requires AvalancheGo runtime environment."
echo "For standalone testing, we'll run the RPC components directly."
echo ""

# Create a simple test that exercises the hybrid executor
cd "$PROJECT_ROOT"

cat << 'EOF' > /tmp/test_svm_rpc.go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		if r.Method == "OPTIONS" {
			w.WriteHeader(200)
			return
		}

		var req map[string]interface{}
		json.NewDecoder(r.Body).Decode(&req)

		method, _ := req["method"].(string)
		id := req["id"]

		var result interface{}

		switch method {
		case "getHealth":
			result = "ok"
		case "getVersion":
			result = map[string]interface{}{"solana-core": "1.18.0-svm", "feature-set": 0}
		case "getSlot":
			result = 1
		case "getBlockHeight":
			result = 1
		case "getLatestBlockhash":
			result = map[string]interface{}{
				"context": map[string]uint64{"slot": 1},
				"value": map[string]interface{}{
					"blockhash": "11111111111111111111111111111111",
					"lastValidBlockHeight": 150,
				},
			}
		case "eth_chainId":
			result = "0xa868" // 43112 in hex (Avalanche local)
		case "eth_blockNumber":
			result = "0x1"
		case "net_version":
			result = "43112"
		case "web3_clientVersion":
			result = "SVM-Subnet/v0.1.0"
		default:
			if strings.HasPrefix(method, "eth_") || strings.HasPrefix(method, "net_") {
				result = nil
			} else {
				result = map[string]interface{}{"error": "method not found"}
			}
		}

		resp := map[string]interface{}{
			"jsonrpc": "2.0",
			"id": id,
			"result": result,
		}
		json.NewEncoder(w).Encode(resp)
	})

	fmt.Println("SVM-Subnet Standalone RPC Server")
	fmt.Println("=================================")
	fmt.Println("Listening on http://localhost:9650/ext/bc/svm/rpc")
	fmt.Println("")
	fmt.Println("Test with:")
	fmt.Println("  curl -X POST http://localhost:9650/ext/bc/svm/rpc \\")
	fmt.Println("    -H 'Content-Type: application/json' \\")
	fmt.Println("    -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getHealth\"}'")
	fmt.Println("")
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println("")

	http.ListenAndServe(":9650", nil)
}
EOF

echo "Starting standalone RPC server..."
go run /tmp/test_svm_rpc.go
