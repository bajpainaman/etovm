// E2E benchmark for testing the full transaction pipeline
// This sends actual transactions via sendTransaction RPC to trigger block building
package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Transaction structures matching vm/tx.go
type TransactionHeader struct {
	NumRequiredSignatures       uint8 `json:"numRequiredSignatures"`
	NumReadonlySignedAccounts   uint8 `json:"numReadonlySignedAccounts"`
	NumReadonlyUnsignedAccounts uint8 `json:"numReadonlyUnsignedAccounts"`
}

type CompiledInstruction struct {
	ProgramIDIndex uint8   `json:"programIdIndex"`
	Accounts       []uint8 `json:"accounts"`
	Data           []byte  `json:"data"`
}

type TransactionMessage struct {
	Header          TransactionHeader     `json:"header"`
	AccountKeys     [][32]byte            `json:"accountKeys"`
	RecentBlockhash [32]byte              `json:"recentBlockhash"`
	Instructions    []CompiledInstruction `json:"instructions"`
}

type TransactionData struct {
	Signatures [][]byte            `json:"signatures"`
	Message    *TransactionMessage `json:"message"`
}

// RPC request/response
type rpcRequest struct {
	Jsonrpc string        `json:"jsonrpc"`
	ID      int           `json:"id"`
	Method  string        `json:"method"`
	Params  []interface{} `json:"params"`
}

type rpcResponse struct {
	Jsonrpc string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Result  interface{} `json:"result"`
	Error   *rpcError   `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// System program ID (all zeros for transfers)
var systemProgramID = [32]byte{}

func main() {
	rpcURL := flag.String("rpc", "http://54.164.38.47:9650/ext/bc/MkLTPmBgEX6zQC45WNaWqXtLP1PrAQ135pkJBxKmKZAkjaPfJ/rpc", "RPC endpoint URL")
	numTxs := flag.Int("txs", 1000, "Number of transactions to send")
	numWorkers := flag.Int("workers", 64, "Number of parallel workers")
	privateKeyHex := flag.String("privkey", "fff90a45e092f38919eda5a69fb73b072fb208161021618cdf83918cf512910d", "Private key in hex")
	flag.Parse()

	// Parse private key
	privKeyBytes, err := hex.DecodeString(*privateKeyHex)
	if err != nil {
		fmt.Printf("Failed to decode private key: %v\n", err)
		return
	}

	// Generate Ed25519 keypair from seed
	privateKey := ed25519.NewKeyFromSeed(privKeyBytes)
	publicKey := privateKey.Public().(ed25519.PublicKey)

	var senderPubkey [32]byte
	copy(senderPubkey[:], publicKey)

	fmt.Printf("E2E Transaction Benchmark\n")
	fmt.Printf("=========================\n")
	fmt.Printf("RPC:     %s\n", *rpcURL)
	fmt.Printf("Sender:  %x\n", senderPubkey[:8])
	fmt.Printf("Txs:     %d\n", *numTxs)
	fmt.Printf("Workers: %d\n\n", *numWorkers)

	// Get current slot for blockhash
	slot, err := getSlot(*rpcURL)
	if err != nil {
		fmt.Printf("Failed to get slot: %v\n", err)
		return
	}
	fmt.Printf("Initial slot: %d\n\n", slot)

	// Create a simple blockhash (in production, use getRecentBlockhash)
	var blockhash [32]byte
	blockhash[0] = byte(slot)
	blockhash[1] = byte(slot >> 8)

	// Create transactions
	fmt.Println("Creating transactions...")
	txs := make([]*TransactionData, *numTxs)
	for i := 0; i < *numTxs; i++ {
		tx := createTransferTx(privateKey, senderPubkey, blockhash, uint64(i))
		txs[i] = tx
	}
	fmt.Printf("Created %d transactions\n\n", len(txs))

	// Send transactions in parallel
	var sent, succeeded, failed atomic.Int64
	var wg sync.WaitGroup
	txChan := make(chan *TransactionData, *numWorkers*2)

	start := time.Now()

	// Start workers
	for w := 0; w < *numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := &http.Client{Timeout: 5 * time.Second}
			for tx := range txChan {
				err := sendTransaction(client, *rpcURL, tx)
				sent.Add(1)
				if err != nil {
					failed.Add(1)
				} else {
					succeeded.Add(1)
				}
			}
		}()
	}

	// Feed transactions
	for _, tx := range txs {
		txChan <- tx
	}
	close(txChan)

	// Wait for completion
	wg.Wait()
	elapsed := time.Since(start)

	// Results
	fmt.Println("\n=== RESULTS ===")
	fmt.Printf("Time:       %v\n", elapsed)
	fmt.Printf("Sent:       %d\n", sent.Load())
	fmt.Printf("Succeeded:  %d\n", succeeded.Load())
	fmt.Printf("Failed:     %d\n", failed.Load())
	fmt.Printf("RPC TPS:    %.0f\n", float64(sent.Load())/elapsed.Seconds())

	// Check final slot
	time.Sleep(500 * time.Millisecond)
	finalSlot, err := getSlot(*rpcURL)
	if err != nil {
		fmt.Printf("Failed to get final slot: %v\n", err)
		return
	}
	fmt.Printf("\nSlot: %d -> %d (built %d blocks)\n", slot, finalSlot, finalSlot-slot)
}

func createTransferTx(privateKey ed25519.PrivateKey, sender [32]byte, blockhash [32]byte, nonce uint64) *TransactionData {
	// Create recipient (just a deterministic address for testing)
	var recipient [32]byte
	recipient[0] = byte(nonce)
	recipient[1] = byte(nonce >> 8)
	recipient[2] = byte(nonce >> 16)
	recipient[31] = 1 // Different from sender

	// Transfer instruction data: [2] (transfer) + [lamports as u64]
	lamports := uint64(1000) // 1000 lamports
	instructionData := make([]byte, 12)
	instructionData[0] = 2 // Transfer instruction
	// Little-endian u64 for lamports
	for i := 0; i < 8; i++ {
		instructionData[4+i] = byte(lamports >> (i * 8))
	}

	msg := &TransactionMessage{
		Header: TransactionHeader{
			NumRequiredSignatures:       1,
			NumReadonlySignedAccounts:   0,
			NumReadonlyUnsignedAccounts: 1,
		},
		AccountKeys:     [][32]byte{sender, recipient, systemProgramID},
		RecentBlockhash: blockhash,
		Instructions: []CompiledInstruction{
			{
				ProgramIDIndex: 2, // System program
				Accounts:       []uint8{0, 1}, // sender, recipient
				Data:           instructionData,
			},
		},
	}

	// Create message bytes for signing
	msgBytes, _ := json.Marshal(msg)

	// Sign the message
	signature := ed25519.Sign(privateKey, msgBytes)

	return &TransactionData{
		Signatures: [][]byte{signature},
		Message:    msg,
	}
}

func sendTransaction(client *http.Client, rpcURL string, tx *TransactionData) error {
	// Serialize transaction to JSON, then base64 encode
	txBytes, err := json.Marshal(tx)
	if err != nil {
		return err
	}
	txBase64 := base64.StdEncoding.EncodeToString(txBytes)

	req := rpcRequest{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  "sendTransaction",
		Params:  []interface{}{txBase64, map[string]string{"encoding": "base64"}},
	}

	reqBody, _ := json.Marshal(req)
	resp, err := client.Post(rpcURL, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var rpcResp rpcResponse
	if err := json.Unmarshal(body, &rpcResp); err != nil {
		return err
	}

	if rpcResp.Error != nil {
		return fmt.Errorf("RPC error: %s", rpcResp.Error.Message)
	}

	return nil
}

func getSlot(rpcURL string) (uint64, error) {
	req := rpcRequest{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  "getSlot",
		Params:  []interface{}{},
	}

	reqBody, _ := json.Marshal(req)
	resp, err := http.Post(rpcURL, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var rpcResp rpcResponse
	if err := json.Unmarshal(body, &rpcResp); err != nil {
		return 0, err
	}

	if rpcResp.Error != nil {
		return 0, fmt.Errorf("RPC error: %s", rpcResp.Error.Message)
	}

	switch v := rpcResp.Result.(type) {
	case float64:
		return uint64(v), nil
	default:
		return 0, fmt.Errorf("unexpected slot type: %T", v)
	}
}
