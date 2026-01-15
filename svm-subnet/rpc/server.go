package rpc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/eto-chain/svm-subnet/vm"
)

// Server implements a unified JSON-RPC server supporting both Solana and Ethereum APIs
type Server struct {
	vm     *vm.VM
	evm    *EVMBackend
	server *http.Server
	mu     sync.RWMutex

	// Config
	chainID uint64
}

// Config holds RPC server configuration
type Config struct {
	Host    string
	Port    int
	ChainID uint64
}

// NewServer creates a new unified RPC server
func NewServer(svm *vm.VM, evmBackend *EVMBackend, cfg *Config) *Server {
	s := &Server{
		vm:      svm,
		evm:     evmBackend,
		chainID: cfg.ChainID,
	}

	mux := http.NewServeMux()

	// Unified endpoint - auto-detects Solana vs Ethereum methods
	mux.HandleFunc("/", s.handleRPC)

	// Explicit endpoints for clarity
	mux.HandleFunc("/solana", s.handleSolanaRPC)
	mux.HandleFunc("/evm", s.handleEVMRPC)

	s.server = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
		Handler: corsMiddleware(mux),
	}

	return s
}

// Start starts the RPC server
func (s *Server) Start() error {
	return s.server.ListenAndServe()
}

// StartAsync starts the RPC server in background
func (s *Server) StartAsync() {
	go s.server.ListenAndServe()
}

// Stop gracefully stops the RPC server
func (s *Server) Stop(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

// Addr returns the server address
func (s *Server) Addr() string {
	return s.server.Addr
}

// RPCRequest represents a JSON-RPC request
type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// RPCResponse represents a JSON-RPC response
type RPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
}

// RPCError represents a JSON-RPC error
type RPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Error implements the error interface
func (e *RPCError) Error() string {
	return e.Message
}

// Standard JSON-RPC error codes
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// Solana-specific error codes
const (
	TransactionError    = -32002
	SimulationFailed    = -32003
	BlockNotAvailable   = -32004
	NodeUnhealthy       = -32005
	TransactionNotFound = -32006
	SlotSkipped         = -32007
)

// Ethereum-specific error codes
const (
	ExecutionError    = -32015
	TransactionReject = -32016
)

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// handleRPC auto-detects and routes to appropriate handler
func (s *Server) handleRPC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, nil, ParseError, "Parse error", nil)
		return
	}

	if req.JSONRPC != "2.0" {
		s.writeError(w, req.ID, InvalidRequest, "Invalid Request", nil)
		return
	}

	// Auto-detect based on method prefix
	var result interface{}
	var err error

	if isEthereumMethod(req.Method) {
		result, err = s.dispatchEVM(r.Context(), req.Method, req.Params)
	} else {
		result, err = s.dispatchSolana(r.Context(), req.Method, req.Params)
	}

	if err != nil {
		if rpcErr, ok := err.(*RPCError); ok {
			s.writeError(w, req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		} else {
			s.writeError(w, req.ID, InternalError, err.Error(), nil)
		}
		return
	}

	s.writeResult(w, req.ID, result)
}

// handleSolanaRPC handles Solana-specific RPC
func (s *Server) handleSolanaRPC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, nil, ParseError, "Parse error", nil)
		return
	}

	result, err := s.dispatchSolana(r.Context(), req.Method, req.Params)
	if err != nil {
		if rpcErr, ok := err.(*RPCError); ok {
			s.writeError(w, req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		} else {
			s.writeError(w, req.ID, InternalError, err.Error(), nil)
		}
		return
	}

	s.writeResult(w, req.ID, result)
}

// handleEVMRPC handles Ethereum-specific RPC
func (s *Server) handleEVMRPC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, nil, ParseError, "Parse error", nil)
		return
	}

	result, err := s.dispatchEVM(r.Context(), req.Method, req.Params)
	if err != nil {
		if rpcErr, ok := err.(*RPCError); ok {
			s.writeError(w, req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		} else {
			s.writeError(w, req.ID, InternalError, err.Error(), nil)
		}
		return
	}

	s.writeResult(w, req.ID, result)
}

// isEthereumMethod checks if method is an Ethereum JSON-RPC method
func isEthereumMethod(method string) bool {
	prefixes := []string{"eth_", "net_", "web3_", "debug_", "txpool_", "personal_"}
	for _, prefix := range prefixes {
		if strings.HasPrefix(method, prefix) {
			return true
		}
	}
	return false
}

func (s *Server) writeResult(w http.ResponseWriter, id interface{}, result interface{}) {
	resp := RPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) writeError(w http.ResponseWriter, id interface{}, code int, message string, data interface{}) {
	resp := RPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &RPCError{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
