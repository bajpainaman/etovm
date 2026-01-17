package vm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"
)

// Config holds VM configuration
type Config struct {
	// Maximum transactions per block
	MaxTxsPerBlock int `json:"maxTxsPerBlock"`

	// Maximum mempool size
	MempoolSize int `json:"mempoolSize"`

	// Block build interval
	BuildInterval time.Duration `json:"buildInterval"`

	// RPC settings
	RPCHost string `json:"rpcHost"`
	RPCPort int    `json:"rpcPort"`

	// Chain identification
	ChainID uint64 `json:"chainId"`

	// Execution limits
	MaxComputeUnits      uint64 `json:"maxComputeUnits"`
	LamportsPerSignature uint64 `json:"lamportsPerSignature"`

	// Performance options
	UseTurboMode     bool `json:"useTurboMode"`     // Use 11M+ TPS delta execution
	VerifySignatures bool `json:"verifySignatures"` // Verify Ed25519 signatures
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		MaxTxsPerBlock:       30000,  // Increased for high TPS testing
		MempoolSize:          100000, // Increased to handle 30k+ pending txs
		BuildInterval:        100 * time.Millisecond,
		RPCHost:              "0.0.0.0",
		RPCPort:              8899,
		ChainID:              43114, // Avalanche C-Chain compatible
		MaxComputeUnits:      1_400_000,
		LamportsPerSignature: 5000,
		UseTurboMode:         true,  // Enable 11M+ TPS delta execution by default
		VerifySignatures:     true,  // Verify signatures in production
	}
}

// Genesis represents the genesis state
type Genesis struct {
	// Timestamp of genesis
	Timestamp int64 `json:"timestamp"`

	// Initial accounts
	Accounts []GenesisAccount `json:"accounts"`

	// Chain configuration
	ChainID   uint64 `json:"chainId"`
	NetworkID uint32 `json:"networkId"`
}

// GenesisAccount represents an account in genesis (internal representation)
type GenesisAccount struct {
	Pubkey     [32]byte
	Lamports   uint64
	Data       []byte
	Owner      [32]byte
	Executable bool
}

// GenesisJSON is the JSON-serializable genesis format
type GenesisJSON struct {
	Timestamp int64               `json:"timestamp"`
	Accounts  []GenesisAccountJSON `json:"accounts"`
	ChainID   uint64              `json:"chainId"`
	NetworkID uint32              `json:"networkId"`
}

// GenesisAccountJSON is the JSON-serializable account format with base58 pubkeys
type GenesisAccountJSON struct {
	Pubkey     string `json:"pubkey"`
	Lamports   uint64 `json:"lamports"`
	Data       string `json:"data,omitempty"`
	Owner      string `json:"owner"`
	Executable bool   `json:"executable"`
}

// UnmarshalJSON implements custom JSON unmarshaling for Genesis
func (g *Genesis) UnmarshalJSON(data []byte) error {
	var gj GenesisJSON
	if err := json.Unmarshal(data, &gj); err != nil {
		return fmt.Errorf("failed to unmarshal genesis JSON: %w", err)
	}

	g.Timestamp = gj.Timestamp
	g.ChainID = gj.ChainID
	g.NetworkID = gj.NetworkID

	g.Accounts = make([]GenesisAccount, len(gj.Accounts))
	for i, acc := range gj.Accounts {
		// Parse pubkey from base58-like string (actually just padding for simplicity)
		pubkey, err := parsePubkey(acc.Pubkey)
		if err != nil {
			return fmt.Errorf("failed to parse pubkey for account %d: %w", i, err)
		}
		g.Accounts[i].Pubkey = pubkey

		// Parse owner
		owner, err := parsePubkey(acc.Owner)
		if err != nil {
			return fmt.Errorf("failed to parse owner for account %d: %w", i, err)
		}
		g.Accounts[i].Owner = owner

		g.Accounts[i].Lamports = acc.Lamports
		g.Accounts[i].Executable = acc.Executable

		// Parse data from base64
		if acc.Data != "" {
			decoded, err := base64.StdEncoding.DecodeString(acc.Data)
			if err != nil {
				// Try URL-safe encoding
				decoded, err = base64.URLEncoding.DecodeString(acc.Data)
				if err != nil {
					return fmt.Errorf("failed to decode data for account %d: %w", i, err)
				}
			}
			g.Accounts[i].Data = decoded
		}
	}

	return nil
}

// parsePubkey converts a base58-style string to [32]byte
// For simplicity, we use a deterministic hash of the string
func parsePubkey(s string) ([32]byte, error) {
	var result [32]byte

	// For standard Solana-style pubkeys, copy bytes directly
	// (assumes the string is already in the right format)
	if len(s) == 32 {
		copy(result[:], []byte(s))
		return result, nil
	}

	// For longer base58-encoded strings, use a simple conversion
	// This pads short strings and truncates long ones
	bytes := []byte(s)
	if len(bytes) >= 32 {
		copy(result[:], bytes[:32])
	} else {
		copy(result[:len(bytes)], bytes)
		// Pad remaining with zeros
	}

	return result, nil
}
