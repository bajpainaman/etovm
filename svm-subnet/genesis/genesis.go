package genesis

import (
	"encoding/base64"
	"encoding/json"
	"time"

	"github.com/eto-chain/svm-subnet/vm"
	"github.com/mr-tron/base58"
)

// Well-known program IDs (Solana compatible)
var (
	// System Program - handles account creation, transfers
	SystemProgramID = mustPubkey("11111111111111111111111111111111")

	// BPF Loader - loads and executes BPF programs
	BPFLoaderProgramID = mustPubkey("BPFLoader2111111111111111111111111111111111")

	// Native Loader - loads native programs
	NativeLoaderID = mustPubkey("NativeLoader1111111111111111111111111111111")

	// Sysvar IDs
	ClockSysvarID = mustPubkey("SysvarC1ock11111111111111111111111111111111")
	RentSysvarID  = mustPubkey("SysvarRent111111111111111111111111111111111")

	// Token Program (SPL Token compatible)
	TokenProgramID = mustPubkey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

	// Associated Token Program
	AssociatedTokenProgramID = mustPubkey("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
)

func mustPubkey(s string) [32]byte {
	bytes, err := base58.Decode(s)
	if err != nil {
		panic(err)
	}
	var pubkey [32]byte
	copy(pubkey[:], bytes)
	return pubkey
}

func pubkeyToBase58(pk [32]byte) string {
	return base58.Encode(pk[:])
}

// DefaultGenesis creates a default genesis configuration
func DefaultGenesis() *vm.Genesis {
	return &vm.Genesis{
		Timestamp: time.Now().Unix(),
		ChainID:   43114,
		NetworkID: 1,
		Accounts:  DefaultAccounts(),
	}
}

// DefaultAccounts returns the default genesis accounts
func DefaultAccounts() []vm.GenesisAccount {
	accounts := []vm.GenesisAccount{
		// System Program
		{
			Pubkey:     SystemProgramID,
			Lamports:   1,
			Data:       []byte("system_program"),
			Owner:      NativeLoaderID,
			Executable: true,
		},
		// BPF Loader
		{
			Pubkey:     BPFLoaderProgramID,
			Lamports:   1,
			Data:       []byte("bpf_loader"),
			Owner:      NativeLoaderID,
			Executable: true,
		},
		// Native Loader
		{
			Pubkey:     NativeLoaderID,
			Lamports:   1,
			Data:       []byte("native_loader"),
			Owner:      NativeLoaderID,
			Executable: true,
		},
		// Clock Sysvar
		{
			Pubkey:     ClockSysvarID,
			Lamports:   1,
			Data:       makeClockData(),
			Owner:      SystemProgramID,
			Executable: false,
		},
		// Rent Sysvar
		{
			Pubkey:     RentSysvarID,
			Lamports:   1,
			Data:       makeRentData(),
			Owner:      SystemProgramID,
			Executable: false,
		},
		// Token Program
		{
			Pubkey:     TokenProgramID,
			Lamports:   1,
			Data:       []byte("spl_token"),
			Owner:      BPFLoaderProgramID,
			Executable: true,
		},
		// Associated Token Program
		{
			Pubkey:     AssociatedTokenProgramID,
			Lamports:   1,
			Data:       []byte("associated_token"),
			Owner:      BPFLoaderProgramID,
			Executable: true,
		},
	}

	// Add funded test accounts
	testAccounts := []struct {
		pubkey   string
		lamports uint64
	}{
		// Faucet account with lots of funds
		{"Faucet11111111111111111111111111111111111111", 1_000_000_000_000_000}, // 1M SOL equivalent
		// Stress test wallet (private key: fff90a45e092f38919eda5a69fb73b072fb208161021618cdf83918cf512910d)
		{"6ZrQwARijYWKZZAXe88D97mQqSqqiuBd2n59KmQRvik6", 100_000_000_000_000}, // 100K SOL
		// Test accounts
		{"Test1111111111111111111111111111111111111111", 100_000_000_000}, // 100 SOL
		{"Test2222222222222222222222222222222222222222", 100_000_000_000}, // 100 SOL
		{"Test3333333333333333333333333333333333333333", 100_000_000_000}, // 100 SOL
	}

	for _, ta := range testAccounts {
		accounts = append(accounts, vm.GenesisAccount{
			Pubkey:     mustPubkey(ta.pubkey),
			Lamports:   ta.lamports,
			Data:       nil,
			Owner:      SystemProgramID,
			Executable: false,
		})
	}

	return accounts
}

func makeClockData() []byte {
	return make([]byte, 40) // Empty clock sysvar
}

func makeRentData() []byte {
	data := make([]byte, 17)
	// lamports_per_byte_year = 3480
	lamportsPerByteYear := uint64(3480)
	data[0] = byte(lamportsPerByteYear)
	data[1] = byte(lamportsPerByteYear >> 8)
	// exemption_threshold = 2.0 (IEEE 754)
	data[8] = 0x00
	data[9] = 0x00
	data[10] = 0x00
	data[11] = 0x00
	data[12] = 0x00
	data[13] = 0x00
	data[14] = 0x00
	data[15] = 0x40
	// burn_percent = 50
	data[16] = 50
	return data
}

// ToJSON converts genesis to JSON with base58-encoded pubkeys
func ToJSON(g *vm.Genesis) ([]byte, error) {
	jsonGenesis := vm.GenesisJSON{
		Timestamp: g.Timestamp,
		ChainID:   g.ChainID,
		NetworkID: g.NetworkID,
		Accounts:  make([]vm.GenesisAccountJSON, len(g.Accounts)),
	}

	for i, acc := range g.Accounts {
		dataStr := ""
		if len(acc.Data) > 0 {
			dataStr = base64.StdEncoding.EncodeToString(acc.Data)
		}

		jsonGenesis.Accounts[i] = vm.GenesisAccountJSON{
			Pubkey:     pubkeyToBase58(acc.Pubkey),
			Lamports:   acc.Lamports,
			Data:       dataStr,
			Owner:      pubkeyToBase58(acc.Owner),
			Executable: acc.Executable,
		}
	}

	return json.MarshalIndent(jsonGenesis, "", "  ")
}

// FromJSON parses genesis from JSON
func FromJSON(data []byte) (*vm.Genesis, error) {
	var jsonGenesis vm.GenesisJSON
	if err := json.Unmarshal(data, &jsonGenesis); err != nil {
		return nil, err
	}

	g := &vm.Genesis{
		Timestamp: jsonGenesis.Timestamp,
		ChainID:   jsonGenesis.ChainID,
		NetworkID: jsonGenesis.NetworkID,
		Accounts:  make([]vm.GenesisAccount, len(jsonGenesis.Accounts)),
	}

	for i, acc := range jsonGenesis.Accounts {
		pubkeyBytes, _ := base58.Decode(acc.Pubkey)
		ownerBytes, _ := base58.Decode(acc.Owner)

		var pubkey, owner [32]byte
		copy(pubkey[:], pubkeyBytes)
		copy(owner[:], ownerBytes)

		var accData []byte
		if acc.Data != "" {
			accData, _ = base64.StdEncoding.DecodeString(acc.Data)
		}

		g.Accounts[i] = vm.GenesisAccount{
			Pubkey:     pubkey,
			Lamports:   acc.Lamports,
			Data:       accData,
			Owner:      owner,
			Executable: acc.Executable,
		}
	}

	return g, nil
}
