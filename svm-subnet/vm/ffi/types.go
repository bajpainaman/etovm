package ffi

import (
	"encoding/hex"
	"fmt"
)

// Pubkey represents a 32-byte public key (Solana-compatible)
type Pubkey [32]byte

// NewPubkey creates a pubkey from bytes
func NewPubkey(bytes []byte) (Pubkey, error) {
	if len(bytes) != 32 {
		return Pubkey{}, fmt.Errorf("invalid pubkey length: expected 32, got %d", len(bytes))
	}
	var pk Pubkey
	copy(pk[:], bytes)
	return pk, nil
}

// String returns the base58 string representation
func (pk Pubkey) String() string {
	// Simple hex encoding for now - would use base58 in production
	return hex.EncodeToString(pk[:])
}

// Bytes returns the raw bytes
func (pk Pubkey) Bytes() []byte {
	return pk[:]
}

// IsZero checks if the pubkey is all zeros
func (pk Pubkey) IsZero() bool {
	for _, b := range pk {
		if b != 0 {
			return false
		}
	}
	return true
}

// SystemProgram returns the system program pubkey (all zeros)
func SystemProgram() Pubkey {
	return Pubkey{}
}

// Account represents a Solana-style account
type Account struct {
	Lamports   uint64
	Data       []byte
	Owner      Pubkey
	Executable bool
	RentEpoch  uint64
}

// NewAccount creates a new account
func NewAccount(lamports uint64, space int, owner Pubkey) *Account {
	return &Account{
		Lamports:   lamports,
		Data:       make([]byte, space),
		Owner:      owner,
		Executable: false,
		RentEpoch:  0,
	}
}

// IsEmpty checks if the account has no lamports and no data
func (a *Account) IsEmpty() bool {
	return a.Lamports == 0 && len(a.Data) == 0
}

// Clone creates a copy of the account
func (a *Account) Clone() *Account {
	data := make([]byte, len(a.Data))
	copy(data, a.Data)

	return &Account{
		Lamports:   a.Lamports,
		Data:       data,
		Owner:      a.Owner,
		Executable: a.Executable,
		RentEpoch:  a.RentEpoch,
	}
}

// Instruction represents a single instruction
type Instruction struct {
	ProgramID Pubkey
	Accounts  []AccountMeta
	Data      []byte
}

// AccountMeta describes an account for an instruction
type AccountMeta struct {
	Pubkey     Pubkey
	IsSigner   bool
	IsWritable bool
}

// NewAccountMeta creates a writable account meta
func NewAccountMeta(pubkey Pubkey, isSigner bool) AccountMeta {
	return AccountMeta{
		Pubkey:     pubkey,
		IsSigner:   isSigner,
		IsWritable: true,
	}
}

// NewReadonlyAccountMeta creates a readonly account meta
func NewReadonlyAccountMeta(pubkey Pubkey, isSigner bool) AccountMeta {
	return AccountMeta{
		Pubkey:     pubkey,
		IsSigner:   isSigner,
		IsWritable: false,
	}
}

// Signature represents a 64-byte ed25519 signature
type Signature [64]byte

// String returns the hex string representation
func (s Signature) String() string {
	return hex.EncodeToString(s[:])
}

// IsZero checks if the signature is all zeros
func (s Signature) IsZero() bool {
	for _, b := range s {
		if b != 0 {
			return false
		}
	}
	return true
}

// Hash represents a 32-byte hash
type Hash [32]byte

// String returns the hex string representation
func (h Hash) String() string {
	return hex.EncodeToString(h[:])
}

// IsZero checks if the hash is all zeros
func (h Hash) IsZero() bool {
	for _, b := range h {
		if b != 0 {
			return false
		}
	}
	return true
}
