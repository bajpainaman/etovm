package test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/ava-labs/avalanchego/database/memdb"
	"github.com/ava-labs/avalanchego/ids"
	"github.com/ava-labs/avalanchego/snow"
	"github.com/ava-labs/avalanchego/snow/engine/common"
	"github.com/ava-labs/avalanchego/utils/logging"
	"github.com/ava-labs/avalanchego/utils/set"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/eto-chain/svm-subnet/vm"
)

// Genesis structure for testing
type TestGenesis struct {
	Timestamp int64            `json:"timestamp"`
	ChainID   uint64           `json:"chainId"`
	NetworkID uint64           `json:"networkId"`
	Accounts  []TestAccount    `json:"accounts"`
}

type TestAccount struct {
	Pubkey     string `json:"pubkey"`
	Lamports   uint64 `json:"lamports"`
	Data       string `json:"data,omitempty"`
	Owner      string `json:"owner"`
	Executable bool   `json:"executable,omitempty"`
}

// MockAppSender implements common.AppSender for testing
type MockAppSender struct{}

func (m *MockAppSender) SendAppRequest(ctx context.Context, nodeIDs set.Set[ids.NodeID], requestID uint32, appRequestBytes []byte) error {
	return nil
}
func (m *MockAppSender) SendAppResponse(ctx context.Context, nodeID ids.NodeID, requestID uint32, appResponseBytes []byte) error {
	return nil
}
func (m *MockAppSender) SendAppError(ctx context.Context, nodeID ids.NodeID, requestID uint32, errorCode int32, errorMessage string) error {
	return nil
}
func (m *MockAppSender) SendAppGossip(ctx context.Context, config common.SendConfig, appGossipBytes []byte) error {
	return nil
}

func TestVMInitializeTimestamp(t *testing.T) {
	// Create test genesis with a specific timestamp
	genesisTimestamp := int64(1704067200) // Jan 1, 2024

	genesis := TestGenesis{
		Timestamp: genesisTimestamp,
		ChainID:   43114,
		NetworkID: 1,
		Accounts: []TestAccount{
			{
				Pubkey:     "11111111111111111111111111111111",
				Lamports:   1,
				Owner:      "NativeLoader1111111111111111111111111111111",
				Executable: true,
			},
		},
	}

	genesisBytes, err := json.Marshal(genesis)
	if err != nil {
		t.Fatalf("Failed to marshal genesis: %v", err)
	}
	t.Logf("Genesis JSON: %s", string(genesisBytes))

	// Create in-memory database
	db := memdb.New()
	defer db.Close()

	// Create snow context
	snowCtx := &snow.Context{
		Log: logging.NoLog{},
	}

	// Create VM
	svmVM := vm.NewVM()

	// Initialize
	err = svmVM.Initialize(
		context.Background(),
		snowCtx,
		db,
		genesisBytes,
		nil, // upgrade bytes
		nil, // config bytes
		nil, // fxs
		&MockAppSender{},
	)
	if err != nil {
		t.Fatalf("VM Initialize failed: %v", err)
	}
	t.Log("VM initialized successfully")

	// Get last accepted (simulating what SDK does)
	lastAcceptedID, err := svmVM.LastAccepted(context.Background())
	if err != nil {
		t.Fatalf("LastAccepted failed: %v", err)
	}
	t.Logf("LastAccepted ID: %s", lastAcceptedID.String())

	// Get block (simulating what SDK does)
	block, err := svmVM.GetBlock(context.Background(), lastAcceptedID)
	if err != nil {
		t.Fatalf("GetBlock failed: %v", err)
	}
	t.Logf("Block height: %d", block.Height())

	// Get timestamp (this is what the SDK sends in InitializeResponse)
	blockTimestamp := block.Timestamp()
	t.Logf("Block timestamp: %s (Unix: %d)", blockTimestamp.String(), blockTimestamp.Unix())

	// Verify timestamp is valid
	if blockTimestamp.IsZero() {
		t.Error("Block timestamp is zero!")
	}

	// Simulate what grpcutils.TimestampFromTime does
	protoTimestamp := timestamppb.New(blockTimestamp)
	t.Logf("Proto timestamp: %+v", protoTimestamp)
	t.Logf("Proto timestamp IsValid: %v", protoTimestamp.IsValid())

	// Verify proto timestamp is valid
	if protoTimestamp == nil {
		t.Error("Proto timestamp is nil!")
	}

	if err := protoTimestamp.CheckValid(); err != nil {
		t.Errorf("Proto timestamp check failed: %v", err)
	}

	// The expected timestamp should match genesis
	expectedUnix := genesisTimestamp
	if blockTimestamp.Unix() != expectedUnix {
		t.Errorf("Timestamp mismatch: expected %d, got %d", expectedUnix, blockTimestamp.Unix())
	}

	t.Log("All timestamp checks passed!")
}

func TestProtobufTimestampEdgeCases(t *testing.T) {
	testCases := []struct {
		name string
		time time.Time
	}{
		{"Unix epoch", time.Unix(0, 0)},
		{"One second after epoch", time.Unix(1, 0)},
		{"Negative", time.Unix(-1, 0)},
		{"Genesis timestamp", time.Unix(1704067200, 0)},
		{"Go zero time", time.Time{}},
		{"Now", time.Now()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			proto := timestamppb.New(tc.time)
			t.Logf("Time: %s (Unix: %d) -> Proto: %+v, Valid: %v",
				tc.time.String(), tc.time.Unix(), proto, proto.IsValid())

			if proto == nil {
				t.Error("Proto timestamp is nil")
			}

			if err := proto.CheckValid(); err != nil {
				t.Logf("CheckValid error (expected for some cases): %v", err)
			}
		})
	}
}
