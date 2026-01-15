package vm

import (
	"github.com/ava-labs/avalanchego/utils/logging"
	"github.com/ava-labs/avalanchego/vms"
)

var _ vms.Factory = (*Factory)(nil)

// Factory implements vms.Factory for creating VM instances
type Factory struct{}

// New returns a new VM instance
func (*Factory) New(log logging.Logger) (interface{}, error) {
	return NewVM(), nil
}
