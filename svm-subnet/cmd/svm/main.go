package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ava-labs/avalanchego/utils/logging"
	"github.com/ava-labs/avalanchego/utils/ulimit"
	"github.com/ava-labs/avalanchego/vms/rpcchainvm"

	"github.com/eto-chain/svm-subnet/vm"
)

func main() {
	version := printVersion()
	fmt.Fprint(os.Stderr, version)

	// Set resource limits
	if err := ulimit.Set(ulimit.DefaultFDLimit, logging.NoLog{}); err != nil {
		fmt.Fprintf(os.Stderr, "failed to set fd limit: %v\n", err)
		os.Exit(1)
	}

	// Create VM instance
	svmVM := vm.NewVM()

	// Serve the VM via gRPC
	if err := rpcchainvm.Serve(context.Background(), svmVM); err != nil {
		fmt.Fprintf(os.Stderr, "failed to serve vm: %v\n", err)
		os.Exit(1)
	}
}

func printVersion() string {
	version := fmt.Sprintf(
		"SVM-Subnet VM\n"+
			"  Version:    %s\n"+
			"  VM ID:      %s\n",
		vm.VersionStr,
		vm.Name,
	)
	return version
}
