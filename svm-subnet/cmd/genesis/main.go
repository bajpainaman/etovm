package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/eto-chain/svm-subnet/genesis"
)

func main() {
	outputFile := flag.String("output", "genesis.json", "Output file path")
	chainID := flag.Uint64("chain-id", 43114, "Chain ID")
	networkID := flag.Uint("network-id", 1, "Network ID")
	flag.Parse()

	g := genesis.DefaultGenesis()
	g.ChainID = *chainID
	g.NetworkID = uint32(*networkID)

	data, err := genesis.ToJSON(g)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to marshal genesis: %v\n", err)
		os.Exit(1)
	}

	if *outputFile == "-" {
		fmt.Println(string(data))
	} else {
		if err := os.WriteFile(*outputFile, data, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write genesis: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Genesis written to %s\n", *outputFile)
	}
}
