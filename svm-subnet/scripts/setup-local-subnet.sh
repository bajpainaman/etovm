#!/bin/bash
set -e

# SVM-Subnet Local Deployment Script
# Uses avalanche-cli to deploy our custom SVM VM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
AVALANCHE_CLI="$HOME/bin/avalanche"

# VM configuration
VM_NAME="svm"
VM_ID="srEXiWaHq7HVqKPT3HTPM8iSdLVatYC49pLVfxzdjwc9hLKm2"
SUBNET_NAME="svmSubnet"

echo "=== SVM-Subnet Local Deployment ==="
echo ""

# Check prerequisites
if [ ! -f "$BUILD_DIR/svm" ]; then
    echo "Error: VM binary not found at $BUILD_DIR/svm"
    echo "Run 'make build' first"
    exit 1
fi

if [ ! -f "$AVALANCHE_CLI" ]; then
    echo "Error: avalanche-cli not found"
    exit 1
fi

# Ensure plugin is installed
PLUGIN_DIR="$HOME/.avalanchego/plugins"
mkdir -p "$PLUGIN_DIR"
cp "$BUILD_DIR/svm" "$PLUGIN_DIR/$VM_ID"
chmod +x "$PLUGIN_DIR/$VM_ID"
echo "✓ VM plugin installed to $PLUGIN_DIR/$VM_ID"

# Create genesis configuration for the subnet
GENESIS_FILE="$BUILD_DIR/genesis.json"
if [ ! -f "$GENESIS_FILE" ]; then
    cat > "$GENESIS_FILE" << 'GENESIS'
{
  "timestamp": 1704067200,
  "accounts": [
    {
      "pubkey": "11111111111111111111111111111111",
      "lamports": 1000000000000,
      "data": "",
      "owner": "11111111111111111111111111111111",
      "executable": false
    }
  ]
}
GENESIS
    echo "✓ Genesis configuration created"
fi

# Create subnet config file for avalanche-cli
CONFIG_DIR="$HOME/.avalanche-cli"
mkdir -p "$CONFIG_DIR/vms"

cat > "$CONFIG_DIR/vms/${VM_NAME}.json" << VMCONFIG
{
  "vmName": "${VM_NAME}",
  "vmId": "${VM_ID}",
  "vmPath": "${PLUGIN_DIR}/${VM_ID}"
}
VMCONFIG

echo "✓ VM configuration registered"

# Check if subnet already exists
if $AVALANCHE_CLI subnet list 2>/dev/null | grep -q "$SUBNET_NAME"; then
    echo "Subnet '$SUBNET_NAME' already exists"
else
    echo ""
    echo "Creating subnet with custom VM..."
    echo "This will use a custom VM configuration."
fi

echo ""
echo "=== Deployment Ready ==="
echo ""
echo "To deploy the SVM subnet locally, run:"
echo ""
echo "  1. Start local network:"
echo "     $AVALANCHE_CLI network start"
echo ""
echo "  2. Create subnet with custom VM:"
echo "     $AVALANCHE_CLI subnet create $SUBNET_NAME --custom --vm-path $BUILD_DIR/svm --genesis $GENESIS_FILE"
echo ""
echo "  3. Deploy to local network:"
echo "     $AVALANCHE_CLI subnet deploy $SUBNET_NAME --local"
echo ""
echo "  4. Test RPC endpoint (after deployment):"
echo "     curl -X POST http://127.0.0.1:9650/ext/bc/$SUBNET_NAME/rpc \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getHealth\"}'"
echo ""
