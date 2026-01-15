#!/bin/bash
set -e

# SVM-Subnet Plugin Installer
# Installs the SVM VM plugin to AvalancheGo plugins directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

# VM ID (base58 encoded 32-byte hash of "svm")
VM_ID="srEXiWaHq7HVqKPT3HTPM8iSdLVatYC49pLVfxzdjwc9hLKm2"

# Default plugin directory
PLUGIN_DIR="${AVALANCHEGO_PLUGIN_DIR:-$HOME/.avalanchego/plugins}"

echo "=== SVM-Subnet Plugin Installer ==="
echo "Project root: $PROJECT_ROOT"
echo "VM ID: $VM_ID"
echo "Plugin directory: $PLUGIN_DIR"
echo ""

# Check if build exists
if [ ! -f "$BUILD_DIR/svm" ]; then
    echo "Error: VM binary not found at $BUILD_DIR/svm"
    echo "Run 'make build' first"
    exit 1
fi

# Create plugin directory if needed
mkdir -p "$PLUGIN_DIR"

# Copy plugin
echo "Installing SVM plugin..."
cp "$BUILD_DIR/svm" "$PLUGIN_DIR/$VM_ID"
chmod +x "$PLUGIN_DIR/$VM_ID"

echo "Plugin installed successfully!"
echo ""
echo "=== Next Steps ==="
echo "1. Start AvalancheGo with the SVM subnet"
echo ""
echo "2. Create a subnet using avalanche-cli or network-runner"
echo ""
echo "3. Or run standalone for testing:"
echo "   cd $PROJECT_ROOT && ./scripts/run-standalone.sh"
echo ""
