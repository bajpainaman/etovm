#!/bin/bash
set -e

# Configuration
AVALANCHEGO_PATH=${AVALANCHEGO_PATH:-"avalanchego"}
DATA_DIR=${DATA_DIR:-"/tmp/svm-subnet-data"}
HTTP_PORT=${HTTP_PORT:-9650}
STAKING_PORT=${STAKING_PORT:-9651}

# Clean previous data
rm -rf $DATA_DIR
mkdir -p $DATA_DIR

echo "Starting AvalancheGo with SVM-Subnet..."

$AVALANCHEGO_PATH \
    --network-id=local \
    --data-dir=$DATA_DIR \
    --http-port=$HTTP_PORT \
    --staking-port=$STAKING_PORT \
    --log-level=info \
    --log-display-level=info

echo "AvalancheGo started on http://localhost:$HTTP_PORT"
