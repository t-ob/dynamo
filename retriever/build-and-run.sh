#! /bin/bash

set -e

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd "$SCRIPT_DIR/../dynamo"

CARGO_TARGET_DIR=$HOME/dynamo/.build/target cargo build -p file://$(pwd)/components/http
CARGO_TARGET_DIR=$HOME/dynamo/.build/target cargo build -p llmctl

popd

PYTHONPATH="$SCRIPT_DIR/src" dynamo serve retriever.graph:Frontend -f "$SCRIPT_DIR/src/retriever/config.yaml"
