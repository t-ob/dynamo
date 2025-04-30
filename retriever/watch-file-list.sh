#! /bin/bash

# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUST_FILES=$(find "$SCRIPT_DIR/../dynamo/components" "$SCRIPT_DIR/../dynamo/lib" "$SCRIPT_DIR/../dynamo/launch" -type f -name "*.rs")
PYTHON_FILES=$(find "$SCRIPT_DIR/src/retriever" -type f -name "*.py")
BASH_FILES=$(find "$SCRIPT_DIR" -type f -name "*.sh")
YAML_FILES=$(find "$SCRIPT_DIR/src/retriever" -type f -name "*.yaml")

echo "${RUST_FILES[@]}"
echo "${PYTHON_FILES[@]}"
echo "${BASH_FILES[@]}"
echo "${YAML_FILES[@]}"
