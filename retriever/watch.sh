#! /bin/bash

set -e

# Check if `entr` is installed
if ! command -v entr &> /dev/null; then
    # Clone https://github.com/eradman/entr to temp directory
    echo "entr could not be found, installing..."
    TMP_DIR=$(mktemp -d)
    git clone https://github.com/eradman/entr $TMP_DIR
    cd $TMP_DIR
    ./configure
    make test
    sudo make install
fi

# Catch Ctrl-C and exit
trap "exit" INT

# Temporarily set working directory to the root of the repo
# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
    "$SCRIPT_DIR/watch-file-list.sh" | entr -d -r "$SCRIPT_DIR/build-and-run.sh"
done

