#! /bin/bash

set -e

source ~/.bashrc

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

# Check if `ngc` is installed
if ! command -v ngc &> /dev/null; then
    echo "ngc could not be found, installing..."

    pushd $HOME

    wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_linux.zip -O ngccli_linux.zip && unzip ngccli_linux.zip
    chmod u+x ngc-cli/ngc
    echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bashrc && source ~/.bashrc
    rm ngccli_linux.zip

    popd
fi

# Check if `cuda-python[all]` is installed
if ! python -c "import cuda_python" &> /dev/null; then
    echo "cuda-python could not be found, installing..."
    python3 -m pip install cuda-python[all]
fi

# Check if `tensorrt==10.9.0.34` is installed
if ! python -c "import tensorrt" &> /dev/null; then
    echo "tensorrt could not be found, installing..."
    python3 -m pip install tensorrt==10.9.0.34
fi



# Catch Ctrl-C and exit
trap "exit" INT

# Temporarily set working directory to the root of the repo
# Get the absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
    "$SCRIPT_DIR/watch-file-list.sh" | entr -d -r "$SCRIPT_DIR/build-and-run.sh"
done

