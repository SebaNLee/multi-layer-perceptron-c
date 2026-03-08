#!/usr/bin/env bash

set -e
shopt -s extglob
ROOT_DIR=$(pwd)
TARGET=$1

# args validation
if [ "$TARGET" != "all" ] && [ "$TARGET" != "emnist" ]; then
    echo "Usage: ./datasets.sh [all|emnist]"
    exit 1
fi

# download
if [ "$TARGET" = "all" ] || [ "$TARGET" = "emnist" ]; then
    EMNIST_DIR="$ROOT_DIR/datasets/emnist"
    EMNIST_URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    rm -rf "$EMNIST_DIR"
    mkdir -p "$EMNIST_DIR"
    cd "$EMNIST_DIR"
    wget -q --show-progress "$EMNIST_URL" -O emnist.zip
    unzip -q -j emnist.zip
    rm emnist.zip
    rm !(emnist-digits-*|emnist-letters-*)
    gunzip -f *.gz
fi
# if [ "$TARGET" = "all" ] || [ "$TARGET" = "" ]; then
#     # TODO
# fi
