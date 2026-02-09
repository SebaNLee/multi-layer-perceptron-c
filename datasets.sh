#!/usr/bin/env bash

set -e
shopt -s extglob
ROOT_DIR=$(pwd)

# EMNIST
EMNIST_DIR="$ROOT_DIR/datasets/emnist"
EMNIST_URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
mkdir -p "$EMNIST_DIR"
cd "$EMNIST_DIR"
wget -q --show-progress "$EMNIST_URL" -O emnist.zip
unzip -q -j emnist.zip
rm emnist.zip
rm !(emnist-digits-*|emnist-letters-*)
gunzip -f *.gz
