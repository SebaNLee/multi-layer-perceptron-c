#!/usr/bin/env bash

set -e
shopt -s extglob
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"
TARGET=$1

# args validation
if [ "$TARGET" != "all" ] && [ "$TARGET" != "emnist" ] && [ "$TARGET" != "mushroom" ] && [ "$TARGET" != "meteorite" ] && [ "$TARGET" != "engine" ]; then
    echo "Usage: ./datasets.sh [all|emnist|mushroom|meteorite|engine]"
    exit 1
fi

# helper
wget_unzip() {
  local name="$1"
  local url="$2"
  local zip="${name}.zip"
  local dir="$DATASETS_DIR/$name"
  rm -rf "$dir"
  mkdir -p "$dir"
  (
    cd "$dir"
    wget -q --show-progress "$url" -O "$zip"
    unzip -q -j -o "$zip"
    rm -f "$zip"
  )
}

# download and process
if [ "$TARGET" = "all" ] || [ "$TARGET" = "emnist" ]; then
    wget_unzip "emnist" "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    (
        cd "$DATASETS_DIR/emnist"
        rm -f !(emnist-digits-*|emnist-letters-*)
        gunzip -f *.gz
    )
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "mushroom" ]; then
    wget_unzip "mushroom" "https://archive.ics.uci.edu/static/public/73/mushroom.zip"

fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "meteorite" ]; then
    rm -rf "$DATASETS_DIR/meteorite"
    mkdir -p "$DATASETS_DIR/meteorite"
    (
        cd "$DATASETS_DIR/meteorite"
        wget -q --show-progress "https://data.nasa.gov/docs/legacy/meteorite_landings/Meteorite_Landings.csv"
    )

fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "engine" ]; then
    wget_unzip "engine" "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

fi
