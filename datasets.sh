#!/usr/bin/env bash

set -e
shopt -s extglob
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"
TARGET=$1

# args validation
if [ "$TARGET" != "all" ] && [ "$TARGET" != "emnist" ] && [ "$TARGET" != "mushroom" ] && [ "$TARGET" != "shopper" ] && [ "$TARGET" != "heart" ]; then
    echo "Usage: ./datasets.sh [all|emnist|mushroom|shopper|heart]"
    exit 1
fi

# download
if [ "$TARGET" = "all" ] || [ "$TARGET" = "emnist" ]; then
    EMNIST_DIR="$DATASETS_DIR/emnist"
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
if [ "$TARGET" = "all" ] || [ "$TARGET" = "mushroom" ]; then
    MUSHROOM_DIR="$DATASETS_DIR/mushroom"
    MUSHROOM_URL="https://archive.ics.uci.edu/static/public/73/mushroom.zip"
    rm -rf "$MUSHROOM_DIR"
    mkdir -p "$MUSHROOM_DIR"
    cd "$MUSHROOM_DIR"
    wget -q --show-progress "$MUSHROOM_URL" -O mushroom.zip

fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "shopper" ]; then
    SHOPPER_DIR="$DATASETS_DIR/shopper"
    SHOPPER_URL="https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    rm -rf "$SHOPPER_DIR"
    mkdir -p "$SHOPPER_DIR"
    cd "$SHOPPER_DIR"
    wget -q --show-progress "$SHOPPER_URL" -O shopper.zip

fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "heart" ]; then
    HEART_DIR="$DATASETS_DIR/heart"
    HEART_URL="https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    rm -rf "$HEART_DIR"
    mkdir -p "$HEART_DIR"
    cd "$HEART_DIR"
    wget -q --show-progress "$HEART_URL" -O heart.zip

fi
