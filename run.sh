#!/usr/bin/env bash

set -e

MODE=$1
EXAMPLE=$2

# clean
rm -rf ./build
mkdir build

if [ "$MODE" = "debug" ]; then
    BUILD_DIR=build/debug
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug
    cmake --build "$BUILD_DIR" --target "$EXAMPLE"
    ./"$BUILD_DIR"/examples/"$EXAMPLE"
elif [ "$MODE" = "release" ]; then
    BUILD_DIR=build/release
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target "$EXAMPLE"
    ./"$BUILD_DIR"/examples/"$EXAMPLE"
else
    echo "Usage: ./run.sh [debug|release] <example>"
    exit 1
fi
