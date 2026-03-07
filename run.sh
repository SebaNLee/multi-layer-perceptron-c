#!/usr/bin/env bash

set -e

MODE=$1
EXAMPLE=$2

# clean
rm -rf ./build
mkdir build

# configure CMake mode
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release

# compile
cmake --build build/debug
cmake --build build/release

# run
if [ "$MODE" = "debug" ]; then
    ./build/debug/examples/$EXAMPLE
elif [ "$MODE" = "release" ]; then
    ./build/release/examples/$EXAMPLE
else
    echo "Usage: ./run.sh [debug|release] <example>"
    exit 1
fi