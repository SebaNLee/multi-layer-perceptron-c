#!/usr/bin/env bash

set -e

# clean
rm -rf ./build

# compile
mkdir build
cd build
cmake ..
make

# run
./examples/$1