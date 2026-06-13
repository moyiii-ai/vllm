#!/bin/bash

# sudo apt-get update
# sudo apt-get install -y librdmacm-dev libibverbs-dev

rm -rf CMakeCache.txt CMakeFiles/ bin/
cmake .
make