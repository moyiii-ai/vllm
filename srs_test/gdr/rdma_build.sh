#!/bin/bash

# Only need to install these packages once
# sudo apt-get update
# sudo apt-get install -y librdmacm-dev libibverbs-dev

rm -rf CMakeCache.txt CMakeFiles/ bin/
cmake .
make