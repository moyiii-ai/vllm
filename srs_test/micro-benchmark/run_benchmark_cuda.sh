#!/bin/bash

SRC_FILE="benchmark_cuda.cu"
BIN_FILE="benchmark_cuda"

if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

echo "Compiling $SRC_FILE ..."
nvcc -O2 -o $BIN_FILE $SRC_FILE
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Running benchmark..."
./$BIN_FILE
