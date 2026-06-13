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
if [ $# -eq 1 ]; then
    if [ "$1" = "read" ] || [ "$1" = "write" ]; then
        ./$BIN_FILE "$1"
    else
        echo "Invalid argument. Use: $0 [read|write]"
        exit 1
    fi
else
    ./$BIN_FILE
fi
    