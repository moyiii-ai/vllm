#!/bin/bash

SRC_FILE="benchmark_global.cu"
BIN_FILE="benchmark_global"

if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

GPU_COMPUTE="compute_80"
GPU_CODE="sm_80"

echo "Compiling $SRC_FILE for $GPU_COMPUTE/$GPU_CODE ..."
nvcc -O2 -arch=$GPU_COMPUTE -code=$GPU_CODE -o $BIN_FILE $SRC_FILE
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
    