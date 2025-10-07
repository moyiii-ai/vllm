#!/bin/bash

SRC_FILE="benchmark_copy.cu"
BIN_FILE="benchmark_copy"

if ! command -v nvcc &> /dev/null
then
    echo "Error: nvcc not found. Please install CUDA Toolkit first."
    exit 1
fi

GPU_COMPUTE="compute_80"
GPU_CODE="sm_80"

echo "=== Compiling $SRC_FILE for GPU architecture $GPU_COMPUTE/$GPU_CODE ==="
nvcc -O2 -arch=$GPU_COMPUTE -code=$GPU_CODE -o $BIN_FILE $SRC_FILE

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed! Please check your CUDA code or GPU architecture settings."
    exit 1
fi

echo -e "\n=== Running $BIN_FILE benchmark ==="
if [ $# -eq 1 ]; then
    if [ "$1" = "read" ] || [ "$1" = "write" ] || [ "$1" = "both" ]; then
        ./$BIN_FILE "$1"
    else
        echo "Error: Invalid argument. Supported arguments: read / write / both"
        echo "Usage: $0 [read|write|both] (default: both)"
        exit 1
    fi
else
    echo "No argument provided. Running default mode: both (write + read)"
    ./$BIN_FILE "both"
fi

if [ $? -eq 0 ]; then
    echo -e "\n=== Benchmark finished successfully ==="
else
    echo -e "\nError: Benchmark execution failed!"
    exit 1
fi