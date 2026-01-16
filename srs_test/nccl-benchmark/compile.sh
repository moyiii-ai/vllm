#!/bin/bash
# Simple compilation script for NCCL benchmark
# Uses conda environment for NCCL

CONDA_ENV="${CONDA_ENV:-/home/ruoshi/miniconda3/envs/nccl229}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

echo "Using NCCL from: $CONDA_ENV"
echo "Using CUDA from: $CUDA_PATH"

nvcc -O3 -arch=sm_70 -std=c++11 \
  -I"$CONDA_ENV/include" \
  -I"$CUDA_PATH/include" \
  nccl_allgather_inplace.cu \
  -o nccl_allgather_inplace \
  -L"$CONDA_ENV/lib" \
  -L"$CUDA_PATH/lib64" \
  -lnccl -lcudart -lpthread

if [ $? -eq 0 ]; then
  echo "Compilation successful!"
  echo "Run with: ./nccl_allgather_inplace [count_per_rank] [iters]"
else
  echo "Compilation failed!"
  exit 1
fi
