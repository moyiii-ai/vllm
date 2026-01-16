#!/bin/bash
# Run script that sets up library paths automatically

CONDA_ENV="${CONDA_ENV:-/home/ruoshi/miniconda3/envs/nccl229}"
CUDA_PATH="${CUDA_PATH:-$(dirname $(dirname $(which nvcc 2>/dev/null)) 2>/dev/null || echo /usr/local/cuda)}"

# Set library paths
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"

# Run the benchmark with all arguments passed through
./nccl_allgather_inplace "$@"
