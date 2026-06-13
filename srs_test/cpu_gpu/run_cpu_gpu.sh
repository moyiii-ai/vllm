#!/bin/bash
# Run CPU-GPU cudaMemcpy test (same data size and iterations as run_gdr.sh).
#   read:  CPU -> GPU (cudaMemcpy HostToDevice)
#   write: GPU -> CPU (cudaMemcpy DeviceToHost)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments (aligned with run_gdr.sh):
#   1: MODE = read | write
#   2: MSG_SIZE  (default: 33554432 = 32MB)
#   3: ITERATIONS (default: 1000)
#   4: GPU_ID (default: 4)

MODE=${1:-write}
if [[ "$MODE" != "read" && "$MODE" != "write" ]]; then
  echo "Usage: $0 read|write [MSG_SIZE] [ITERATIONS] [GPU_ID]"
  exit 2
fi

MSG_SIZE=${2:-33554432}
ITERATIONS=${3:-1000}
GPU_ID=${4:-4}

echo "================================================================"
echo "Run CPU-GPU cudaMemcpy $MODE"
echo "================================================================"
echo "Mode: $MODE ($([ "$MODE" = "write" ] && echo "GPU->CPU" || echo "CPU->GPU"))"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "GPU ID: $GPU_ID"
echo "================================================================"

cd "$SCRIPT_DIR"

# Ensure binary exists
if [[ ! -x "./cpu_to_gpu_dma" ]]; then
  echo "Building cpu_to_gpu_dma..."
  make cpu_to_gpu_dma
fi

./cpu_to_gpu_dma "$MODE" "$GPU_ID" "$MSG_SIZE" "$ITERATIONS"

echo "Done."
