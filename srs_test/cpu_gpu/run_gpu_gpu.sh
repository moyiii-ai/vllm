#!/bin/bash
# Run GPU-GPU cudaMemcpyPeer test between GPU3 and GPU4 (same data size and iterations as run_gdr.sh).
#   read:  GPU4 reads from GPU3 (GPU3 -> GPU4)
#   write: GPU4 writes to GPU3 (GPU4 -> GPU3)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments (aligned with run_gdr.sh):
#   1: MODE = read | write
#   2: MSG_SIZE  (default: 33554432 = 32MB)
#   3: ITERATIONS (default: 1000)

MODE=${1:-write}
if [[ "$MODE" != "read" && "$MODE" != "write" ]]; then
  echo "Usage: $0 read|write [MSG_SIZE] [ITERATIONS]"
  exit 2
fi

MSG_SIZE=${2:-33554432}
ITERATIONS=${3:-1000}

# GPU3 and GPU4
GPU_SRC=4
GPU_DST=3

echo "================================================================"
echo "Run GPU-GPU cudaMemcpyPeer $MODE (GPU3 <-> GPU4)"
echo "================================================================"
echo "Mode: $MODE (GPU4 $MODE = $([ "$MODE" = "write" ] && echo "GPU4->GPU3" || echo "GPU3->GPU4"))"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

cd "$SCRIPT_DIR"

# Ensure binary exists
if [[ ! -x "./gpu_to_gpu_dma" ]]; then
  echo "Building gpu_to_gpu_dma..."
  make gpu_to_gpu_dma
fi

# gpu_to_gpu_dma <read|write> <source_gpu_id> <dest_gpu_id> [size] [iterations]
# For write: GPU4 writes (GPU4 -> GPU3) → source=4, dest=3
# For read: GPU4 reads (GPU3 -> GPU4) → source=4, dest=3 (code swaps internally for read)
./gpu_to_gpu_dma "$MODE" "$GPU_SRC" "$GPU_DST" "$MSG_SIZE" "$ITERATIONS"

echo "Done."
