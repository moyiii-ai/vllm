#!/bin/bash
# Usage: ./ncu.sh read|write

set -e

MODE=$1

if [ "$MODE" == "read_global" ]; then
    KERNEL="peerReadKernelV4"
    BENCH_SCRIPT="./run_simple_global.sh read"
    EXPORT_FILE="read_kernel_metrics"
elif [ "$MODE" == "write_global" ]; then
    KERNEL="peerWriteKernelV4"
    BENCH_SCRIPT="./run_simple_global.sh write"
    EXPORT_FILE="write_kernel_metrics"
elif [ "$MODE" == "read_copy" ]; then
    KERNEL="copyKernel"
    BENCH_SCRIPT="./run_simple_copy.sh read"
    EXPORT_FILE="read_kernel_metrics_copy"
elif [ "$MODE" == "write_copy" ]; then
    KERNEL="copyKernel"
    BENCH_SCRIPT="./run_simple_copy.sh write"
    EXPORT_FILE="write_kernel_metrics_copy"
else
    echo "Invalid argument: $MODE. Use 'read' or 'write'."
    exit 1
fi

echo "Running Nsight Compute for $MODE kernel..."

ncu \
  -k "$KERNEL" -f \
  --metrics pcie__read_bytes,pcie__write_bytes \
  --print-summary=per-kernel \
  --export="$EXPORT_FILE" \
  $BENCH_SCRIPT

echo "Exported metrics to $EXPORT_FILE"
