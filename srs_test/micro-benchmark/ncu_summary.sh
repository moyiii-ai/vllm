#!/bin/bash
# Usage: ./ncu_summary.sh read|write

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <read|write>"
    exit 1
fi

MODE=$1

if [ "$MODE" == "read" ]; then
    NCU_REP_FILE="read_kernel_metrics.ncu-rep"
    OUTPUT_FILE="ncu_read.result"
elif [ "$MODE" == "write" ]; then
    NCU_REP_FILE="write_kernel_metrics.ncu-rep"
    OUTPUT_FILE="ncu_write.result"
elif [ "$MODE" == "read_copy" ]; then
    NCU_REP_FILE="read_kernel_metrics_copy.ncu-rep"
    OUTPUT_FILE="ncu_read_copy.result"
elif [ "$MODE" == "write_copy" ]; then
    NCU_REP_FILE="write_kernel_metrics_copy.ncu-rep"
    OUTPUT_FILE="ncu_write_copy.result"
else
    echo "Invalid argument: $MODE. Use 'read' or 'write'."
    exit 1
fi

if [ ! -f "$NCU_REP_FILE" ]; then
    echo "File not found: $NCU_REP_FILE"
    exit 1
fi

echo "Analyzing $NCU_REP_FILE ..."

ncu \
  --import "$NCU_REP_FILE" \
  > "$OUTPUT_FILE"