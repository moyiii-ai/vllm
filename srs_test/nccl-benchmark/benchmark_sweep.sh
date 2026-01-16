#!/bin/bash
# Generic batch benchmark script to test different buffer sizes
# Usage: ./benchmark_sweep.sh [operation] [output_file]
#   operation: allgather (default) or allreduce

OPERATION="${1:-allgather}"
OUTPUT_FILE="${2:-benchmark_results_${OPERATION}.txt}"

# Validate operation
if [ "$OPERATION" != "allgather" ] && [ "$OPERATION" != "allreduce" ]; then
    echo "Error: operation must be 'allgather' or 'allreduce'"
    echo "Usage: ./benchmark_sweep.sh [allgather|allreduce] [output_file]"
    exit 1
fi

# Set executable based on operation
if [ "$OPERATION" == "allgather" ]; then
    EXECUTABLE="./nccl_allgather_inplace"
    OP_NAME="All-Gather"
else
    EXECUTABLE="./nccl_allreduce"
    OP_NAME="All-Reduce"
fi

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please build it first with 'make $OPERATION'"
    exit 1
fi

echo "=== NCCL ${OP_NAME} Buffer Size Sweep ===" > "$OUTPUT_FILE"
echo "Start time: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Define buffer sizes to test (in elements per rank)
# You can modify this array to test different sizes
SIZES=(
    1024          # 4 KB
    4096          # 16 KB
    16384         # 64 KB
    65536         # 256 KB
    262144        # 1 MB
    1048576       # 4 MB
    4194304       # 16 MB
    16777216      # 64 MB
    67108864      # 256 MB
    268435456     # 1 GB
)

# Number of iterations for each test
ITERS=100

echo "Testing ${OP_NAME} with buffer sizes (elements per rank):"
for size in "${SIZES[@]}"; do
    echo "  - $size elements ($(echo "scale=2; $size * 4 / 1024 / 1024" | bc) MB per rank)"
done
echo ""

echo "Running benchmarks..." | tee -a "$OUTPUT_FILE"
echo "----------------------------------------" >> "$OUTPUT_FILE"

# Run benchmark for each size
for size in "${SIZES[@]}"; do
    echo "" >> "$OUTPUT_FILE"
    echo "Buffer size: $size elements ($(echo "scale=2; $size * 4 / 1024 / 1024" | bc) MB per rank)" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    
    echo "Testing size: $size elements..."
    
    # Run the benchmark and append output to file
    "$EXECUTABLE" "$size" "$ITERS" 2>&1 | tee -a "$OUTPUT_FILE"
    
    # Add separator
    echo "" >> "$OUTPUT_FILE"
done

echo "" >> "$OUTPUT_FILE"
echo "End time: $(date)" >> "$OUTPUT_FILE"
echo "=== Benchmark Complete ===" >> "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
