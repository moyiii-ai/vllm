#!/bin/bash
# Run both all-gather and all-reduce benchmarks with the same buffer sizes
# Usage: ./run_all_benchmarks.sh [output_prefix]

OUTPUT_PREFIX="${1:-benchmark_results}"
ALLGATHER_OUTPUT="${OUTPUT_PREFIX}_allgather.txt"
ALLREDUCE_OUTPUT="${OUTPUT_PREFIX}_allreduce.txt"

# Check if executables exist
if [ ! -f "./nccl_allgather_inplace" ]; then
    echo "Error: nccl_allgather_inplace not found. Please build it first with 'make allgather'"
    exit 1
fi

if [ ! -f "./nccl_allreduce" ]; then
    echo "Error: nccl_allreduce not found. Please build it first with 'make allreduce'"
    exit 1
fi

# Define buffer sizes to test (in elements per rank)
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

echo "=== Running All-Gather and All-Reduce Benchmarks ==="
echo "Buffer sizes: ${SIZES[@]}"
echo "Iterations per test: $ITERS"
echo ""

# Run All-Gather benchmark
echo "Running All-Gather benchmark..."
echo "=== NCCL All-Gather Buffer Size Sweep ===" > "$ALLGATHER_OUTPUT"
echo "Start time: $(date)" >> "$ALLGATHER_OUTPUT"
echo "" >> "$ALLGATHER_OUTPUT"

for size in "${SIZES[@]}"; do
    echo "  Testing All-Gather size: $size elements..."
    echo "" >> "$ALLGATHER_OUTPUT"
    echo "Buffer size: $size elements ($(echo "scale=2; $size * 4 / 1024 / 1024" | bc) MB per rank)" >> "$ALLGATHER_OUTPUT"
    echo "----------------------------------------" >> "$ALLGATHER_OUTPUT"
    ./nccl_allgather_inplace "$size" "$ITERS" 2>&1 | tee -a "$ALLGATHER_OUTPUT"
    echo "" >> "$ALLGATHER_OUTPUT"
done

echo "End time: $(date)" >> "$ALLGATHER_OUTPUT"
echo "=== All-Gather Benchmark Complete ===" >> "$ALLGATHER_OUTPUT"

echo ""
echo "Running All-Reduce benchmark..."

# Run All-Reduce benchmark
echo "=== NCCL All-Reduce Buffer Size Sweep ===" > "$ALLREDUCE_OUTPUT"
echo "Start time: $(date)" >> "$ALLREDUCE_OUTPUT"
echo "" >> "$ALLREDUCE_OUTPUT"

for size in "${SIZES[@]}"; do
    echo "  Testing All-Reduce size: $size elements..."
    echo "" >> "$ALLREDUCE_OUTPUT"
    echo "Buffer size: $size elements ($(echo "scale=2; $size * 4 / 1024 / 1024" | bc) MB per rank)" >> "$ALLREDUCE_OUTPUT"
    echo "----------------------------------------" >> "$ALLREDUCE_OUTPUT"
    ./nccl_allreduce "$size" "$ITERS" 2>&1 | tee -a "$ALLREDUCE_OUTPUT"
    echo "" >> "$ALLREDUCE_OUTPUT"
done

echo "End time: $(date)" >> "$ALLREDUCE_OUTPUT"
echo "=== All-Reduce Benchmark Complete ===" >> "$ALLREDUCE_OUTPUT"

echo ""
echo "=== Benchmarks Complete ==="
echo "All-Gather results: $ALLGATHER_OUTPUT"
echo "All-Reduce results: $ALLREDUCE_OUTPUT"
echo ""
echo "To parse and visualize results, run:"
echo "  python3 parse_results.py $ALLGATHER_OUTPUT $ALLREDUCE_OUTPUT"
