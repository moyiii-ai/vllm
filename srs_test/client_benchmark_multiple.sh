#!/bin/bash

# Configuration parameters
MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"
START_RATE=0.2       # Starting request rate
END_RATE=2.0         # Ending request rate
NUM_RUNS=10          # Number of runs (from start to end)

# Calculate step size between request rates
STEP=$(echo "scale=4; ($END_RATE - $START_RATE) / ($NUM_RUNS - 1)" | bc)

echo "Starting $NUM_RUNS benchmark runs with request rates from $START_RATE to $END_RATE..."
echo "Step size: $STEP"

# Run benchmarks with different request rates
for ((i=0; i<NUM_RUNS; i++)); do
    # Calculate current request rate
    CURRENT_RATE=$(echo "scale=4; $START_RATE + ($i * $STEP)" | bc)
    
    # Round to 1 decimal place for cleaner values (optional)
    CURRENT_RATE=$(printf "%.1f" $CURRENT_RATE)
    
    echo "============================================="
    echo "Starting benchmark run $((i+1))/$NUM_RUNS"
    echo "Request rate: $CURRENT_RATE req/s"
    echo "============================================="
    
    # Run the benchmark with current request rate
    python3 "$BENCHMARK_SCRIPT" \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --request-rate "$CURRENT_RATE" \
        --num-prompts 200 \
        --output-len 32 \
        --ignore-eos \
        --save-result
    
    # Check if benchmark failed
    if [ $? -ne 0 ]; then
        echo "ERROR: Benchmark failed at request rate $CURRENT_RATE"
        exit 1
    fi
    
    echo "Benchmark run $((i+1)) completed successfully"
    echo ""
done

echo "All $NUM_RUNS benchmark runs completed successfully!"
