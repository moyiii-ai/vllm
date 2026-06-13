#!/bin/bash

# Configuration parameters
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

# Hardcoded list of request rates (adjust these values as needed)
# Example: covering a range from 0.5 to 15 with varying step sizes
REQUEST_RATES=(
    0.4   
    0.8
    1.2
    1.6
    2.0
    2.4
    2.8
    3.2
    3.6
    4.0
)

# Calculate total number of runs from the list length
NUM_RUNS=${#REQUEST_RATES[@]}

echo "Starting $NUM_RUNS benchmark runs with custom request rates..."
echo "Request rates to test: ${REQUEST_RATES[*]}"
echo ""

# Run benchmarks for each request rate in the list
for ((i=0; i<NUM_RUNS; i++)); do
    CURRENT_RATE=${REQUEST_RATES[$i]}
    RUN_NUM=$((i+1))
    
    echo "============================================="
    echo "Starting benchmark run $RUN_NUM/$NUM_RUNS"
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
    
    echo "Benchmark run $RUN_NUM completed successfully"
    echo ""
done

echo "All $NUM_RUNS benchmark runs completed successfully!"
