#!/bin/bash

# Configuration parameters
MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"

# DATASET_PART1="narrativeqa_part1.jsonl"
# DATASET_PART2="narrativeqa_part2.jsonl"
DATASET_PART1="narrativeqa.jsonl"
DATASET_PART2="narrativeqa.jsonl"

BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

TOTAL_REQUEST_RATE=4
TOTAL_NUM_PROMPTS=800
NUM_PROCS=2  # Number of parallel processes
BASE_PORT=8000

# Calculate request rate and prompts per process
# REQ_RATE=$(awk "BEGIN{printf \"%.2f\", $TOTAL_REQUEST_RATE / $NUM_PROCS}")
REQ_RATE=$TOTAL_REQUEST_RATE
PROMPTS=$((TOTAL_NUM_PROMPTS / NUM_PROCS))

# Array to store process IDs
PIDS=()

# Function to start a benchmark process
# Parameters:
#   $1 - Dataset path
#   $2 - Port number
start_benchmark() {
    local dataset_path=$1
    local port=$2
    local logfile="benchmark_${port}.log"
    
    echo "Launching benchmark on port $port for dataset: $dataset_path..."
    python3 "$BENCHMARK_SCRIPT" \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$dataset_path" \
        --request-rate "$REQ_RATE" \
        --num-prompts "$PROMPTS" \
        --output-len 32 \
        --ignore-eos \
        --save-result \
        --port "$port" \
        > "$logfile" 2>&1 &
    
    # Store the process ID
    PIDS+=($!)
    echo "Benchmark process for $dataset_path started with PID ${PIDS[-1]}"
}

# Start benchmark for each dataset part on separate ports
start_benchmark "$DATASET_PART1" $BASE_PORT
start_benchmark "$DATASET_PART2" $((BASE_PORT + 1))

echo "Waiting for all benchmark processes to finish..."
for idx in "${!PIDS[@]}"; do
    port=$((BASE_PORT + idx))
    wait "${PIDS[$idx]}"
    if [ $? -ne 0 ]; then
        echo "Error: Benchmark process on port $port failed"
        exit 1
    fi
    echo "Benchmark process on port $port completed successfully"
done

echo "All benchmark tests for both datasets completed successfully."

echo -e "\n"
echo -e "\033[1;32m=======================================================\033[0m"
echo -e "\033[1;32m!!!                IMPORTANT NOTICE                   !!!\033[0m"
echo -e "\033[1;32m!!!  All Benchmark tests have been completed successfully!  !!!\033[0m"
echo -e "\033[1;32m!!!  Log files: benchmark_8000.log / benchmark_8001.log  !!!\033[0m"
echo -e "\033[1;32m=======================================================\033[0m"
echo -e "\n"