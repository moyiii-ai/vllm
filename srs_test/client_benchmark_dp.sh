#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

TOTAL_REQUEST_RATE=3.6
TOTAL_NUM_PROMPTS=200
NUM_PROCS=2 
BASE_PORT=8000

REQ_RATE=$(awk "BEGIN{printf \"%.2f\", $TOTAL_REQUEST_RATE / $NUM_PROCS}")
PROMPTS=$((TOTAL_NUM_PROMPTS / NUM_PROCS))

PIDS=()

for i in $(seq 0 $((NUM_PROCS-1))); do
    PORT=$((BASE_PORT + i))
    LOGFILE="benchmark_${PORT}.log"

    echo "Launching benchmark on port $PORT..."
    python3 "$BENCHMARK_SCRIPT" \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --request-rate "$REQ_RATE" \
        --num-prompts "$PROMPTS" \
        --output-len 32 \
        --ignore-eos \
        --save-result \
        --port "$PORT" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
done

for idx in "${!PIDS[@]}"; do
    wait "${PIDS[$idx]}"
    if [ $? -ne 0 ]; then
        echo "Benchmark on port $((BASE_PORT + idx)) failed."
        exit 1
    fi
done

echo "All benchmark tests completed successfully."
