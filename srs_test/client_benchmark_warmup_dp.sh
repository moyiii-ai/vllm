#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

DATASET_PART1="narrativeqa_part1.jsonl"
DATASET_PART2="narrativeqa_part2.jsonl"

PORT1=8000
PORT2=8001
REQ_RATE=2

echo "Starting warmup benchmark on port $PORT1..."
python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "longbench" \
    --dataset-path "$DATASET_PART1" \
    --request-rate "$REQ_RATE" \
    --warmup \
    --output-len 1 \
    --port "$PORT1" \
    > "warmup_${PORT1}.log" 2>&1 &

PID1=$!

echo "Starting warmup benchmark on port $PORT2..."
python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "longbench" \
    --dataset-path "$DATASET_PART2" \
    --request-rate "$REQ_RATE" \
    --warmup \
    --output-len 1 \
    --port "$PORT2" \
    > "warmup_${PORT2}.log" 2>&1 &

PID2=$!

wait $PID1
if [ $? -ne 0 ]; then
    echo "Warmup benchmark on port $PORT1 failed."
    exit 1
fi

wait $PID2
if [ $? -ne 0 ]; then
    echo "Warmup benchmark on port $PORT2 failed."
    exit 1
fi

echo "Both warmup benchmarks completed successfully."
