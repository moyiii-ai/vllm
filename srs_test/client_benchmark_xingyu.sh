#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

echo "Starting first benchmark test..."
# Run first benchmark command with warmup and output length 1
python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --request-rate 2 \
    --warmup \
    --output-len 1

# Check if first command executed successfully
if [ $? -ne 0 ]; then
    echo "First benchmark failed. Terminating script."
    exit 1
fi

echo "First benchmark completed. Starting second benchmark test..."
# Run second benchmark command with more prompts and longer output
python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --request-rate 2 \
    --num-prompts 200 \
    --output-len 32 \
    --ignore-eos \
    --save-result

# Check if second command executed successfully
if [ $? -ne 0 ]; then
    echo "Second benchmark failed."
    exit 1
fi

echo "All benchmark tests completed successfully."
