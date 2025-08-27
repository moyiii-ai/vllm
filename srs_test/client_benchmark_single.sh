#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --request-rate 3.6 \
    --num-prompts 200 \
    --output-len 32 \
    --ignore-eos \
    --save-result

# Check if command executed successfully
if [ $? -ne 0 ]; then
    echo "Benchmark failed."
    exit 1
fi

echo "Benchmark tests completed successfully."
