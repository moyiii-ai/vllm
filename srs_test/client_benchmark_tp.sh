#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

python3 "$BENCHMARK_SCRIPT" \
    --backend vllm \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --request-rate 1.5 \
    --num-prompts 50 \
    --output-len 32 \
    --ignore-eos \
    --save-result

# Check if command executed successfully
if [ $? -ne 0 ]; then
    echo "Benchmark failed."
    exit 1
fi

echo "Benchmark tests completed successfully."

echo -e "\n"
echo -e "\033[1;32m=======================================================\033[0m"
echo -e "\033[1;32m!!!                IMPORTANT NOTICE                   !!!\033[0m"
echo -e "\033[1;32m!!!  All Benchmark tests have been completed successfully!  !!!\033[0m"
echo -e "\033[1;32m=======================================================\033[0m"
echo -e "\n"
