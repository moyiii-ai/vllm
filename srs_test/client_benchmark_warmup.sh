MODEL="meta-llama/Llama-3.1-8B"
DATASET_NAME="longbench"
DATASET_PATH="narrativeqa.jsonl"
BENCHMARK_SCRIPT="../benchmarks/benchmark_serving_xingyu.py"

echo "Starting warmup benchmark."
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
    echo "Warmup benchmark failed."
    exit 1
fi

echo "Warmup benchmark completed. "
