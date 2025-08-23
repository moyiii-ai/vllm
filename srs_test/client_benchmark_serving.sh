#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.1-8B"
SERVER_URL="http://127.0.0.1:8000"  
ENDPOINT="/v1/completions"          
BENCHMARK_LOG="benchmark.log"       
BENCHMARK_SCRIPT_PATH="../benchmarks/benchmark_serving.py"

if [ ! -f "$BENCHMARK_SCRIPT_PATH" ]; then
  echo "ERROR: Benchmark script not found at $BENCHMARK_SCRIPT_PATH"
  exit 1
fi

echo "Checking if vLLM server is reachable at $SERVER_URL..."
if ! curl -s "$SERVER_URL$ENDPOINT" > /dev/null; then
  echo "ERROR: vLLM server is not running or unreachable at $SERVER_URL"
  exit 1
fi

echo "Starting vLLM benchmark (logs in: $BENCHMARK_LOG)..."
python3 $BENCHMARK_SCRIPT_PATH \
  --backend vllm \
  --base-url $SERVER_URL \
  --endpoint $ENDPOINT \
  --model $MODEL_NAME \
  --tokenizer $MODEL_NAME \
  --served-model-name $MODEL_NAME \
  --dataset-name random \
  --num-prompts 200 > $BENCHMARK_LOG 2>&1

if [ $? -eq 0 ]; then
  echo "Benchmark completed successfully! Results in: $BENCHMARK_LOG"
else
  echo "ERROR: Benchmark failed (check logs: $BENCHMARK_LOG)"
  exit 1
fi