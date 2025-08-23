#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.1-8B"
HOST="0.0.0.0"
PORT=8000

vllm serve $MODEL_NAME \
  --host $HOST \
  --port $PORT > server.log 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > server.pid

echo "Waiting for vLLM server to be ready..."
for i in {1..120}; do
  if curl -s http://127.0.0.1:$PORT/v1/completions > /dev/null; then
    echo "vLLM server is up after $((i * 3)) seconds."
    break
  fi
  sleep 3
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
  echo "vLLM server failed to start."
  exit 1
fi

cleanup() {
  echo "Stopping vLLM server (PID: $SERVER_PID)..."
  kill $SERVER_PID 2>/dev/null || true
  rm -f server.pid
}
trap cleanup EXIT

python3 ../benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://127.0.0.1:$PORT \
  --endpoint /v1/completions \
  --model $MODEL_NAME \
  --tokenizer $MODEL_NAME \
  --served-model-name $MODEL_NAME \
  --dataset-name random \
  --num-prompts 200 > benchmark.log 2>&1

echo "Benchmark completed successfully"
