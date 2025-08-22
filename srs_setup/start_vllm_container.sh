#!/bin/bash

CONTAINER_NAME="vllm-editable-manual"

IMAGE="vllm/vllm-openai:v0.10.1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
SSH_PATH="$HOME/.ssh"

SHM_SIZE="48g"
CPUSET_CPUS="16-31"
CPUSET_MEMS="1"

docker rm -f $CONTAINER_NAME 2>/dev/null || true

docker run -d --name $CONTAINER_NAME \
    --gpus '"device=2,3"' \
    --shm-size=$SHM_SIZE \
    --cpuset-cpus=$CPUSET_CPUS \
    --cpuset-mems=$CPUSET_MEMS \
    -v $VLLM_PATH:/vllm-workspace/vllm \
    -v $SSH_PATH:/root/.ssh \
    -w /vllm-workspace \
    --entrypoint sleep \
    $IMAGE infinity

echo "Container '$CONTAINER_NAME' has been started successfully!"
