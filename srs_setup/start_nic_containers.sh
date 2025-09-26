#!/bin/bash
set -e

CTN0_NAME="ctn0"
CTN1_NAME="ctn1"

CTN0_GPU="0"
CTN0_NIC="enp47s0np0"
CTN0_IP="10.1.1.1/24"

CTN1_NIC="enp193s0np0"
CTN1_IP="10.1.1.2/24"

IMAGE="vllm-with-nettools-rdma:v0.10.1"
WORKDIR="/vllm-workspace"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_PATH="$(dirname "$SCRIPT_DIR")"
SSH_PATH="$HOME/.ssh"

# Run containers with default bridge (eth0)
docker run -dit \
  --name $CTN0_NAME \
  --cpuset-mems=0 \
  --cpuset-cpus=0-15 \
  --gpus "device=$CTN0_GPU" \
  --cap-add=NET_ADMIN \
  -v $VLLM_PATH:/vllm-workspace/vllm \
  -v $SSH_PATH:/root/.ssh \
  -w $WORKDIR \
  --entrypoint sleep \
  $IMAGE infinity

docker run -dit \
  --name $CTN1_NAME \
  --cpuset-mems=1 \
  --cpuset-cpus=16-31 \
  --cap-add=NET_ADMIN \
  -w $WORKDIR \
  --entrypoint sleep \
  $IMAGE infinity

# Get container PIDs
CTN0_PID=$(docker inspect -f '{{.State.Pid}}' $CTN0_NAME)
CTN1_PID=$(docker inspect -f '{{.State.Pid}}' $CTN1_NAME)

# Move physical NICs into containers
sudo ip link set $CTN0_NIC netns $CTN0_PID
sudo nsenter -t $CTN0_PID -n ip link set $CTN0_NIC up
sudo nsenter -t $CTN0_PID -n ip addr add $CTN0_IP dev $CTN0_NIC

sudo ip link set $CTN1_NIC netns $CTN1_PID
sudo nsenter -t $CTN1_PID -n ip link set $CTN1_NIC up
sudo nsenter -t $CTN1_PID -n ip addr add $CTN1_IP dev $CTN1_NIC

# Test container-to-container connectivity via physical NIC
echo "Testing connectivity..."
sudo nsenter -t $CTN0_PID -n ping -c 3 ${CTN1_IP%/*}
sudo nsenter -t $CTN1_PID -n ping -c 3 ${CTN0_IP%/*}
sudo nsenter -t $CTN0_PID -n ip route
sudo nsenter -t $CTN1_PID -n ip route
sudo nsenter -t $CTN0_PID -n ping -I $CTN0_NIC -c 3 ${CTN1_IP%/*}
sudo nsenter -t $CTN1_PID -n ping -I $CTN1_NIC -c 3 ${CTN0_IP%/*}

echo "Setup complete."
