#!/bin/bash
set -e

CTN0_NAME="ctn0"
CTN1_NAME="ctn1"

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
  --cap-add=NET_ADMIN \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/rdma_cm \
  -w $WORKDIR \
  --entrypoint sleep \
  $IMAGE infinity

docker run -dit \
  --name $CTN1_NAME \
  --cpuset-mems=1 \
  --cpuset-cpus=16-31 \
  --gpus '"device=1,2"' \
  --cap-add=NET_ADMIN \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband/uverbs5 \
  --device=/dev/infiniband/rdma_cm \
  -v $VLLM_PATH:/vllm-workspace/vllm \
  -v $SSH_PATH:/root/.ssh \
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
# sudo nsenter -t $CTN0_PID -m mkdir -p /dev/infiniband/
# #sudo nsenter -t $CTN0_PID -m bash -c "touch /dev/infiniband/uverbs0 && chmod 666 /dev/infiniband/uverbs0"
# sudo nsenter -t $CTN0_PID -m mount --bind /dev/infiniband/uverbs0 /dev/infiniband/uverbs0

sudo ip link set $CTN1_NIC netns $CTN1_PID
sudo nsenter -t $CTN1_PID -n ip link set $CTN1_NIC up
sudo nsenter -t $CTN1_PID -n ip addr add $CTN1_IP dev $CTN1_NIC
# sudo nsenter -t $CTN1_PID -m mkdir -p /dev/infiniband/
# # sudo nsenter -t $CTN1_PID -m bash -c "touch /dev/infiniband/uverbs5 && chmod 666 /dev/infiniband/uverbs5"
# sudo nsenter -t $CTN1_PID -m mount --bind /dev/infiniband/uverbs5 /dev/infiniband/uverbs5

# Test container-to-container connectivity via physical NIC
echo "Testing connectivity..."
sudo nsenter -t $CTN0_PID -n ping -c 3 ${CTN1_IP%/*}
sudo nsenter -t $CTN1_PID -n ping -c 3 ${CTN0_IP%/*}
sudo nsenter -t $CTN0_PID -n ip route
sudo nsenter -t $CTN1_PID -n ip route
sudo nsenter -t $CTN0_PID -n ping -I $CTN0_NIC -c 3 ${CTN1_IP%/*}
sudo nsenter -t $CTN1_PID -n ping -I $CTN1_NIC -c 3 ${CTN0_IP%/*}

echo "Setup complete."
