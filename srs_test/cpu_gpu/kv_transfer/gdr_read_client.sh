#!/bin/bash
# GPU Direct RDMA read client script
# Binds to GPU4, runs in specified netns

# Configuration
GPU_ID=4
NETNS=${NETNS:-h3}
IB_DEVICE=${IB_DEVICE:-mlx5_5}
IB_PORT=${IB_PORT:-1}
GID_INDEX=${GID_INDEX:-0}
MSG_SIZE=${MSG_SIZE:-33554432}
ITERATIONS=${ITERATIONS:-55}
SERVER_IP=${SERVER_IP:-10.1.1.2}

# Parse arguments
if [ $# -gt 0 ]; then
    SERVER_IP=$1
fi
if [ $# -gt 1 ]; then
    MSG_SIZE=$2
fi
if [ $# -gt 2 ]; then
    ITERATIONS=$3
fi
if [ $# -gt 3 ]; then
    IB_DEVICE=$4
fi

echo "================================================================"
echo "GPU Direct RDMA Read Client"
echo "================================================================"
echo "GPU ID: $GPU_ID"
echo "Network Namespace: $NETNS"
echo "Server IP: $SERVER_IP"
echo "IB Device: $IB_DEVICE"
echo "IB Port: $IB_PORT"
echo "GID Index: $GID_INDEX"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

# Set CUDA device
# CUDA_VISIBLE_DEVICES sets logical GPU 0 to physical GPU $GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run ib_read_bw client in specified netns
echo "Starting ib_read_bw client in netns $NETNS (GPU $GPU_ID)..."
echo "Connecting to server at $SERVER_IP..."
echo ""

START_NS=$(date +%s%N)

sudo ip netns exec $NETNS ib_read_bw \
    --ib-dev=$IB_DEVICE \
    --ib-port=$IB_PORT \
    --gid-index=$GID_INDEX \
    --size=$MSG_SIZE \
    --iters=$ITERATIONS \
    $SERVER_IP

# Record end time
END_NS=$(date +%s%N)

# Compute elapsed time
ELAPSED_NS=$((END_NS - START_NS))
ELAPSED_SEC=$(awk "BEGIN {printf \"%.6f\", $ELAPSED_NS/1e9}")

echo ""
echo "Total ib_read_bw runtime: ${ELAPSED_SEC} seconds"

echo ""
echo "Client finished."
