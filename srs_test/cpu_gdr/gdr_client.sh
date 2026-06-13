#!/bin/bash
# GPU Direct RDMA client script
# Runs in h1 netns, writes from CPU to server GPU buffer

# Configuration
NETNS=h1
IB_DEVICE=${IB_DEVICE:-mlx5_1}
IB_PORT=${IB_PORT:-1}
GID_INDEX=${GID_INDEX:-0}
MSG_SIZE=${MSG_SIZE:-1048576}
ITERATIONS=${ITERATIONS:-1000}
SERVER_IP=${SERVER_IP:-10.1.1.4}

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
echo "GPU Direct RDMA Client"
echo "================================================================"
echo "Network Namespace: $NETNS"
echo "Server IP: $SERVER_IP"
echo "IB Device: $IB_DEVICE"
echo "IB Port: $IB_PORT"
echo "GID Index: $GID_INDEX"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

# Run ib_bw_write client in netns h1
echo "Starting ib_bw_write client in netns $NETNS..."
echo "Connecting to server at $SERVER_IP..."
echo ""

sudo ip netns exec $NETNS ib_write_bw \
    --ib-dev=$IB_DEVICE \
    --ib-port=$IB_PORT \
    --gid-index=$GID_INDEX \
    --size=$MSG_SIZE \
    --iters=$ITERATIONS \
    $SERVER_IP

echo ""
echo "Client finished."
