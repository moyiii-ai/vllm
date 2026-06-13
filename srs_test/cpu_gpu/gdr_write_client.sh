#!/bin/bash
# RDMA write client script (ib_write_bw).
# Default: h3, numa1, GPU4. Override via env: NETNS, NUMA_NODE, USE_GPU (1=GPU, 0=CPU), GPU_ID.

# Configuration (env overrides supported)
NUMA_NODE=${NUMA_NODE:-1}
NETNS=${NETNS:-h3}
# USE_GPU: 1 or unset = bind to GPU (use GPU_ID), 0 = CPU only
USE_GPU=${USE_GPU:-1}
GPU_ID=${GPU_ID:-4}
IB_DEVICE=${IB_DEVICE:-mlx5_5}
IB_PORT=${IB_PORT:-1}
GID_INDEX=${GID_INDEX:-0}
# TCP port for control connection (must match server)
PORT=${PORT:-18515}
MSG_SIZE=${MSG_SIZE:-1048576}
ITERATIONS=${ITERATIONS:-1000}
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
echo "RDMA Write Client"
echo "================================================================"
echo "NUMA Node: $NUMA_NODE"
echo "Network Namespace: $NETNS"
echo "GPU: $([ "$USE_GPU" = "1" ] && echo "$GPU_ID" || echo "CPU only")"
echo "Server IP: $SERVER_IP"
echo "IB Device: $IB_DEVICE"
echo "IB Port: $IB_PORT"
echo "TCP Port: $PORT"
echo "GID Index: $GID_INDEX"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

if [[ "$USE_GPU" = "1" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Bind to NUMA node
if command -v numactl &> /dev/null; then
    NUMACTL_CMD="numactl --membind=$NUMA_NODE --cpunodebind=$NUMA_NODE"
    echo "Using numactl to bind to NUMA node $NUMA_NODE"
else
    NUMACTL_CMD=""
    echo "Warning: numactl not found, skipping NUMA binding"
fi

# Run ib_write_bw client in netns
echo "Starting ib_write_bw client in netns $NETNS..."
echo "Connecting to server at $SERVER_IP..."
echo ""

EXTRA_CUDA=""
[[ "$USE_GPU" = "1" ]] && EXTRA_CUDA="--use_cuda=$GPU_ID"

sudo ip netns exec $NETNS $NUMACTL_CMD ib_write_bw \
    --ib-dev=$IB_DEVICE \
    --ib-port=$IB_PORT \
    -p "$PORT" \
    --gid-index=$GID_INDEX \
    --size=$MSG_SIZE \
    --iters=$ITERATIONS \
    $EXTRA_CUDA \
    $SERVER_IP

echo ""
echo "Client finished."
