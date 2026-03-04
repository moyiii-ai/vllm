#!/bin/bash
# GPU Direct RDMA read server script
# Binds to specified netns, no GPU, CPU only

# Configuration
NUMA_NODE=${NUMA_NODE:-0}
NETNS=${NETNS:-h2}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
IB_PORT=${IB_PORT:-1}
GID_INDEX=${GID_INDEX:-0}
MSG_SIZE=${MSG_SIZE:-33554432}
ITERATIONS=${ITERATIONS:-55}

# Parse arguments
if [ $# -gt 0 ]; then
    MSG_SIZE=$1
fi
if [ $# -gt 1 ]; then
    ITERATIONS=$2
fi
if [ $# -gt 2 ]; then
    IB_DEVICE=$3
fi
if [ $# -gt 3 ]; then
    NETNS=$4
fi

echo "================================================================"
echo "GPU Direct RDMA Read Server (CPU only)"
echo "================================================================"
echo "NUMA Node: $NUMA_NODE"
echo "Network Namespace: $NETNS"
echo "IB Device: $IB_DEVICE"
echo "IB Port: $IB_PORT"
echo "GID Index: $GID_INDEX"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

# Bind to NUMA node
if command -v numactl &> /dev/null; then
    NUMACTL_CMD="numactl --membind=$NUMA_NODE --cpunodebind=$NUMA_NODE"
    echo "Using numactl to bind to NUMA node $NUMA_NODE"
else
    NUMACTL_CMD=""
    echo "Warning: numactl not found, skipping NUMA binding"
fi

# Run ib_read_bw server in specified netns, CPU only (no GPU)
echo "Starting ib_read_bw server in netns $NETNS (CPU only)..."
echo "Waiting for client connection..."
echo ""

sudo ip netns exec $NETNS $NUMACTL_CMD ib_read_bw \
    --ib-dev=$IB_DEVICE \
    --ib-port=$IB_PORT \
    --gid-index=$GID_INDEX \
    --size=$MSG_SIZE \
    --iters=$ITERATIONS \
    --use_cuda=0

echo ""
echo "Server finished."
