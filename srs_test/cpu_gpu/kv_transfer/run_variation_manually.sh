#!/bin/bash
# Manual Variation Test Script
# Orchestrates 3 concurrent operations in network namespaces:
# 1. RDMA Server (CPU-only mode, h2 netns, NUMA0)
# 2. RDMA Client (GPU4, write mode, 32MB, h3 netns, NUMA1, waits for signal)
# 3. GPU-to-GPU variation transfers (main netns)
# 
# The RDMA Client signals this script after RDMA handshake is complete,
# then waits for this script to signal it to start RDMA writes.
# All processes auto-exit based on iteration count.

set -e  # Exit on error

# Set CUDA_VISIBLE_DEVICES only for GPU-to-GPU variation (not RDMA client)
# Logical GPU 0,1,2 -> Physical GPU 2,3,4
VARIATION_CUDA_VISIBLE_DEVICES=${VARIATION_CUDA_VISIBLE_DEVICES:-2,3,4}

# Configuration
ITERATIONS=${ITERATIONS:-55}
DATA_SIZE="32MB"
GPU_ID=4
RDMA_SERVER_PORT=60000
SERVER_IP=${SERVER_IP:-10.1.1.2}

# Network namespaces
SERVER_NETNS=h2
CLIENT_NETNS=h3

# NUMA nodes
SERVER_NUMA=0
CLIENT_NUMA=1

# IB device configuration
IB_DEVICE=${IB_DEVICE:-mlx5_0}
IB_DEVICE_CLIENT=${IB_DEVICE_CLIENT:-mlx5_5}

# Log files
LOG_DIR="variation_results_manual"
VARIATION_LOG="$LOG_DIR/variation.log"
RDMA_CLIENT_LOG="$LOG_DIR/rdma_client.log"
RDMA_SERVER_LOG="$LOG_DIR/rdma_server.log"

# Check if required binaries exist
if [ ! -f "./variation" ]; then
    echo "Error: variation binary not found. Please compile it first using: make"
    exit 1
fi

if [ ! -f "./rdma_server" ]; then
    echo "Error: rdma_server not found. Please compile it first using: make"
    exit 1
fi

if [ ! -f "./rdma_client" ]; then
    echo "Error: rdma_client not found. Please compile it first using: make"
    exit 1
fi

# Parse arguments
if [ $# -gt 0 ]; then
    ITERATIONS=$1
fi

if [ $# -gt 1 ]; then
    SERVER_IP=$2
fi

if [ $# -gt 2 ]; then
    IB_DEVICE=$3
fi

if [ $# -gt 3 ]; then
    IB_DEVICE_CLIENT=$4
fi

# Validate iterations
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: Iterations must be a positive integer"
    echo "Usage: $0 [iterations] [server_ip] [ib_device_server] [ib_device_client]"
    echo "Example: $0 55 10.1.1.2 mlx5_0 mlx5_5"
    exit 1
fi

# Check if network namespaces exist
if ! ip netns list | grep -q "^$SERVER_NETNS"; then
    echo "Error: Network namespace '$SERVER_NETNS' not found"
    echo "Available namespaces:"
    ip netns list || echo "  (none)"
    exit 1
fi

if ! ip netns list | grep -q "^$CLIENT_NETNS"; then
    echo "Error: Network namespace '$CLIENT_NETNS' not found"
    echo "Available namespaces:"
    ip netns list || echo "  (none)"
    exit 1
fi

echo "================================================================"
echo "Manual Variation Test"
echo "================================================================"
echo "Configuration:"
echo "  RDMA Server: $SERVER_NETNS netns, NUMA$SERVER_NUMA, CPU-only mode, $SERVER_IP:$RDMA_SERVER_PORT"
echo "  RDMA Client: $CLIENT_NETNS netns, NUMA$CLIENT_NUMA, GPU$GPU_ID, write mode, 32MB, $SERVER_IP:$RDMA_SERVER_PORT"
echo "  GPU-to-GPU: variation.cu with $ITERATIONS iterations"
echo "  Data Transfer: 32MB per iteration"
echo "  IB Device (Server): $IB_DEVICE"
echo "  IB Device (Client): $IB_DEVICE_CLIENT"
echo ""
echo "Process orchestration:"
echo "  1. Start RDMA Server in $SERVER_NETNS netns (CPU-only)"
echo "  2. Start RDMA Client in $CLIENT_NETNS netns (GPU$GPU_ID)"
echo "     - client sends SIGUSR1 to this script when handshake is complete"
echo "  3. After ready signal, start variation and send SIGUSR1 to client to begin RDMA writes"
echo "  4. Wait for variation + client to finish iterations, then stop server"
echo "================================================================"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

CLEANED_UP=0

cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo ""
    echo "Cleanup: Stopping all processes..."
    
    # Kill variation first
    if [ -n "$VARIATION_PID" ]; then
        if kill -0 "$VARIATION_PID" 2>/dev/null; then
            echo "Stopping variation (PID: $VARIATION_PID)..."
            kill -SIGTERM "$VARIATION_PID" 2>/dev/null || true
            wait "$VARIATION_PID" 2>/dev/null || true
        fi
    fi

    # Kill RDMA Client
    if [ -n "$RDMA_CLIENT_PID" ]; then
        if kill -0 "$RDMA_CLIENT_PID" 2>/dev/null; then
            echo "Stopping RDMA Client (PID: $RDMA_CLIENT_PID)..."
            kill -SIGINT "$RDMA_CLIENT_PID" 2>/dev/null || true
            wait "$RDMA_CLIENT_PID" 2>/dev/null || true
        fi
    fi

    # Kill RDMA Server
    if [ -n "$RDMA_SERVER_PID" ]; then
        if kill -0 "$RDMA_SERVER_PID" 2>/dev/null; then
            echo "Stopping RDMA Server (PID: $RDMA_SERVER_PID)..."
            kill -SIGTERM "$RDMA_SERVER_PID" 2>/dev/null || true
            wait "$RDMA_SERVER_PID" 2>/dev/null || true
        fi
    fi

    echo ""
    echo "================================================================"
    echo "Test Complete - Results:"
    echo "================================================================"

    # Show results from each component
    if [ -f "$VARIATION_LOG" ]; then
        echo ""
        echo "=== GPU-to-GPU Variation Results ==="
        tail -20 "$VARIATION_LOG" || true
    fi

    if [ -f "$RDMA_CLIENT_LOG" ]; then
        echo ""
        echo "=== RDMA Client Transfer Results ==="
        tail -20 "$RDMA_CLIENT_LOG" || true
    fi

    if [ -f "$RDMA_SERVER_LOG" ]; then
        echo ""
        echo "=== RDMA Server Log ==="
        tail -10 "$RDMA_SERVER_LOG" || true
    fi

    echo ""
    echo "Log files saved in: $LOG_DIR/"
    echo "================================================================"
}

trap cleanup SIGINT SIGTERM EXIT

# Get NUMA binding command
get_numactl_cmd() {
    local numa_node=$1
    if command -v numactl &> /dev/null; then
        echo "numactl --membind=$numa_node --cpunodebind=$numa_node"
    else
        echo ""
    fi
}

NUMACTL_SERVER=$(get_numactl_cmd "$SERVER_NUMA")
NUMACTL_CLIENT=$(get_numactl_cmd "$CLIENT_NUMA")

# Arrays to store PIDs
PIDS=()
VARIATION_PID=""
RDMA_CLIENT_PID=""
RDMA_SERVER_PID=""

echo "[1/4] Starting RDMA Server (CPU-only mode in $SERVER_NETNS netns, NUMA$SERVER_NUMA)..."
sudo ip netns exec $SERVER_NETNS $NUMACTL_SERVER ./rdma_server -c > "$RDMA_SERVER_LOG" 2>&1 &
RDMA_SERVER_PID=$!
PIDS+=($RDMA_SERVER_PID)
echo "RDMA Server started (PID: $RDMA_SERVER_PID)"
sleep 1  # Give server time to start

echo "[2/4] Starting RDMA Client (in $CLIENT_NETNS netns, NUMA$CLIENT_NUMA, GPU$GPU_ID, write mode)..."
# RDMA client will signal this script (SIGUSR1) after handshake completes
RDMA_READY=0
trap 'RDMA_READY=1' SIGUSR1
sudo ip netns exec $CLIENT_NETNS $NUMACTL_CLIENT ./rdma_client -a "$SERVER_IP" -p "$RDMA_SERVER_PORT" -w -g "$GPU_ID" -i "$ITERATIONS" -l "$RDMA_CLIENT_LOG" -P "$$" -t 1 > "$RDMA_CLIENT_LOG.stdout" 2>&1 &
RDMA_CLIENT_PID=$!
PIDS+=($RDMA_CLIENT_PID)
echo "RDMA Client started (PID: $RDMA_CLIENT_PID)"

echo "Waiting for RDMA Client handshake completion signal (SIGUSR1)..."
READY_TIMEOUT_SEC=30
READY_TICKS=$((READY_TIMEOUT_SEC * 10))
for ((i=0; i<READY_TICKS; i++)); do
    if [ "$RDMA_READY" -eq 1 ]; then
        echo "✓ RDMA Client handshake complete (signal received)"
        break
    fi
    sleep 0.1
done
if [ "$RDMA_READY" -ne 1 ]; then
    echo "Error: did not receive RDMA Client ready signal within ${READY_TIMEOUT_SEC}s"
    echo "Client stdout log: $RDMA_CLIENT_LOG.stdout"
    exit 1
fi

echo "[3/4] Starting GPU-to-GPU variation transfers ($ITERATIONS iterations)..."
CUDA_VISIBLE_DEVICES="$VARIATION_CUDA_VISIBLE_DEVICES" ./variation "$ITERATIONS" > "$VARIATION_LOG" 2>&1 &
VARIATION_PID=$!
PIDS+=($VARIATION_PID)
echo "variation started (PID: $VARIATION_PID)"

echo "[3b/4] Sending SIGUSR1 signal to RDMA Client to start RDMA writes..."
kill -SIGUSR1 "$RDMA_CLIENT_PID" 2>/dev/null || true

echo ""
echo "All processes started. Waiting for completion..."
echo ""

# Wait for variation to complete (it will exit after ITERATIONS)
echo "Waiting for variation to complete ($ITERATIONS iterations)..."
wait "$VARIATION_PID"
echo "variation completed."

# Wait for RDMA Client to complete naturally (it will exit after ITERATIONS)
echo "Waiting for RDMA Client to complete ($ITERATIONS iterations)..."
wait "$RDMA_CLIENT_PID" 2>/dev/null || true
echo "RDMA Client completed."

# Stop RDMA Server (now that client finished)
if kill -0 "$RDMA_SERVER_PID" 2>/dev/null; then
    echo "Stopping RDMA Server..."
    kill -SIGTERM "$RDMA_SERVER_PID" 2>/dev/null || true
    wait "$RDMA_SERVER_PID" 2>/dev/null || true
fi

echo ""
echo "All processes stopped successfully."

# Cleanup will be called by trap
exit 0
