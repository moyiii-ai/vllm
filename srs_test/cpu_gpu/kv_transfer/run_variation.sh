#!/bin/bash
# Variation test script
# Runs GPU-to-GPU DMA transfers concurrently with GDR read test
# GPU IDs 0,1,2 in code are mapped to physical GPUs 2,3,4 via CUDA_VISIBLE_DEVICES

# Set CUDA_VISIBLE_DEVICES to map logical GPUs 0,1,2 to physical GPUs 2,3,4
export CUDA_VISIBLE_DEVICES=2,3,4

# Parse arguments
if [ $# -gt 0 ]; then
    ITERATIONS=$1
else
    ITERATIONS=${ITERATIONS:-100000}
fi

# Validate iterations
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: Iterations must be a positive integer"
    echo "Usage: $0 [iterations]"
    echo "Example: $0 55"
    exit 1
fi

# Configuration
VARIATION_BIN=${VARIATION_BIN:-./variation}
GDR_READ_SERVER=${GDR_READ_SERVER:-./gdr_read_server.sh}
GDR_READ_CLIENT=${GDR_READ_CLIENT:-./gdr_read_client.sh}
MSG_SIZE=${MSG_SIZE:-33554432}
SERVER_IP=${SERVER_IP:-10.1.1.2}
NETNS_SERVER=${NETNS_SERVER:-h2}
NETNS_CLIENT=${NETNS_CLIENT:-h3}
IB_DEVICE_SERVER=${IB_DEVICE_SERVER:-mlx5_0}
IB_DEVICE_CLIENT=${IB_DEVICE_CLIENT:-mlx5_5}

# Check if binaries exist
if [ ! -f "$VARIATION_BIN" ]; then
    echo "Error: Binary not found: $VARIATION_BIN"
    echo "Please compile it first using: make variation"
    exit 1
fi

if [ ! -f "$GDR_READ_SERVER" ]; then
    echo "Error: Script not found: $GDR_READ_SERVER"
    exit 1
fi

if [ ! -f "$GDR_READ_CLIENT" ]; then
    echo "Error: Script not found: $GDR_READ_CLIENT"
    exit 1
fi

echo "================================================================"
echo "Variation Test: GPU-to-GPU DMA + GDR Read"
echo "================================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Logical GPU 0,1,2 -> Physical GPU 2,3,4"
echo "================================================================"
echo "GPU-to-GPU DMA:"
echo "  1. GPU2 -> GPU1 (logical: GPU2 -> GPU1)"
echo "  2. GPU2 -> GPU0 (logical: GPU2 -> GPU0)"
echo "  Transfer size: 32 MB"
echo "  Iterations: $ITERATIONS"
echo ""
echo "GDR Read Test:"
echo "  Server: $NETNS_SERVER netns, CPU only, size=$MSG_SIZE bytes, $ITERATIONS iterations"
echo "  Client: $NETNS_CLIENT netns, GPU4, size=$MSG_SIZE bytes, $ITERATIONS iterations"
echo "  Server IP: $SERVER_IP"
echo "================================================================"
echo ""

PIDS=()

# Step 1: Start GDR read server in specified netns (CPU only)
echo "Starting GDR read server in $NETNS_SERVER netns (CPU only)..."
$GDR_READ_SERVER $MSG_SIZE $ITERATIONS $IB_DEVICE_SERVER $NETNS_SERVER > gdr_read_server.log 2>&1 &
GDR_READ_SERVER_PID=$!
PIDS+=($GDR_READ_SERVER_PID)
echo "GDR read server started (PID: $GDR_READ_SERVER_PID)"

# Give server a moment to start
sleep 0.6

# Step 2: Start GDR read client in specified netns (GPU4)
echo "Starting GDR read client in $NETNS_CLIENT netns (GPU4)..."
$GDR_READ_CLIENT $SERVER_IP $MSG_SIZE $ITERATIONS $IB_DEVICE_CLIENT > gdr_read_client.log 2>&1 &
GDR_READ_CLIENT_PID=$!
PIDS+=($GDR_READ_CLIENT_PID)
echo "GDR read client started (PID: $GDR_READ_CLIENT_PID)"

sleep 0.6

# Step 3: Start variation (GPU-to-GPU DMA)
echo "Starting variation (GPU-to-GPU DMA)..."
$VARIATION_BIN $ITERATIONS > variation.log 2>&1 &
VARIATION_PID=$!
PIDS+=($VARIATION_PID)
echo "Variation started (PID: $VARIATION_PID)"

echo ""
echo "All processes started. PIDs: ${PIDS[@]}"
echo "Waiting for all processes to complete ($ITERATIONS iterations)..."
echo ""

# Wait for all processes to finish
wait $VARIATION_PID
VARIATION_EXIT=$?
wait $GDR_READ_CLIENT_PID
CLIENT_EXIT=$?
wait $GDR_READ_SERVER_PID
SERVER_EXIT=$?

echo ""
echo "All processes finished."
echo ""

# Output results
echo "================================================================"
echo "Final Statistics:"
echo "================================================================"

if [ -f "variation.log" ]; then
    echo "=== GPU-to-GPU DMA (Variation) ==="
    grep -A 10 "=== Variation Results ===" variation.log || cat variation.log
    echo ""
fi

if [ -f "gdr_read_client.log" ]; then
    echo "=== GDR Read Client ==="
    cat gdr_read_client.log
    echo ""
fi

# if [ -f "gdr_read_server.log" ]; then
#     echo "=== GDR Read Server ==="
#     cat gdr_read_server.log
#     echo ""
# fi

echo "================================================================"
echo "Exit codes:"
echo "  Variation: $VARIATION_EXIT"
echo "  GDR Client: $CLIENT_EXIT"
echo "  GDR Server: $SERVER_EXIT"
echo "================================================================"

exit 0
