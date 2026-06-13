#!/bin/bash
# GDR RDMA read throughput test script
# Tests RDMA read throughput from h2 to h3 (GPU4)

# Configuration
GPU_ID=4
SERVER_NETNS=h2
CLIENT_NETNS=h3
SERVER_IP=10.1.1.2
BASE_PORT=60000
RDMA_PORT=$((BASE_PORT + GPU_ID))
LOG_FILE="rdma_read_throughput.log"

# Parse arguments
if [ $# -gt 0 ]; then
    GPU_ID=$1
    RDMA_PORT=$((BASE_PORT + GPU_ID))
fi
if [ $# -gt 1 ]; then
    SERVER_IP=$2
fi

# Check if binaries exist
if [ ! -f "./rdma_server" ]; then
    echo "Error: rdma_server not found. Please compile it first using: make"
    exit 1
fi

if [ ! -f "./rdma_client" ]; then
    echo "Error: rdma_client not found. Please compile it first using: make"
    exit 1
fi

echo "================================================================"
echo "GDR RDMA Read Throughput Test"
echo "================================================================"
echo "Server: GPU $GPU_ID, $SERVER_NETNS netns, port $RDMA_PORT"
echo "Client: $CLIENT_NETNS netns -> $SERVER_IP:$RDMA_PORT"
echo "Log file: $LOG_FILE"
echo "================================================================"
echo "Press Ctrl+C to stop the test"
echo ""

PIDS=()
CLEANED_UP=0

cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo ""
    echo "Stopping all processes..."
    
    # Stop client
    if [ -n "$CLIENT_PID" ]; then
        echo "Stopping RDMA client (PID: $CLIENT_PID)..."
        kill -SIGINT $CLIENT_PID 2>/dev/null || true
        wait $CLIENT_PID 2>/dev/null || true
    fi

    # Stop server
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping RDMA server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi

    echo ""
    echo "================================================================"
    echo "Test completed. Results saved to: $LOG_FILE"
    echo "================================================================"
    if [ -f "$LOG_FILE" ]; then
        echo "Last 20 lines of throughput log:"
        tail -20 "$LOG_FILE"
    fi
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

# Step 1: Start RDMA server in h3 netns with GPU4
echo "Starting RDMA server in $SERVER_NETNS netns (GPU $GPU_ID)..."
sudo ip netns exec $SERVER_NETNS ./rdma_server -d $GPU_ID > rdma_server.log 2>&1 &
SERVER_PID=$!
PIDS+=($SERVER_PID)
echo "RDMA server started (PID: $SERVER_PID)"

# Give server a moment to start
sleep 2

# Step 2: Start RDMA client in h2 netns
echo "Starting RDMA client in $CLIENT_NETNS netns (connecting to $SERVER_IP:$RDMA_PORT)..."
sudo ip netns exec $CLIENT_NETNS ./rdma_client -a $SERVER_IP -p $RDMA_PORT -w -l "$LOG_FILE" > rdma_client.log 2>&1 &
CLIENT_PID=$!
PIDS+=($CLIENT_PID)
echo "RDMA client started (PID: $CLIENT_PID)"

echo ""
echo "All processes started. PIDs: ${PIDS[@]}"
echo "Throughput data is being logged to: $LOG_FILE"
echo ""

# Wait for client to finish (it will run until Ctrl+C)
wait $CLIENT_PID

# Cleanup will be called by trap
exit 0
