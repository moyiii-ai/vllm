#!/bin/bash
# Concurrent GDR test script
# Runs CPU->GPU DMA write in loop, and two GDR server-client pairs

# Parse arguments
if [ $# -gt 0 ]; then
    ITERATIONS=$1
else
    ITERATIONS=${ITERATIONS:-20000}
fi

# Validate iterations
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "Error: Iterations must be a positive integer"
    echo "Usage: $0 [iterations]"
    echo "Example: $0 200000"
    exit 1
fi

# Configuration
GPU_ID=4
CPU_TO_GPU_BIN=${CPU_TO_GPU_BIN:-./cpu_to_gpu_dma}
SIZE=${SIZE:-10MB}
MSG_SIZE=${MSG_SIZE:-10485760}

# Check if binary exists
if [ ! -f "$CPU_TO_GPU_BIN" ]; then
    echo "Error: Binary not found: $CPU_TO_GPU_BIN"
    echo "Please compile it first using: make"
    exit 1
fi

echo "================================================================"
echo "Concurrent GDR Test"
echo "================================================================"
echo "CPU->GPU DMA: GPU $GPU_ID, size=$SIZE"
echo "GDR Server 0: h3 netns (10.1.1.3), IB device mlx5_5, size=10MB"
echo "GDR Server 1: h4 netns (10.1.1.4), IB device mlx5_3, size=10MB"
echo "GDR Client 0: h2 netns -> h3 server (10.1.1.3), size=10MB, $ITERATIONS iterations"
echo "GDR Client 1: h1 netns -> h4 server (10.1.1.4), size=10MB, $ITERATIONS iterations"
echo "================================================================"
echo "Press Ctrl+C to stop all operations"
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
    
    # Stop CPU->GPU DMA
    if [ -n "$CPU_TO_GPU_PID" ]; then
        echo "Stopping CPU->GPU DMA (PID: $CPU_TO_GPU_PID)..."
        kill -SIGINT $CPU_TO_GPU_PID 2>/dev/null || true
        wait $CPU_TO_GPU_PID 2>/dev/null || true
    fi

    # Stop GDR clients
    for pid in "${GDR_CLIENT_PIDS[@]}"; do
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null || true
        fi
    done

    # Wait for GDR clients to finish
    wait "${GDR_CLIENT_PIDS[@]}" 2>/dev/null

    # Stop GDR servers
    for pid in "${GDR_SERVER_PIDS[@]}"; do
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null || true
        fi
    done

    # Wait for GDR servers to finish
    wait "${GDR_SERVER_PIDS[@]}" 2>/dev/null

    echo ""
    echo "================================================================"
    echo "Final Statistics:"
    echo "================================================================"
    if [ -f "cpu_to_gpu_dma.log" ]; then
        cat cpu_to_gpu_dma.log
    fi
    echo ""
    if [ -f "gdr0_server_dma.log" ]; then
        echo "=== GDR Server 0 (h3) ==="
        cat gdr0_server_dma.log
        echo ""
    fi
    if [ -f "gdr1_server_dma.log" ]; then
        echo "=== GDR Server 1 (h4) ==="
        cat gdr1_server_dma.log
        echo ""
    fi
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

# # Step 0: Start CPU->GPU DMA write in background, output to log
# echo "Starting CPU->GPU DMA write (GPU $GPU_ID)..."
# $CPU_TO_GPU_BIN write $GPU_ID $SIZE > cpu_to_gpu_dma.log 2>&1 &
# CPU_TO_GPU_PID=$!
# PIDS+=($CPU_TO_GPU_PID)
# echo "CPU->GPU DMA started (PID: $CPU_TO_GPU_PID)"

# # Give CPU->GPU a moment to initialize
# sleep 1

# Step 1: Start GDR server 0 in h3 netns
# Note: Both server and client set --iters for consistency (SYMMETRIC parameter in ib_write_bw)
# In practice, client's setting usually takes precedence, but setting both ensures alignment
echo "Starting GDR server 0 in h3 netns..."
sudo ip netns exec h3 bash -c "
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    numactl --membind=1 --cpunodebind=1 ib_write_bw \
        --ib-dev=mlx5_5 \
        --ib-port=1 \
        --gid-index=0 \
        --size=$MSG_SIZE \
        --iters=$ITERATIONS \
        --use_cuda=0 \
        > gdr0_server_dma.log 2>&1
" &
GDR_SERVER_PIDS[0]=$!
PIDS+=(${GDR_SERVER_PIDS[0]})
echo "GDR server 0 started (PID: ${GDR_SERVER_PIDS[0]})"

# # Step 2: Start GDR server 1 in h4 netns
# echo "Starting GDR server 1 in h4 netns..."
# sudo ip netns exec h4 bash -c "
#     export CUDA_VISIBLE_DEVICES=$GPU_ID
#     numactl --membind=1 --cpunodebind=1 ib_write_bw \
#         --ib-dev=mlx5_3 \
#         --ib-port=1 \
#         --gid-index=0 \
#         --size=$MSG_SIZE \
#         --iters=$ITERATIONS \
#         --use_cuda=0 \
#         > gdr1_server_dma.log 2>&1
# " &
# GDR_SERVER_PIDS[1]=$!
# PIDS+=(${GDR_SERVER_PIDS[1]})
# echo "GDR server 1 started (PID: ${GDR_SERVER_PIDS[1]})"

# Give servers a moment to start
sleep 2

# Step 3: Start GDR client 0 in h2 netns, connecting to h3 (10.1.1.3)
echo "Starting GDR client 0 in h2 netns (connecting to 10.1.1.3)..."
sudo ip netns exec h2 ib_write_bw \
    --ib-dev=mlx5_0 \
    --ib-port=1 \
    --gid-index=0 \
    --size=$MSG_SIZE \
    --iters=$ITERATIONS \
    10.1.1.3 > gdr0_client_dma.log 2>&1 &
GDR_CLIENT_PIDS[0]=$!
PIDS+=(${GDR_CLIENT_PIDS[0]})
echo "GDR client 0 started (PID: ${GDR_CLIENT_PIDS[0]})"

# # Step 4: Start GDR client 1 in h1 netns, connecting to h4 (10.1.1.4)
# echo "Starting GDR client 1 in h1 netns (connecting to 10.1.1.4)..."
# sudo ip netns exec h1 ib_write_bw \
#     --ib-dev=mlx5_1 \
#     --ib-port=1 \
#     --gid-index=0 \
#     --size=$MSG_SIZE \
#     --iters=$ITERATIONS \
#     10.1.1.4 > gdr1_client_dma.log 2>&1 &
# GDR_CLIENT_PIDS[1]=$!
# PIDS+=(${GDR_CLIENT_PIDS[1]})
# echo "GDR client 1 started (PID: ${GDR_CLIENT_PIDS[1]})"

echo ""
echo "All processes started. PIDs: ${PIDS[@]}"
echo ""

# Wait for GDR clients to finish (they will complete after $ITERATIONS iterations)
echo "Waiting for GDR clients to complete ($ITERATIONS iterations)..."
wait "${GDR_CLIENT_PIDS[@]}"
echo "GDR clients finished."

# Step 5: Stop CPU->GPU DMA after GDR clients finish
# echo "Stopping CPU->GPU DMA..."
# kill -SIGINT $CPU_TO_GPU_PID 2>/dev/null || true
# wait $CPU_TO_GPU_PID 2>/dev/null || true
# echo "CPU->GPU DMA stopped."

# # Stop GDR servers
# echo "Stopping GDR servers..."
# for pid in "${GDR_SERVER_PIDS[@]}"; do
#     kill $pid 2>/dev/null || true
# done
# wait "${GDR_SERVER_PIDS[@]}" 2>/dev/null || true
# echo "GDR servers stopped."

# # Cleanup will be called by trap to show final statistics
# exit 0
