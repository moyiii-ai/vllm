#!/bin/bash
# Variation test script
# Runs 2 GPU-to-GPU DMA reads (GPU2 and GPU3 from GPU4) concurrently with GDR read test
# GPU IDs 0,1,2 in code are mapped to physical GPUs 2,3,4 via CUDA_VISIBLE_DEVICES

# Set CUDA_VISIBLE_DEVICES to map logical GPUs 0,1,2 to physical GPUs 2,3,4
export CUDA_VISIBLE_DEVICES=2,3,4

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
    echo "Example: $0 1000"
    exit 1
fi

# Configuration
GPU_TO_GPU_BIN=${GPU_TO_GPU_BIN:-./gpu_to_gpu_dma}
SIZE=${SIZE:-32MB}
GDR_WRITE_SERVER=${GDR_WRITE_SERVER:-./gdr_write_server.sh}
GDR_WRITE_CLIENT=${GDR_WRITE_CLIENT:-./gdr_write_client.sh}
MSG_SIZE=${MSG_SIZE:-33554432}
SERVER_IP=${SERVER_IP:-10.1.1.2}

# Check if binaries exist
if [ ! -f "$GPU_TO_GPU_BIN" ]; then
    echo "Error: Binary not found: $GPU_TO_GPU_BIN"
    echo "Please compile it first using: make"
    exit 1
fi

if [ ! -f "$GDR_WRITE_SERVER" ]; then
    echo "Error: Script not found: $GDR_WRITE_SERVER"
    exit 1
fi

if [ ! -f "$GDR_WRITE_CLIENT" ]; then
    echo "Error: Script not found: $GDR_WRITE_CLIENT"
    exit 1
fi

echo "================================================================"
echo "Variation Test: GPU-to-GPU DMA Read + GDR Write"
echo "================================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Logical GPU 0,1,2 -> Physical GPU 2,3,4"
echo "================================================================"
echo "GPU-to-GPU DMA Read:"
echo "  1. GPU2 from GPU4 (logical: GPU0 from GPU2)"
echo "  2. GPU3 from GPU4 (logical: GPU1 from GPU2)"
echo "  Transfer size: $SIZE"
echo ""
echo "GDR Write Test:"
echo "  Server: h2 netns, NUMA0, CPU only, size=$MSG_SIZE bytes, $ITERATIONS iterations"
echo "  Client: h3 netns, NUMA1, GPU4 -> h2 server ($SERVER_IP)"
echo "================================================================"
echo "Press Ctrl+C to stop all operations"
echo ""

PIDS=()
GPU_TO_GPU_PIDS=()
CLEANED_UP=0

cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo ""
    echo "Stopping all processes..."
    
    # Stop GPU-to-GPU DMA reads
    for pid in "${GPU_TO_GPU_PIDS[@]}"; do
        if [ -n "$pid" ]; then
            echo "Stopping GPU-to-GPU DMA (PID: $pid)..."
            kill -SIGINT $pid 2>/dev/null || true
            wait $pid 2>/dev/null || true
        fi
    done

    # Stop GDR write client
    if [ -n "$GDR_WRITE_CLIENT_PID" ]; then
        echo "Stopping GDR write client (PID: $GDR_WRITE_CLIENT_PID)..."
        kill $GDR_WRITE_CLIENT_PID 2>/dev/null || true
        wait $GDR_WRITE_CLIENT_PID 2>/dev/null || true
    fi

    # Stop GDR write server
    if [ -n "$GDR_WRITE_SERVER_PID" ]; then
        echo "Stopping GDR write server (PID: $GDR_WRITE_SERVER_PID)..."
        kill $GDR_WRITE_SERVER_PID 2>/dev/null || true
        wait $GDR_WRITE_SERVER_PID 2>/dev/null || true
    fi

    echo ""
    echo "================================================================"
    echo "Final Statistics:"
    echo "================================================================"
    if [ -f "gpu0_to_gpu2_dma_read.log" ]; then
        echo "=== GPU0 to GPU2 DMA Read ==="
        cat gpu0_to_gpu2_dma_read.log
        echo ""
    fi
    if [ -f "gpu1_to_gpu2_dma_read.log" ]; then
        echo "=== GPU1 to GPU2 DMA Read ==="
        cat gpu1_to_gpu2_dma_read.log
        echo ""
    fi
    if [ -f "gdr_write_client.log" ]; then
        echo "=== GDR Write Client ==="
        cat gdr_write_client.log
        echo ""
    fi
    if [ -f "gdr_write_server.log" ]; then
        echo "=== GDR Write Server ==="
        cat gdr_write_server.log
        echo ""
    fi
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

# Step 1: Start GPU-to-GPU DMA reads
# Logical GPU 0 (physical GPU2) reads from logical GPU 2 (physical GPU4)
echo "Starting GPU0 from GPU2 DMA read (logical: GPU0 from GPU2)..."
$GPU_TO_GPU_BIN read 0 2 $SIZE > gpu0_to_gpu2_dma_read.log 2>&1 &
GPU_TO_GPU_PIDS[0]=$!
PIDS+=(${GPU_TO_GPU_PIDS[0]})
echo "GPU0->GPU2 DMA read started (PID: ${GPU_TO_GPU_PIDS[0]})"

# Logical GPU 1 (physical GPU3) reads from logical GPU 2 (physical GPU4)
echo "Starting GPU1 from GPU2 DMA read (logical: GPU1 from GPU2)..."
$GPU_TO_GPU_BIN read 1 2 $SIZE > gpu1_to_gpu2_dma_read.log 2>&1 &
GPU_TO_GPU_PIDS[1]=$!
PIDS+=(${GPU_TO_GPU_PIDS[1]})
echo "GPU1->GPU2 DMA read started (PID: ${GPU_TO_GPU_PIDS[1]})"

# Step 2: Start GDR write server in h2 netns (NUMA0, CPU only)
echo "Starting GDR write server in h2 netns (NUMA0, CPU only)..."
bash $GDR_WRITE_SERVER $MSG_SIZE $ITERATIONS > gdr_write_server.log 2>&1 &
GDR_WRITE_SERVER_PID=$!
PIDS+=($GDR_WRITE_SERVER_PID)
echo "GDR write server started (PID: $GDR_WRITE_SERVER_PID)"

# Give server a moment to start
sleep 1

# Step 3: Start GDR write client in h3 netns (NUMA1, GPU4)
echo "Starting GDR write client in h3 netns (NUMA1, GPU4, connecting to $SERVER_IP)..."
bash $GDR_WRITE_CLIENT $SERVER_IP $MSG_SIZE $ITERATIONS > gdr_write_client.log 2>&1 &
GDR_WRITE_CLIENT_PID=$!
PIDS+=($GDR_WRITE_CLIENT_PID)
echo "GDR write client started (PID: $GDR_WRITE_CLIENT_PID)"

echo ""
echo "All processes started. PIDs: ${PIDS[@]}"
echo ""

# Wait for GDR write client to finish (ib_write_bw test will complete after $ITERATIONS iterations)
echo "Waiting for GDR write test to complete ($ITERATIONS iterations)..."
wait $GDR_WRITE_CLIENT_PID
echo "GDR write test finished."

# Step 4: Stop GPU-to-GPU DMA reads after ib_write_bw test ends
echo "Stopping GPU-to-GPU DMA reads..."
# First, send SIGINT to all processes
for pid in "${GPU_TO_GPU_PIDS[@]}"; do
    if [ -n "$pid" ]; then
        kill -SIGINT $pid 2>/dev/null || true
    fi
done
# Then wait for all processes to finish
for pid in "${GPU_TO_GPU_PIDS[@]}"; do
    if [ -n "$pid" ]; then
        wait $pid 2>/dev/null || true
    fi
done
echo "GPU-to-GPU DMA reads stopped."

# Stop GDR write server
echo "Stopping GDR write server..."
kill $GDR_WRITE_SERVER_PID 2>/dev/null || true
wait $GDR_WRITE_SERVER_PID 2>/dev/null || true
echo "GDR write server stopped."

# Cleanup will be called by trap to show final statistics
exit 0
