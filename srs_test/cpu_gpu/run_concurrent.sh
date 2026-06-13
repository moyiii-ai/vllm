#!/bin/bash
# Shell script wrapper for concurrent execution
# This script runs GPU0->GPU2, GPU1->GPU2, and CPU<->GPU2 operations concurrently
# GPU IDs 0,1,2 in code are mapped to physical GPUs 2,3,4 via CUDA_VISIBLE_DEVICES

# Set CUDA_VISIBLE_DEVICES to map logical GPUs 0,1,2 to physical GPUs 2,3,4
export CUDA_VISIBLE_DEVICES=2,3,4

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dma|mmio> <read|write> [size] [gpu_to_gpu_bin] [cpu_to_gpu_dma_bin] [cpu_to_gpu_mmio_bin]"
    echo "Example: $0 dma write 1GB"
    echo "         $0 dma read 1GB"
    echo "         $0 mmio write 512MB"
    echo ""
    echo "Transfer modes:"
    echo "  dma  - DMA transfer (initiated from GPU via cudaMemcpy)"
    echo "  mmio - MMIO transfer (using mapped pinned memory)"
    echo ""
    echo "Direction:"
    echo "  write - CPU/GPU0/1 -> GPU2"
    echo "  read  - GPU2 -> CPU/GPU0/1"
    exit 1
fi

TRANSFER_MODE=$1
DIRECTION=$2

if [ "$TRANSFER_MODE" != "dma" ] && [ "$TRANSFER_MODE" != "mmio" ]; then
    echo "Error: transfer mode must be 'dma' or 'mmio'"
    exit 1
fi

if [ "$DIRECTION" != "read" ] && [ "$DIRECTION" != "write" ]; then
    echo "Error: direction must be 'read' or 'write'"
    exit 1
fi

SIZE=${3:-1GB}
GPU_TO_GPU_BIN=${4:-./gpu_to_gpu_dma}
CPU_TO_GPU_DMA_BIN=${5:-./cpu_to_gpu_dma}
CPU_TO_GPU_MMIO_BIN=${6:-./cpu_to_gpu_mmio}

# Check if binaries exist
if [ ! -f "$GPU_TO_GPU_BIN" ]; then
    echo "Error: Binary not found: $GPU_TO_GPU_BIN"
    echo "Please compile it first using: make"
    exit 1
fi

if [ "$TRANSFER_MODE" = "dma" ]; then
    if [ ! -f "$CPU_TO_GPU_DMA_BIN" ]; then
        echo "Error: Binary not found: $CPU_TO_GPU_DMA_BIN"
        echo "Please compile it first using: make"
        exit 1
    fi
    CPU_BIN="$CPU_TO_GPU_DMA_BIN"
    if [ "$DIRECTION" = "write" ]; then
        CPU_LOG="cpu_to_gpu2_dma_write.log"
        CPU_DESC="CPU->GPU2 DMA write"
    else
        CPU_LOG="cpu_to_gpu2_dma_read.log"
        CPU_DESC="GPU2->CPU DMA read"
    fi
else
    if [ ! -f "$CPU_TO_GPU_MMIO_BIN" ]; then
        echo "Error: Binary not found: $CPU_TO_GPU_MMIO_BIN"
        echo "Please compile it first using: make"
        exit 1
    fi
    CPU_BIN="$CPU_TO_GPU_MMIO_BIN"
    CPU_LOG="cpu_to_gpu2_mmio_write.log"
    CPU_DESC="CPU->GPU2 MMIO write"
    if [ "$DIRECTION" = "read" ]; then
        echo "Warning: MMIO mode only supports write direction, ignoring read mode"
        DIRECTION="write"
    fi
fi

echo "================================================================"
echo "Concurrent CPU-GPU and GPU-GPU Operations"
echo "================================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Logical GPU 0,1,2 -> Physical GPU 2,3,4"
echo "================================================================"
echo "Transfer mode: $TRANSFER_MODE"
echo "Direction: $DIRECTION"
echo "Transfer size: $SIZE"
echo "Running:"
if [ "$DIRECTION" = "write" ]; then
    echo "  1. $CPU_DESC (physical: CPU -> GPU4)"
    echo "  2. GPU0 -> GPU2 DMA $DIRECTION ($TRANSFER_MODE mode, physical: GPU2 -> GPU4)"
    echo "  3. GPU1 -> GPU2 DMA $DIRECTION ($TRANSFER_MODE mode, physical: GPU3 -> GPU4)"
else
    echo "  1. $CPU_DESC (physical: GPU4 -> CPU)"
    echo "  2. GPU2 -> GPU0 DMA $DIRECTION ($TRANSFER_MODE mode, physical: GPU4 -> GPU2)"
    echo "  3. GPU2 -> GPU1 DMA $DIRECTION ($TRANSFER_MODE mode, physical: GPU4 -> GPU3)"
fi
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
    echo "Waiting for child processes to finish..."
    wait "${PIDS[@]}" 2>/dev/null

    echo ""
    echo "================================================================"
    echo "Final Statistics:"
    echo "================================================================"
    cat "$CPU_LOG" 2>/dev/null
    echo ""
    cat "gpu0_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>/dev/null
    echo ""
    cat "gpu1_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>/dev/null
    echo ""
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

# Start CPU to GPU first (as requested)
echo "Starting $CPU_DESC..."
$CPU_BIN $DIRECTION 2 $SIZE > "$CPU_LOG" 2>&1 &
PIDS+=($!)

# Give CPU->GPU a moment to initialize
sleep 1

# Then start GPU peer transfers
if [ "$DIRECTION" = "write" ]; then
    echo "Starting GPU0->GPU2 DMA $DIRECTION ($TRANSFER_MODE mode)..."
    $GPU_TO_GPU_BIN $DIRECTION 0 2 $SIZE > "gpu0_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>&1 &
    PIDS+=($!)

    echo "Starting GPU1->GPU2 DMA $DIRECTION ($TRANSFER_MODE mode)..."
    $GPU_TO_GPU_BIN $DIRECTION 1 2 $SIZE > "gpu1_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>&1 &
    PIDS+=($!)
else
    echo "Starting GPU2->GPU0 DMA $DIRECTION ($TRANSFER_MODE mode)..."
    $GPU_TO_GPU_BIN $DIRECTION 0 2 $SIZE > "gpu0_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>&1 &
    PIDS+=($!)

    echo "Starting GPU2->GPU1 DMA $DIRECTION ($TRANSFER_MODE mode)..."
    $GPU_TO_GPU_BIN $DIRECTION 1 2 $SIZE > "gpu1_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" 2>&1 &
    PIDS+=($!)
fi

echo "All processes started. PIDs: ${PIDS[@]}"
echo ""

# Monitor and display output
tail -f "$CPU_LOG" "gpu0_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" "gpu1_to_gpu2_${TRANSFER_MODE}_${DIRECTION}.log" &
TAIL_PID=$!

# Wait for all processes
wait "${PIDS[@]}"

# Stop tail
kill $TAIL_PID 2>/dev/null || true

# Cleanup will be called by trap
exit 0
