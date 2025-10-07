#!/bin/bash
# Usage: ./run_loop.sh <1|2> <read|write>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <cuda|global|copy> <1|2> <read|write>"
    exit 1
fi

IMPL=$1      # Implementation: cuda or global or copy
MODE=$2      # 1 or 2
OP=$3        # read or write

if [ "$IMPL" = "cuda" ]; then
    SRC_FILE="loop_cuda.cu"
    BIN_FILE="gpu_peer_loop_cuda"
elif [ "$IMPL" = "global" ]; then
    SRC_FILE="loop_global.cu"
    BIN_FILE="gpu_peer_loop_global"
elif [ "$IMPL" = "copy" ]; then 
    SRC_FILE="loop_copy.cu"
    BIN_FILE="gpu_peer_loop_copy"
else
    echo "Invalid implementation: $IMPL. Must be cuda, global, or copy."
    exit 1
fi

echo "Compiling $SRC_FILE..."
nvcc -arch=sm_80 -O2 "$SRC_FILE" -o "$BIN_FILE"
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Array to store PIDs
PIDS=()

CLEANED_UP=0
cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo "Waiting for child processes to finish..."
    # Just wait — don't send SIGINT again
    wait "${PIDS[@]}" 2>/dev/null

    cat gpu0.log
    echo 

    if [ "$MODE" -eq 2 ]; then
        cat gpu1.log
        echo
    fi
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

if [ "$MODE" -eq 1 ]; then
    echo "Starting single process on GPU 0 ($OP mode)..."
    ./gpu_peer_loop "$OP" 0 2>gpu0.log &
    PIDS+=($!)
elif [ "$MODE" -eq 2 ]; then
    echo "Starting synchronized dual processes ($OP mode)..."
    ./gpu_peer_loop "$OP" 0 sync 2>gpu0.log &
    PIDS+=($!)
    ./gpu_peer_loop "$OP" 1 sync 2>gpu1.log &
    PIDS+=($!)
else
    echo "Invalid mode: $MODE. Must be 1 or 2."
    exit 1
fi

# Wait for all child processes
wait "${PIDS[@]}"