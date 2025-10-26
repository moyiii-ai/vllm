#!/bin/bash
# Usage: ./run_loop.sh [-s <data_size>] <cuda|global|copy> <1|2> <read|write>

DATA_SIZE=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -s)
            DATA_SIZE=$2
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL[@]}"

if [ $# -ne 3 ]; then
    echo "Usage: $0 [-s <data_size>] <cuda|global|copy> <1|2> <read|write>"
    exit 1
fi

IMPL=$1
MODE=$2
OP=$3

if [ "$IMPL" = "cuda" ]; then
    SRC_FILE="loop_cuda.cu"
    BIN_FILE="gpu_peer_loop"
elif [ "$IMPL" = "global" ]; then
    SRC_FILE="loop_global.cu"
    BIN_FILE="gpu_peer_loop"
elif [ "$IMPL" = "copy" ]; then 
    SRC_FILE="loop_copy.cu"
    BIN_FILE="gpu_peer_loop"
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

PIDS=()

CLEANED_UP=0
cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    echo "Waiting for child processes to finish..."
    wait "${PIDS[@]}" 2>/dev/null

    cat gpu0.log
    echo 

    if [ "$MODE" -eq 2 ]; then
        cat gpu1.log
        echo
    fi
}

trap cleanup SIGINT SIGTERM SIGQUIT SIGHUP EXIT

RUN_CMD="./$BIN_FILE $OP"

if [ "$MODE" -eq 1 ]; then
    echo "Starting single process on GPU 0 ($OP mode)..."
    if [ -n "$DATA_SIZE" ]; then
        $RUN_CMD 0 "$DATA_SIZE" 2>gpu0.log &
    else
        $RUN_CMD 0 2>gpu0.log &
    fi
    PIDS+=($!)
elif [ "$MODE" -eq 2 ]; then
    echo "Starting synchronized dual processes ($OP mode)..."
    if [ -n "$DATA_SIZE" ]; then
        $RUN_CMD 0 sync "$DATA_SIZE" 2>gpu0.log &
        $RUN_CMD 1 sync "$DATA_SIZE" 2>gpu1.log &
    else
        $RUN_CMD 0 sync 2>gpu0.log &
        $RUN_CMD 1 sync 2>gpu1.log &
    fi
    PIDS+=($!)
    PIDS+=($!)
fi

wait "${PIDS[@]}"
