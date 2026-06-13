#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4

BINARY=${BINARY:-./baseline_2}

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    echo "Please build it first using: make"
    exit 1
fi

echo "Running $BINARY transfers..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Logical GPU 0,1,2 -> Physical GPU 2,3,4"
echo "================================================================"
$BINARY
echo "================================================================"
