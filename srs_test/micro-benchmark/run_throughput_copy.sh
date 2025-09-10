#!/bin/bash

# Define source file and executable names
SOURCE_FILE="throughput_copy.cu"
EXECUTABLE="throughput_copy"

# Check if enough arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <1|2> <read|write> [0|1 for one-way base GPU]"
    exit 1
fi

MODE_NUM="$1"   # 1 for one-way, 2 for two-way
MODE_OP="$2"    # read or write
BASE_GPU="$3"   # Optional: 0 or 1, only used when MODE_NUM==1

# Validate first argument
if [ "$MODE_NUM" != "1" ] && [ "$MODE_NUM" != "2" ]; then
    echo "Error: First argument must be 1 (one-way) or 2 (two-way)"
    exit 1
fi

# Validate second argument
if [ "$MODE_OP" != "read" ] && [ "$MODE_OP" != "write" ]; then
    echo "Error: Second argument must be 'read' or 'write'"
    exit 1
fi

# Validate optional third argument for one-way mode
if [ "$MODE_NUM" == "1" ]; then
    if [ -z "$BASE_GPU" ]; then
        echo "Error: One-way mode requires a third argument: base GPU (0 or 1)"
        exit 1
    fi
    if [ "$BASE_GPU" != "0" ] && [ "$BASE_GPU" != "1" ]; then
        echo "Error: Base GPU must be 0 or 1"
        exit 1
    fi
fi

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found in current directory."
    exit 1
fi

# Compile the CUDA program
echo "Compiling $SOURCE_FILE..."
nvcc -arch=sm_80 -o "$EXECUTABLE" "$SOURCE_FILE"

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Verify executable was created
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not created."
    exit 1
fi

# Run the executable with the arguments
echo "Running $EXECUTABLE with mode $MODE_NUM and operation $MODE_OP..."
if [ "$MODE_NUM" == "1" ]; then
    ./"$EXECUTABLE" "$MODE_NUM" "$MODE_OP" "$BASE_GPU"
else
    ./"$EXECUTABLE" "$MODE_NUM" "$MODE_OP"
fi

if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "Execution completed successfully."
exit 0
