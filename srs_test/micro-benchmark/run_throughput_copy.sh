#!/bin/bash

# Define source file and executable names
SOURCE_FILE="throughput_copy.cu"
EXECUTABLE="throughput_copy"

# Check if exactly 2 arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <1|2> <read|write>"
    exit 1
fi

MODE_NUM="$1"   # 1 for one-way, 2 for two-way
MODE_OP="$2"    # read or write

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

# Run the executable with the two arguments
echo "Running $EXECUTABLE with mode $MODE_NUM and operation $MODE_OP..."
./"$EXECUTABLE" "$MODE_NUM" "$MODE_OP"

if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "Execution completed successfully."
exit 0
