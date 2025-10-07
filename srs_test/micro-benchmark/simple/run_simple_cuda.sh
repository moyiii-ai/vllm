#!/bin/bash

# Define the source file and executable names
SOURCE_FILE="simple_cuda.cu"
EXECUTABLE="simple_cuda"

# Check if mode argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [read|write]"
    exit 1
fi

MODE="$1"

# Validate mode argument
if [ "$MODE" != "read" ] && [ "$MODE" != "write" ]; then
    echo "Error: Invalid mode. Use 'read' or 'write'"
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

# Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# Verify executable was created
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not created."
    exit 1
fi

# Run with the specified mode
echo "Running in $MODE mode..."
./"$EXECUTABLE" "$MODE"

# Check if execution succeeded
if [ $? -ne 0 ]; then
    echo "$MODE mode execution failed."
    exit 1
fi

echo "$MODE mode completed."
exit 0
