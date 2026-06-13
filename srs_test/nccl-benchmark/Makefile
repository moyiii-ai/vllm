# Makefile for NCCL benchmarks (all-gather and all-reduce)
# Uses NCCL from conda environment

# Default conda environment path (can be overridden)
CONDA_ENV ?= /home/ruoshi/miniconda3/envs/nccl229

# NCCL paths
NCCL_INCLUDE = $(CONDA_ENV)/include
NCCL_LIB = $(CONDA_ENV)/lib

# CUDA paths (usually in system or conda)
# Try to auto-detect CUDA path, fallback to default
CUDA_PATH ?= $(shell dirname $$(dirname $$(which nvcc 2>/dev/null)) 2>/dev/null || echo /usr/local/cuda)
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Compiler and flags
NVCC = nvcc
# Use sm_89 for L40S (compute capability 8.9), or remove -arch to auto-detect
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++11
INCLUDES = -I$(NCCL_INCLUDE) -I$(CUDA_INCLUDE)
# Add rpath so runtime can find libraries
RPATH = -Xlinker -rpath=$(NCCL_LIB) -Xlinker -rpath=$(CUDA_LIB)
LIBS = -L$(NCCL_LIB) -L$(CUDA_LIB) -lnccl -lcudart -lpthread $(RPATH)

# Targets
ALLGATHER_TARGET = nccl_allgather_inplace
ALLGATHER_SOURCE = nccl_allgather_inplace.cu

ALLREDUCE_TARGET = nccl_allreduce
ALLREDUCE_SOURCE = nccl_allreduce.cu

.PHONY: all allgather allreduce clean run-allgather run-allreduce help

all: $(ALLGATHER_TARGET) $(ALLREDUCE_TARGET)

allgather: $(ALLGATHER_TARGET)

allreduce: $(ALLREDUCE_TARGET)

$(ALLGATHER_TARGET): $(ALLGATHER_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALLGATHER_SOURCE) -o $(ALLGATHER_TARGET) $(LIBS)

$(ALLREDUCE_TARGET): $(ALLREDUCE_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALLREDUCE_SOURCE) -o $(ALLREDUCE_TARGET) $(LIBS)

clean:
	rm -f $(ALLGATHER_TARGET) $(ALLREDUCE_TARGET)

run-allgather: $(ALLGATHER_TARGET)
	./$(ALLGATHER_TARGET)

run-allreduce: $(ALLREDUCE_TARGET)
	./$(ALLREDUCE_TARGET)

# Help target
help:
	@echo "Usage:"
	@echo "  make              - Build all benchmarks"
	@echo "  make allgather    - Build all-gather benchmark only"
	@echo "  make allreduce     - Build all-reduce benchmark only"
	@echo "  make clean        - Remove all built binaries"
	@echo "  make run-allgather - Build and run all-gather with default settings"
	@echo "  make run-allreduce - Build and run all-reduce with default settings"
	@echo ""
	@echo "Environment variables:"
	@echo "  CONDA_ENV         - Path to conda environment (default: $(CONDA_ENV))"
	@echo "  CUDA_PATH         - Path to CUDA installation (default: $(CUDA_PATH))"
	@echo ""
	@echo "Run with custom arguments:"
	@echo "  ./$(ALLGATHER_TARGET) <count_per_rank> <iters>"
	@echo "  ./$(ALLREDUCE_TARGET) <count_per_rank> <iters>"
	@echo "  Example: ./$(ALLGATHER_TARGET) 8388608 200"
