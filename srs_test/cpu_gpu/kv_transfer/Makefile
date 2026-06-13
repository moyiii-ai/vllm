# CUDA configuration
CUDA_PATH ?= /usr/local/cuda-13
NVCC = $(CUDA_PATH)/bin/nvcc
CC = gcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_120 -lineinfo
CC_FLAGS   = -O2 -g -std=c11 -D_DEFAULT_SOURCE

# Libraries
RDMA_LIBS = -libverbs -lrdmacm -lpthread
CUDA_LIBS = -lcudart

# Targets
CUDA_BENCH_TARGETS = baseline baseline_2 variation
RDMA_TARGETS = rdma_client rdma_server
TARGETS = $(CUDA_BENCH_TARGETS) $(RDMA_TARGETS)

.PHONY: all clean

all: $(TARGETS)

# -----------------------
# CUDA benchmark targets
# -----------------------
baseline: baseline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

baseline_2: baseline_2.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

variation: variation.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# -----------------------
# RDMA targets
# -----------------------
rdma_client: rdma_client.c rdma_common.c
	$(CC) $(CC_FLAGS) -I$(CUDA_PATH)/include -o $@ $^ $(RDMA_LIBS) $(CUDA_LIBS) -L$(CUDA_PATH)/lib64

rdma_server: rdma_server.c rdma_common.c
	$(CC) $(CC_FLAGS) -c rdma_common.c -o rdma_common.o
	$(NVCC) -x cu -O2 -g -std=c++11 -arch=sm_89 \
		-c rdma_server.c -o rdma_server.o -I$(CUDA_PATH)/include
	$(NVCC) -O2 -g -arch=sm_89 -o $@ \
		rdma_common.o rdma_server.o \
		$(RDMA_LIBS) $(CUDA_LIBS) -L$(CUDA_PATH)/lib64
	rm -f rdma_common.o rdma_server.o

# -----------------------
# Generic rule
# -----------------------
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o
