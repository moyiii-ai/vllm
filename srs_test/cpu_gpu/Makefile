CUDA_PATH ?= /usr/local/cuda-13
NVCC = $(CUDA_PATH)/bin/nvcc
CC = gcc
CXX = g++

NVCC_FLAGS = -std=c++14 -O3 -arch=sm_120 -lineinfo
CXX_FLAGS = -std=c++14 -O3
CC_FLAGS = -O2 -g -std=c11 -D_DEFAULT_SOURCE
RDMA_LIBS = -libverbs -lrdmacm -lpthread
CUDA_LIBS = -lcudart

# Targets
TARGETS = gpu_to_gpu_dma cpu_to_gpu_dma cpu_to_gpu_mmio rdma_client rdma_server

.PHONY: all clean

all: $(TARGETS)

gpu_to_gpu_dma: gpu_to_gpu_dma.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cpu_to_gpu_dma: cpu_to_gpu_dma.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cpu_to_gpu_mmio: cpu_to_gpu_mmio.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# RDMA client: standard C program
rdma_client: rdma_client.c rdma_common.c
	$(CC) $(CC_FLAGS) -o $@ $^ $(RDMA_LIBS)

# RDMA server: CUDA program (use nvcc to compile C code with CUDA headers)
rdma_server: rdma_server.c rdma_common.c
	$(CC) $(CC_FLAGS) -c rdma_common.c -o rdma_common.o
	$(NVCC) -x cu -O2 -g -std=c++11 -arch=sm_89 -c rdma_server.c -o rdma_server.o -I$(CUDA_PATH)/include
	$(NVCC) -O2 -g -arch=sm_89 -o $@ rdma_common.o rdma_server.o $(RDMA_LIBS) $(CUDA_LIBS) -L$(CUDA_PATH)/lib64
	rm -f rdma_common.o rdma_server.o

clean:
	rm -f $(TARGETS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
