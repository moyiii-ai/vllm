CUDA_PATH ?= /usr/local/cuda-13
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

NVCC_FLAGS = -std=c++14 -O3 -arch=sm_120 -lineinfo
CXX_FLAGS = -std=c++14 -O3

# Targets
TARGETS = cpu_to_gpu_dma

.PHONY: all clean

all: $(TARGETS)

cpu_to_gpu_dma: cpu_to_gpu_dma.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
