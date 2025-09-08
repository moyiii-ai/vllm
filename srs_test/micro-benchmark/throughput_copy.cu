#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>

#define DATA_SIZE (8ULL * 1024 * 1024 * 1024) // 8GB
#define BLOCK_SIZE 256
#define REPEAT 50

// Simple copy kernel
__global__ void copyKernel(int* destination, const int* source, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        destination[idx] = source[idx];
    }
}

// Check CUDA errors
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <1|2> <read|write>" << std::endl;
        return 1;
    }

    int mode = std::stoi(argv[1]);        // 1: one-way, 2: two-way
    std::string op = argv[2];             // "read" or "write"

    size_t numElements = DATA_SIZE / sizeof(int);
    int gridSize = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Device pointers
    int *d_src0, *d_src1, *d_dst0, *d_dst1 = nullptr;

    // Allocate and initialize GPU0 memory
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst0, DATA_SIZE));
    checkCuda(cudaMemset(d_src0, 1, DATA_SIZE)); // initialize source on GPU0
    checkCuda(cudaMemset(d_dst0, 0, DATA_SIZE)); // initialize destination on GPU0

    // Allocate and initialize GPU1 memory
    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst1, DATA_SIZE));
    checkCuda(cudaMemset(d_src1, 1, DATA_SIZE)); // initialize source on GPU1
    checkCuda(cudaMemset(d_dst1, 0, DATA_SIZE)); // initialize destination on GPU1

    // Enable P2P access
    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01) {
        checkCuda(cudaSetDevice(0));
        checkCuda(cudaDeviceEnablePeerAccess(1, 0));
    }
    if (canAccess10) {
        checkCuda(cudaSetDevice(1));
        checkCuda(cudaDeviceEnablePeerAccess(0, 0));
    }

    double total_time_sec = 0.0;

    for (int i = 0; i < REPEAT; i++) {
        if (mode == 1) {
            // Single device: use CUDA events for accurate GPU timing
            cudaEvent_t start, stop;
            checkCuda(cudaSetDevice(0));
            checkCuda(cudaEventCreate(&start));
            checkCuda(cudaEventCreate(&stop));
            checkCuda(cudaEventRecord(start));

            if (op == "read") {
                // GPU0 reads from GPU1
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, numElements);
            } else {
                // GPU0 writes to GPU1
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, numElements);
            }

            checkCuda(cudaEventRecord(stop));
            checkCuda(cudaEventSynchronize(stop));
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop));
            total_time_sec += ms / 1000.0;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        } else {
            // Two-way: use host clock for cross-GPU measurement
            auto t_start = std::chrono::high_resolution_clock::now();

            // Launch kernels on GPU0 and GPU1
            checkCuda(cudaSetDevice(0));
            if (op == "read")
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, numElements);
            else
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, numElements);

            checkCuda(cudaSetDevice(1));
            if (op == "read")
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, numElements);
            else
                copyKernel<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, numElements); // cross GPU write

            // Synchronize both devices
            checkCuda(cudaSetDevice(0)); cudaDeviceSynchronize();
            checkCuda(cudaSetDevice(1)); cudaDeviceSynchronize();

            auto t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = t_end - t_start;
            total_time_sec += diff.count();
        }
    }

    double avg_time_sec = total_time_sec / REPEAT;
    double throughput_GBps = (DATA_SIZE / 1.0e9) / avg_time_sec;

    std::cout << "Mode: " << mode << ", Operation: " << op
              << ", Avg Time: " << avg_time_sec << " s"
              << ", Throughput: " << throughput_GBps << " GB/s" << std::endl;

    // Cleanup
    checkCuda(cudaSetDevice(0)); cudaFree(d_src0); cudaFree(d_dst0);
    checkCuda(cudaSetDevice(1)); cudaFree(d_src1); cudaFree(d_dst1);

    return 0;
}
