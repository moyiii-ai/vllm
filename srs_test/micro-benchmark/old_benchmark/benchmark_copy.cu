#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// Kernel function: Used for data transfer between GPUs
__global__ void copyKernel(int* destination, const int* source, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        destination[idx] = source[idx];
    }
}

// Check for CUDA errors
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper function: Enable Peer Access between two GPUs (if supported)
void enableGPUPeerAccess(int dev1, int dev2) {
    int can_access = 0;
    CHECK(cudaSetDevice(dev1));
    CHECK(cudaDeviceCanAccessPeer(&can_access, dev1, dev2));
    
    if (!can_access) {
        std::cerr << "Fatal Error: GPU" << dev1 << " cannot access GPU" << dev2 << " (no peer capability)" << std::endl;
        std::cerr << "Check GPU topology with 'nvidia-smi topo -m' to verify cross-GPU connectivity" << std::endl;
        exit(EXIT_FAILURE);
    }

    CHECK(cudaDeviceEnablePeerAccess(dev2, 0));
    CHECK(cudaSetDevice(dev2));
    CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    
    std::cout << "Successfully enabled bidirectional peer access between GPU" << dev1 << " and GPU" << dev2 << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [direction]" << std::endl;
    std::cout << "Direction parameters:" << std::endl;
    std::cout << "  write   - Transfer data from GPU0 to GPU1" << std::endl;
    std::cout << "  read    - Transfer data from GPU1 to GPU0" << std::endl;
    std::cout << "  both    - Perform write then read (default)" << std::endl;
}

// Write test function: GPU0 -> GPU1, create stream/events inside
void runWriteTest(int* d0_data, int* d1_data, int numElements, int gridSize, int blockSize, int iterations) {
    std::cout << "\nTest: On GPU0, GPU0 writes data to GPU1 using copyKernel..." << std::endl;
    
    // Create stream and events (inside function, bound to GPU0)
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Start timing and run iterations
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        copyKernel<<<gridSize, blockSize, 0, stream>>>(d1_data, d0_data, numElements);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    // Calculate and print metrics
    float elapsedMs;
    CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    double elapsedSec = elapsedMs / 1000.0;
    double dataSizeGB = (numElements * sizeof(int)) / (1024.0 * 1024 * 1024);
    double avgTimePerIter = elapsedSec / iterations;
    double bandwidth = dataSizeGB / avgTimePerIter;
    
    std::cout << "Average bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Cleanup function-specific stream/events
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

// Read test function: GPU1 -> GPU0, create stream/events inside
void runReadTest(int* d0_data, int* d1_data, int numElements, int gridSize, int blockSize, int iterations) {
    std::cout << "\nTest: On GPU0, GPU0 reads data from GPU1 using copyKernel..." << std::endl;
    
    // Create stream and events (inside function, bound to GPU1)
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Start timing and run iterations
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        copyKernel<<<gridSize, blockSize, 0, stream>>>(d0_data, d1_data, numElements);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    // Calculate and print metrics
    float elapsedMs;
    CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    double elapsedSec = elapsedMs / 1000.0;
    double dataSizeGB = (numElements * sizeof(int)) / (1024.0 * 1024 * 1024);
    double avgTimePerIter = elapsedSec / iterations;
    double bandwidth = dataSizeGB / avgTimePerIter;
    
    std::cout << "Average bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Cleanup function-specific stream/events
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main(int argc, char* argv[]) {
    // Parse command line arguments (keep simple, no redundant checks)
    const char* mode = "both";
    if (argc == 2)
        mode = argv[1];

    // Configuration (centralized parameters)
    const int iterations = 500;
    const size_t dataSize = 256ULL * 1024 * 1024; // 256MB
    const int numElements = dataSize / sizeof(int);
    const int blockSize = 256;
    const int gridSize = (numElements + blockSize - 1) / blockSize;
    
    // Print basic config
    std::cout << "Data size: " << dataSize / (1024*1024) << "MB" << std::endl;
    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Kernel configuration: " << gridSize << " grids, " << blockSize << " blocks" << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
    std::cout << "Test mode: " << mode << std::endl;

    // Step 1: Enable peer access first
    enableGPUPeerAccess(0, 1);

    // Step 2: Allocate memory on both GPUs
    int *d0_data, *d1_data;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d0_data, dataSize));
    CHECK(cudaMemset(d0_data, 0, dataSize));
    
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d1_data, dataSize));
    CHECK(cudaMemset(d1_data, 0, dataSize));

    // Wait for user to start counters
    printf("Initialization done! Start counter polling and press...\n");
    getchar();

    // Run tests based on mode (call separate functions)
    if (std::string(mode) == "write" || std::string(mode) == "both") {
        runWriteTest(d0_data, d1_data, numElements, gridSize, blockSize, iterations);
    }
    if (std::string(mode) == "read" || std::string(mode) == "both") {
        runReadTest(d0_data, d1_data, numElements, gridSize, blockSize, iterations);
    }

    // Wait for user to stop counters
    printf("All tests done! Stop counter polling and press...\n");
    getchar();

    // Cleanup global memory
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(d0_data));
    
    CHECK(cudaSetDevice(1));
    CHECK(cudaFree(d1_data));

    std::cout << "\nTests completed" << std::endl;
    return 0;
}