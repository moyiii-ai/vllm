#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void copyKernel(int* destination, const int* source, size_t numElements) {  // Changed to size_t
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // Changed to size_t
    if (idx < numElements) {
        destination[idx] = source[idx];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [read|write]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    bool isReadMode = (strcmp(argv[1], "read") == 0);
    if (!isReadMode && strcmp(argv[1], "write") != 0) {
        fprintf(stderr, "Invalid mode: Use 'read' or 'write'\n");
        exit(EXIT_FAILURE);
    }

    // Configuration - use size_t for large values
    const size_t totalBytes = 8ULL * 1024 * 1024 * 1024;
    const size_t elemSize = sizeof(int);
    const size_t numElements = totalBytes / elemSize;  // Now correctly holds 2,147,483,648
    const double dataSizeGB = static_cast<double>(totalBytes) / (1024 * 1024 * 1024);

    // Kernel launch params - use dim3 for CUDA configuration
    const int blockSize = 256;
    const dim3 gridDim((numElements + blockSize - 1) / blockSize);  // Safe grid size calculation
    const dim3 blockDim(blockSize);

    CHECK(cudaSetDevice(0));
    
    int deviceCount;
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs\n");
        exit(EXIT_FAILURE);
    }

    int canGPU0AccessGPU1, canGPU1AccessGPU0;
    CHECK(cudaDeviceCanAccessPeer(&canGPU0AccessGPU1, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&canGPU1AccessGPU0, 1, 0));
    
    if (!canGPU0AccessGPU1 || !canGPU1AccessGPU0) {
        fprintf(stderr, "Error: P2P access not supported\n");
        exit(EXIT_FAILURE);
    }

    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceEnablePeerAccess(0, 0));
    CHECK(cudaSetDevice(0));

    int *d_GPU0, *d_GPU1;
    CHECK(cudaMalloc(&d_GPU0, totalBytes));
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d_GPU1, totalBytes));
    CHECK(cudaSetDevice(0));

    unsigned char initPattern = isReadMode ? 0xBB : 0xAA;
    if (isReadMode) {
        CHECK(cudaMemset(d_GPU1, initPattern, totalBytes));
    } else {
        CHECK(cudaMemset(d_GPU0, initPattern, totalBytes));
    }

    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&stopEvent));

    printf("Initialization done! Start counter polling and press...\n");
    getchar();

    CHECK(cudaEventRecord(startEvent, stream));
    
    if (isReadMode) {
        // Use dim3 variables for kernel launch
        copyKernel<<<gridDim, blockDim, 0, stream>>>(d_GPU0, d_GPU1, numElements);
    } else {
        copyKernel<<<gridDim, blockDim, 0, stream>>>(d_GPU1, d_GPU0, numElements);
    }
    
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventRecord(stopEvent, stream));
    CHECK(cudaEventSynchronize(stopEvent));

    printf("All tests done! Stop counter polling and press...\n");
    getchar();

    float elapsedMs;
    CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    double elapsedSec = elapsedMs / 1000.0;
    double throughput = dataSizeGB / elapsedSec;

    printf("%s mode - Transfer completed\n", isReadMode ? "Read" : "Write");
    printf("Size: %.2f GB | Time: %.2f ms | Throughput: %.4f GB/s\n",
           dataSizeGB, elapsedMs, throughput);

    CHECK(cudaDeviceDisablePeerAccess(1));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceDisablePeerAccess(0));
    CHECK(cudaFree(d_GPU1));
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(d_GPU0));

    cudaStreamDestroy(stream);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
