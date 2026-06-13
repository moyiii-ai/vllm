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

int main(int argc, char* argv[]) {
    if (argc != 2) { fprintf(stderr, "Usage: %s [read|write]\n", argv[0]); exit(EXIT_FAILURE); }
    bool isReadMode = (strcmp(argv[1], "read") == 0);
    if (!isReadMode && strcmp(argv[1], "write") != 0) { fprintf(stderr, "Invalid mode: use 'read'/'write'\n"); exit(EXIT_FAILURE); }

    const size_t dataSize = 8ULL * 1024 * 1024 * 1024; // 8GB
    const double dataSizeGB = static_cast<double>(dataSize) / (1024 * 1024 * 1024);

    // GPU count & P2P check
    int deviceCount; CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) { fprintf(stderr, "Need at least 2 GPUs\n"); exit(EXIT_FAILURE); }
    int canAccess0to1, canAccess1to0;
    CHECK(cudaSetDevice(0)); CHECK(cudaDeviceCanAccessPeer(&canAccess0to1, 0, 1));
    CHECK(cudaSetDevice(1)); CHECK(cudaDeviceCanAccessPeer(&canAccess1to0, 1, 0));
    if (!canAccess0to1 || !canAccess1to0) { fprintf(stderr, "P2P not supported\n"); exit(EXIT_FAILURE); }

    // Enable P2P & allocate memory
    CHECK(cudaSetDevice(0)); CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaSetDevice(1)); CHECK(cudaDeviceEnablePeerAccess(0, 0)); CHECK(cudaSetDevice(0));
    double *d0, *d1; CHECK(cudaMalloc(&d0, dataSize));
    CHECK(cudaSetDevice(1)); CHECK(cudaMalloc(&d1, dataSize));

    // Initialize memory with pattern
    unsigned char pattern = isReadMode ? 0xBB : 0xAA;
    CHECK(cudaMemset(d1, pattern, dataSize));
    if (!isReadMode) { CHECK(cudaSetDevice(0)); CHECK(cudaMemset(d0, pattern, dataSize)); }
    CHECK(cudaSetDevice(0));

    // Create stream & events
    cudaStream_t stream; cudaEvent_t start, stop;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));

    printf("Initialization done! Start counter polling and press...\n");
    getchar();

    // Timed P2P transfer
    CHECK(cudaEventRecord(start, stream));
    if (isReadMode) {
        CHECK(cudaMemcpyPeerAsync(d0, 0, d1, 1, dataSize, stream));
    } else {
        CHECK(cudaMemcpyPeerAsync(d1, 1, d0, 0, dataSize, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    printf("All tests done! Stop counter polling and press...\n");
    getchar();

    // Calculate time & throughput
    float elapsedMs;
    CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    double elapsedSec = elapsedMs / 1000.0;
    double throughput = dataSizeGB / elapsedSec;

    // Print results - removed reference to non-existent 'verified' variable
    printf("%s transfer completed\n", isReadMode ? "Read" : "Write");
    printf("Size: %.2f GB | Time: %.2f ms | Throughput: %.4f GB/s\n", 
           dataSizeGB, elapsedMs, throughput);

    // Cleanup
    CHECK(cudaFree(d0)); 
    CHECK(cudaSetDevice(1)); 
    CHECK(cudaFree(d1));
    CHECK(cudaDeviceDisablePeerAccess(0)); 
    CHECK(cudaSetDevice(0)); 
    CHECK(cudaDeviceDisablePeerAccess(1));
    cudaStreamDestroy(stream); 
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    return 0;
}
