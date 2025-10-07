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

// Your custom kernel for P2P communication using load/store operations
__global__ void peerWriteKernelV4(
    uint32_t* __restrict__ dst_peer_u32,
    const uint32_t* __restrict__ src_local_u32,
    size_t n_vec4,  
    unsigned long long* checksum)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    unsigned long long acc = 0ull;

    for (size_t i = tid; i < n_vec4; i += stride) {
        const uint32_t* src_ptr = src_local_u32 + (i << 2);
        uint32_t* dst_ptr       = dst_peer_u32  + (i << 2);

        // Load 16B locally (volatile to avoid caching artifacts)
        uint32_t x0, x1, x2, x3;
        asm volatile(
            "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "l"(src_ptr)
            : "memory");

        // Store 16B to peer (volatile)
        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
            : "memory");

        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }
    __syncthreads();

    __threadfence_system();

    if (tid == 0) {
        atomicAdd(checksum, acc);
    }
}

// Kernel for read mode (reverse direction)
__global__ void peerReadKernelV4(
    uint32_t* __restrict__ dst_local_u32,
    const uint32_t* __restrict__ src_peer_u32,
    size_t n_vec4, 
    unsigned long long* checksum)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    unsigned long long acc = 0ull;

    for (size_t i = tid; i < n_vec4; i += stride) {
        const uint32_t* src_ptr = src_peer_u32 + (i << 2);
        uint32_t* dst_ptr       = dst_local_u32  + (i << 2);

        // Load 16B from peer (volatile to avoid caching artifacts)
        uint32_t x0, x1, x2, x3;
        asm volatile(
            "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "l"(src_ptr)
            : "memory");

        // Store 16B locally (volatile)
        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
            : "memory");

        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }
    __syncthreads();

    __threadfence_system();

    if (tid == 0) {
        atomicAdd(checksum, acc);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) { fprintf(stderr, "Usage: %s [read|write]\n", argv[0]); exit(EXIT_FAILURE); }
    bool isReadMode = (strcmp(argv[1], "read") == 0);
    if (!isReadMode && strcmp(argv[1], "write") != 0) { fprintf(stderr, "Invalid mode: use 'read'/'write'\n"); exit(EXIT_FAILURE); }

    // 8GB data (using uint32_t for alignment with your kernel)
    const size_t dataSize = 8ULL * 1024 * 1024 * 1024; // 8GB
    const double dataSizeGB = static_cast<double>(dataSize) / (1024 * 1024 * 1024);
    const size_t n_vec4 = dataSize / (4 * sizeof(uint32_t)); // Number of 16B vectors

    // GPU configuration
    const int blockSize = 256;
    const int gridSize = (n_vec4 + blockSize - 1) / blockSize;

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
    
    uint32_t *d0, *d1; 
    unsigned long long *d_checksum;
    CHECK(cudaMalloc(&d0, dataSize));
    CHECK(cudaMalloc(&d_checksum, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));
    
    CHECK(cudaSetDevice(1)); 
    CHECK(cudaMalloc(&d1, dataSize));

    // Initialize memory with pattern (0xAA for write source, 0xBB for read source)
    unsigned char pattern = isReadMode ? 0xBB : 0xAA;
    CHECK(cudaMemset(d1, pattern, dataSize));
    if (!isReadMode) { 
        CHECK(cudaSetDevice(0)); 
        CHECK(cudaMemset(d0, pattern, dataSize)); 
    }
    CHECK(cudaSetDevice(0));

    // Create stream & events
    cudaStream_t stream; cudaEvent_t start, stop;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));

    printf("Initialization done! Start counter polling and press...\n");
    getchar();

    // Timed P2P transfer using custom kernel
    CHECK(cudaEventRecord(start, stream));
    if (isReadMode) {
        // Read mode: GPU1 -> GPU0 using peerReadKernelV4
        peerReadKernelV4<<<gridSize, blockSize, 0, stream>>>(d0, d1, n_vec4, d_checksum);
    } else {
        // Write mode: GPU0 -> GPU1 using your peerWriteKernelV4
        peerWriteKernelV4<<<gridSize, blockSize, 0, stream>>>(d1, d0, n_vec4, d_checksum);
    }
    CHECK(cudaGetLastError()); // Check for kernel launch errors
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

    // Print results
    printf("%s transfer completed\n", isReadMode ? "Read" : "Write");
    printf("Size: %.2f GB | Time: %.2f ms | Throughput: %.4f GB/s\n", 
           dataSizeGB, elapsedMs, throughput);

    // Cleanup
    CHECK(cudaFree(d0));
    CHECK(cudaFree(d_checksum));
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