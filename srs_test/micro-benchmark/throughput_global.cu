#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>
#include <cstdint>

#define DATA_SIZE (8ULL * 1024 * 1024 * 1024) // 8GB
#define BLOCK_SIZE 256
#define REPEAT 50

// Custom P2P write kernel using volatile ld/st
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

        uint32_t x0, x1, x2, x3;
        asm volatile(
            "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "l"(src_ptr)
            : "memory");

        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
            : "memory");

        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }
    __syncthreads();
    __threadfence_system();

    if (tid == 0) atomicAdd(checksum, acc);
}

// Custom P2P read kernel using volatile ld/st
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

        uint32_t x0, x1, x2, x3;
        asm volatile(
            "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "l"(src_ptr)
            : "memory");

        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
            :
            : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
            : "memory");

        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }
    __syncthreads();
    __threadfence_system();

    if (tid == 0) atomicAdd(checksum, acc);
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

    int mode = std::stoi(argv[1]);
    std::string op = argv[2];

    size_t n_vec4 = DATA_SIZE / (4 * sizeof(uint32_t)); // number of 16B vectors
    // size_t copyElements = DATA_SIZE / sizeof(int);
    int gridSize = (n_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint32_t *d_src0, *d_src1, *d_dst0, *d_dst1;
    unsigned long long *d_checksum0, *d_checksum1;

    // Allocate GPU0
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst0, DATA_SIZE));
    checkCuda(cudaMalloc(&d_checksum0, sizeof(unsigned long long)));
    checkCuda(cudaMemset(d_src0, 1, DATA_SIZE));
    checkCuda(cudaMemset(d_dst0, 0, DATA_SIZE));
    checkCuda(cudaMemset(d_checksum0, 0, sizeof(unsigned long long)));

    // Allocate GPU1
    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst1, DATA_SIZE));
    checkCuda(cudaMalloc(&d_checksum1, sizeof(unsigned long long)));
    checkCuda(cudaMemset(d_src1, 1, DATA_SIZE));
    checkCuda(cudaMemset(d_dst1, 0, DATA_SIZE));
    checkCuda(cudaMemset(d_checksum1, 0, sizeof(unsigned long long)));

    // Enable P2P
    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01) { checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1, 0)); }
    if (canAccess10) { checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0, 0)); }

    double total_time_sec = 0.0;

    for (int i = 0; i < REPEAT; i++) {
        if (mode == 1) {            
            checkCuda(cudaSetDevice(0));
            auto t_start = std::chrono::high_resolution_clock::now();

            // cudaEvent_t start, stop;
            // checkCuda(cudaEventCreate(&start));
            // checkCuda(cudaEventCreate(&stop));
            // checkCuda(cudaEventRecord(start));

            if (op == "read")
                peerReadKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
            else
                peerWriteKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);

            checkCuda(cudaDeviceSynchronize());
            auto t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = t_end - t_start;
            total_time_sec += diff.count();

            // checkCuda(cudaEventRecord(stop));
            // checkCuda(cudaEventSynchronize(stop));
            // float ms = 0.0f;
            // checkCuda(cudaEventElapsedTime(&ms, start, stop));
            // total_time_sec += ms / 1000.0;
            // cudaEventDestroy(start);
            // cudaEventDestroy(stop);
        } else {
            auto t_start = std::chrono::high_resolution_clock::now();

            // Launch kernels
            checkCuda(cudaSetDevice(0));
            if (op == "read")
                peerReadKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
            else
                peerWriteKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);

            checkCuda(cudaSetDevice(1));
            if (op == "read")
                peerReadKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);
            else
                peerWriteKernelV4<<<gridSize, BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);

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
    checkCuda(cudaSetDevice(0)); cudaFree(d_src0); cudaFree(d_dst0); cudaFree(d_checksum0);
    checkCuda(cudaSetDevice(1)); cudaFree(d_src1); cudaFree(d_dst1); cudaFree(d_checksum1);

    return 0;
}
