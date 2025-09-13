#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <vector>
#include <cstdio>

#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define REPEAT 50

__global__ void peerWriteKernelV4(
    uint32_t* __restrict__ dst_peer_u32,
    const uint32_t* __restrict__ src_local_u32,
    size_t n_vec4,
    unsigned long long* checksum)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = GRID_SIZE * blockDim.x;
    unsigned long long acc = 0ull;

    for (size_t i = tid; i < n_vec4; i += stride) {
        const uint32_t* src_ptr = src_local_u32 + (i << 2);
        uint32_t* dst_ptr       = dst_peer_u32  + (i << 2);
        uint32_t x0, x1, x2, x3;

        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                     : "l"(src_ptr) : "memory");
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                     :
                     : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
                     : "memory");
        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }

    __syncthreads();
    __threadfence_system();
    if (tid == 0) atomicAdd(checksum, acc);
}

__global__ void peerReadKernelV4(
    uint32_t* __restrict__ dst_local_u32,
    const uint32_t* __restrict__ src_peer_u32,
    size_t n_vec4,
    unsigned long long* checksum)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = GRID_SIZE * blockDim.x;
    unsigned long long acc = 0ull;

    for (size_t i = tid; i < n_vec4; i += stride) {
        const uint32_t* src_ptr = src_peer_u32 + (i << 2);
        uint32_t* dst_ptr       = dst_local_u32  + (i << 2);
        uint32_t x0, x1, x2, x3;

        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                     : "l"(src_ptr) : "memory");
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                     :
                     : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
                     : "memory");
        acc += (unsigned long long)x0 + x1 + x2 + x3;
    }

    __syncthreads();
    __threadfence_system();
    if (tid == 0) atomicAdd(checksum, acc);
}

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

std::string humanReadableSizeInteger(size_t bytes) {
    char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%llu KB", bytes / 1024ULL);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%llu MB", bytes / (1024ULL * 1024ULL));
    } else {
        snprintf(buf, sizeof(buf), "%llu GB", bytes / (1024ULL * 1024ULL * 1024ULL));
    }
    return std::string(buf);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <1|2> <read|write> [base GPU 0|1 for 1-way]" << std::endl;
        return 1;
    }

    int mode = std::stoi(argv[1]);         // 1: one-way, 2: two-way
    std::string op = argv[2];
    int base_gpu = 0;
    if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

    std::vector<size_t> data_sizes = {
        4ULL * 1024ULL,                    // 4 KB
        32ULL * 1024ULL,                   // 32 KB
        128ULL * 1024ULL,                  // 128 KB
        512ULL * 1024ULL,                  // 512 KB
        1ULL * 1024ULL * 1024ULL,          // 1 MB
        100ULL * 1024ULL * 1024ULL,        // 100 MB
        1ULL * 1024ULL * 1024ULL * 1024ULL, // 1 GB
        8ULL * 1024ULL * 1024ULL * 1024ULL  // 8 GB
    };

    size_t max_size = data_sizes.back();

    uint32_t *d_src0 = nullptr, *d_src1 = nullptr, *d_dst0 = nullptr, *d_dst1 = nullptr;
    unsigned long long *d_checksum0 = nullptr, *d_checksum1 = nullptr;

    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, max_size));
    checkCuda(cudaMalloc(&d_dst0, max_size));
    checkCuda(cudaMalloc(&d_checksum0, sizeof(unsigned long long)));
    checkCuda(cudaMemset(d_src0, 5, max_size));
    checkCuda(cudaMemset(d_dst0, 0, max_size));
    checkCuda(cudaMemset(d_checksum0, 0, sizeof(unsigned long long)));

    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, max_size));
    checkCuda(cudaMalloc(&d_dst1, max_size));
    checkCuda(cudaMalloc(&d_checksum1, sizeof(unsigned long long)));
    checkCuda(cudaMemset(d_src1, 5, max_size));
    checkCuda(cudaMemset(d_dst1, 0, max_size));
    checkCuda(cudaMemset(d_checksum1, 0, sizeof(unsigned long long)));

    int canAccess01 = 0, canAccess10 = 0;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01) { checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1,0)); }
    if (canAccess10) { checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0,0)); }

    printf("Press Enter to start the benchmark...\n");
    getchar();

    for (size_t si = 0; si < data_sizes.size(); ++si) {
        size_t DATA_SIZE = data_sizes[si];
        size_t n_vec4 = DATA_SIZE / (4 * sizeof(uint32_t));

        checkCuda(cudaSetDevice(0)); checkCuda(cudaMemset(d_checksum0, 0, sizeof(unsigned long long)));
        checkCuda(cudaSetDevice(1)); checkCuda(cudaMemset(d_checksum1, 0, sizeof(unsigned long long)));

        double total_time0 = 0.0, total_time1 = 0.0;

        for (int iter = 0; iter < REPEAT; ++iter) {
            if (mode == 1) {
                // one-way
                cudaEvent_t start, stop;
                checkCuda(cudaSetDevice(base_gpu));
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&stop));
                checkCuda(cudaEventRecord(start));

                if (op == "read") {
                    if (base_gpu == 0) peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
                    else peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);
                } else {
                    if (base_gpu == 0) peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum0);
                    else peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum1);
                }

                checkCuda(cudaEventRecord(stop));
                checkCuda(cudaEventSynchronize(stop));
                float ms = 0.0f;
                checkCuda(cudaEventElapsedTime(&ms, start, stop));

                if (base_gpu == 0) total_time0 += ms / 1000.0;
                else total_time1 += ms / 1000.0;

                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            } else {
                // two-way
                cudaEvent_t start0, stop0, start1, stop1;
                cudaStream_t stream0, stream1;

                checkCuda(cudaSetDevice(0));
                checkCuda(cudaStreamCreate(&stream0));
                checkCuda(cudaEventCreate(&start0));
                checkCuda(cudaEventCreate(&stop0));

                checkCuda(cudaSetDevice(1));
                checkCuda(cudaStreamCreate(&stream1));
                checkCuda(cudaEventCreate(&start1));
                checkCuda(cudaEventCreate(&stop1));

                checkCuda(cudaSetDevice(0));
                checkCuda(cudaEventRecord(start0, stream0));
                if (op == "read") peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE, 0, stream0>>>(d_dst0, d_src1, n_vec4, d_checksum0);
                else peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE, 0, stream0>>>(d_dst1, d_src0, n_vec4, d_checksum1);
                cudaEventRecord(stop0, stream0);

                checkCuda(cudaSetDevice(1));
                checkCuda(cudaEventRecord(start1, stream1));
                if (op == "read") peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_dst1, d_src0, n_vec4, d_checksum1);
                else peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_dst0, d_src1, n_vec4, d_checksum0);
                cudaEventRecord(stop1, stream1);

                checkCuda(cudaSetDevice(0)); checkCuda(cudaEventSynchronize(stop0));
                checkCuda(cudaSetDevice(1)); checkCuda(cudaEventSynchronize(stop1));

                float ms0 = 0.0f, ms1 = 0.0f;
                checkCuda(cudaEventElapsedTime(&ms0, start0, stop0));
                checkCuda(cudaEventElapsedTime(&ms1, start1, stop1));

                total_time0 += ms0 / 1000.0;
                total_time1 += ms1 / 1000.0;

                // cleanup
                cudaEventDestroy(start0); cudaEventDestroy(stop0);
                cudaEventDestroy(start1); cudaEventDestroy(stop1);
                cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
            }
        } // end REPEAT

        std::cout << std::fixed << std::setprecision(2);
        if (mode == 1) {
            double avg_time = (base_gpu == 0) ? (total_time0 / REPEAT) : (total_time1 / REPEAT);
            double throughput = (double)DATA_SIZE / 1.0e9 / avg_time;
            std::cout << "DataSize: " << humanReadableSizeInteger(DATA_SIZE)
                      << ", 1-way, base GPU " << base_gpu
                      << ", Operation: " << op
                      << ", Avg Throughput: " << throughput << " GB/s\n";
        } else {
            double avg0 = total_time0 / REPEAT;
            double avg1 = total_time1 / REPEAT;
            double tp0 = (double)DATA_SIZE / 1.0e9 / avg0;
            double tp1 = (double)DATA_SIZE / 1.0e9 / avg1;
            std::cout << "DataSize: " << humanReadableSizeInteger(DATA_SIZE)
                      << ", 2-way, GPU0->GPU1 Avg: " << tp0 << " GB/s, "
                      << "GPU1->GPU0 Avg: " << tp1 << " GB/s\n";
        }
    } // end sizes loop

    // Cleanup
    checkCuda(cudaSetDevice(0));
    cudaFree(d_src0); cudaFree(d_dst0); cudaFree(d_checksum0);

    checkCuda(cudaSetDevice(1));
    cudaFree(d_src1); cudaFree(d_dst1); cudaFree(d_checksum1);

    return 0;
}
