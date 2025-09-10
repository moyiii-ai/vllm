#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>

#define DATA_SIZE (8ULL * 1024 * 1024 * 1024) // 8GB
#define BLOCK_SIZE 256
#define REPEAT 50

// P2P write kernel using volatile ld/st
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

// P2P read kernel using volatile ld/st
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

// CUDA error check
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <1|2> <read|write> [base GPU 0|1 for 1-way]" << std::endl;
        return 1;
    }

    int mode = std::stoi(argv[1]);          // 1: one-way, 2: two-way
    std::string op = argv[2];               // "read" or "write"
    int base_gpu = 0;                        // default base GPU
    if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

    size_t n_vec4 = DATA_SIZE / (4 * sizeof(uint32_t));
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
    if (canAccess01) { checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1,0)); }
    if (canAccess10) { checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0,0)); }

    double total_time0 = 0.0, total_time1 = 0.0;

    // Repeat measurement
    for (int i = 0; i < REPEAT; i++) {
        if (mode == 1) {
            // 1-way measurement
            cudaEvent_t start, stop; float ms = 0.0f;
            int gpu = base_gpu;
            checkCuda(cudaSetDevice(gpu));
            checkCuda(cudaEventCreate(&start));
            checkCuda(cudaEventCreate(&stop));
            checkCuda(cudaEventRecord(start));

            if (op == "read") {
                if (gpu == 0) peerReadKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
                else peerReadKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);
            } else {
                if (gpu == 0) peerWriteKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum0);
                else peerWriteKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum1);
            }

            checkCuda(cudaEventRecord(stop));
            checkCuda(cudaEventSynchronize(stop));
            checkCuda(cudaEventElapsedTime(&ms, start, stop));

            if (gpu == 0) total_time0 += ms/1000.0;
            else total_time1 += ms/1000.0;

            cudaEventDestroy(start); cudaEventDestroy(stop);
        } else {
            // 2-way measurement
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
            if (op == "read") peerReadKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
            else peerWriteKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);
            cudaEventRecord(stop0, stream0);

            checkCuda(cudaSetDevice(1));
            checkCuda(cudaEventRecord(start1, stream1));
            if (op == "read") peerReadKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst1, d_src0, n_vec4, d_checksum1);
            else peerWriteKernelV4<<<gridSize,BLOCK_SIZE>>>(d_dst0, d_src1, n_vec4, d_checksum0);
            cudaEventRecord(stop1, stream1); 

            // Synchronize
            checkCuda(cudaSetDevice(0)); cudaEventSynchronize(stop0);
            checkCuda(cudaSetDevice(1)); cudaEventSynchronize(stop1);

            float ms0=0.0f, ms1=0.0f;
            checkCuda(cudaEventElapsedTime(&ms0, start0, stop0));
            checkCuda(cudaEventElapsedTime(&ms1, start1, stop1));

            total_time0 += ms0/1000.0;
            total_time1 += ms1/1000.0;

            cudaEventDestroy(start0); cudaEventDestroy(stop0);
            cudaEventDestroy(start1); cudaEventDestroy(stop1);
        }
    }

    // Compute and print average throughput with 2 decimal places
    std::cout << std::fixed << std::setprecision(2);
    if (mode == 1) {
        double avg_time = (base_gpu==0) ? total_time0/REPEAT : total_time1/REPEAT;
        double throughput = (DATA_SIZE/1.0e9)/avg_time;
        std::cout << "1-way, base GPU " << base_gpu << ", Operation: " << op
                  << ", Avg Throughput: " << throughput << " GB/s\n";
    } else {
        double avg0 = total_time0/REPEAT, avg1 = total_time1/REPEAT;
        double tp0 = (DATA_SIZE/1.0e9)/avg0, tp1 = (DATA_SIZE/1.0e9)/avg1;
        std::cout << "2-way, GPU0->GPU1 Avg: " << tp0 << " GB/s\n"
                  << "2-way, GPU1->GPU0 Avg: " << tp1 << " GB/s\n";
    }

    // Cleanup
    checkCuda(cudaSetDevice(0));
    cudaFree(d_src0); cudaFree(d_dst0); cudaFree(d_checksum0);

    checkCuda(cudaSetDevice(1));
    cudaFree(d_src1); cudaFree(d_dst1); cudaFree(d_checksum1);

    return 0;
}