#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <iomanip>

#define DATA_SIZE (8ULL * 1024 * 1024 * 1024) // 8GB
#define BLOCK_SIZE 256
#define REPEAT 50

__global__ void copyKernel(int* destination, const int* source, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) destination[idx] = source[idx];
}

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <1|2> <read|write> [0|1 for one-way base GPU]" << std::endl;
        return 1;
    }

    int mode = std::stoi(argv[1]);
    std::string op = argv[2];
    int base_gpu = 0;
    if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

    size_t numElements = DATA_SIZE / sizeof(int);
    int gridSize = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_src0, *d_src1, *d_dst0, *d_dst1 = nullptr;

    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst0, DATA_SIZE));
    checkCuda(cudaMemset(d_src0, 1, DATA_SIZE));
    checkCuda(cudaMemset(d_dst0, 0, DATA_SIZE));

    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, DATA_SIZE));
    checkCuda(cudaMalloc(&d_dst1, DATA_SIZE));
    checkCuda(cudaMemset(d_src1, 1, DATA_SIZE));
    checkCuda(cudaMemset(d_dst1, 0, DATA_SIZE));

    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01) { checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1, 0)); }
    if (canAccess10) { checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0, 0)); }

    cudaStream_t stream0, stream1;
    cudaEvent_t start0, stop0, start1, stop1;

    checkCuda(cudaSetDevice(0));
    checkCuda(cudaStreamCreate(&stream0));
    checkCuda(cudaEventCreate(&start0));
    checkCuda(cudaEventCreate(&stop0));

    checkCuda(cudaSetDevice(1));
    checkCuda(cudaStreamCreate(&stream1));
    checkCuda(cudaEventCreate(&start1));
    checkCuda(cudaEventCreate(&stop1));

    double total_time0 = 0.0, total_time1 = 0.0;

    for (int i = 0; i < REPEAT; i++) {
        if (mode == 1) {
            int dev = base_gpu;
            checkCuda(cudaSetDevice(dev));
            cudaStream_t stream = (dev == 0) ? stream0 : stream1;
            cudaEvent_t start = (dev == 0) ? start0 : start1;
            cudaEvent_t stop  = (dev == 0) ? stop0  : stop1;

            checkCuda(cudaEventRecord(start, stream));

            if (op == "read") {
                if (dev == 0) copyKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_dst0, d_src1, numElements);
                else copyKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_dst1, d_src0, numElements);
            } else {
                if (dev == 0) copyKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_dst1, d_src0, numElements);
                else copyKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_dst0, d_src1, numElements);
            }

            checkCuda(cudaEventRecord(stop, stream));
            checkCuda(cudaEventSynchronize(stop));
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, start, stop));
            total_time0 += ms / 1000.0; // seconds
        } else {
            // Two-way
            checkCuda(cudaSetDevice(0)); checkCuda(cudaEventRecord(start0, stream0));
            checkCuda(cudaSetDevice(1)); checkCuda(cudaEventRecord(start1, stream1));

            checkCuda(cudaSetDevice(0));
            if (op == "read") copyKernel<<<gridSize, BLOCK_SIZE, 0, stream0>>>(d_dst0, d_src1, numElements);
            else copyKernel<<<gridSize, BLOCK_SIZE, 0, stream0>>>(d_dst1, d_src0, numElements);

            checkCuda(cudaSetDevice(1));
            if (op == "read") copyKernel<<<gridSize, BLOCK_SIZE, 0, stream1>>>(d_dst1, d_src0, numElements);
            else copyKernel<<<gridSize, BLOCK_SIZE, 0, stream1>>>(d_dst0, d_src1, numElements);

            checkCuda(cudaSetDevice(0)); checkCuda(cudaEventRecord(stop0, stream0));
            checkCuda(cudaSetDevice(1)); checkCuda(cudaEventRecord(stop1, stream1));

            checkCuda(cudaSetDevice(0)); checkCuda(cudaEventSynchronize(stop0));
            float ms0 = 0.0f; checkCuda(cudaEventElapsedTime(&ms0, start0, stop0));
            total_time0 += ms0 / 1000.0;

            checkCuda(cudaSetDevice(1)); checkCuda(cudaEventSynchronize(stop1));
            float ms1 = 0.0f; checkCuda(cudaEventElapsedTime(&ms1, start1, stop1));
            total_time1 += ms1 / 1000.0;
        }
    }

    if (mode == 1) {
        double avg_time = total_time0 / REPEAT;
        double throughput = (DATA_SIZE / 1.0e9) / avg_time;
        std::cout << std::fixed << std::setprecision(2)
                << "1-way, base GPU " << base_gpu
                << ", Operation: " << op
                << ", Avg Throughput: " << throughput << " GB/s\n";
    } else {
        double avg0 = total_time0 / REPEAT;
        double avg1 = total_time1 / REPEAT;
        double throughput0 = (DATA_SIZE / 1.0e9) / avg0;
        double throughput1 = (DATA_SIZE / 1.0e9) / avg1;
        std::cout << std::fixed << std::setprecision(2)
                << "2-way, GPU0 -> GPU1 Avg Throughput: " << throughput0 << " GB/s\n"
                << "2-way, GPU1 -> GPU0 Avg Throughput: " << throughput1 << " GB/s\n";
    }

    checkCuda(cudaSetDevice(0)); cudaFree(d_src0); cudaFree(d_dst0); cudaStreamDestroy(stream0); cudaEventDestroy(start0); cudaEventDestroy(stop0);
    checkCuda(cudaSetDevice(1)); cudaFree(d_src1); cudaFree(d_dst1); cudaStreamDestroy(stream1); cudaEventDestroy(start1); cudaEventDestroy(stop1);

    return 0;
}
