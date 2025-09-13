#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define REPEAT 50

__global__ void copyKernel(int* destination, const int* source, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < numElements; i += stride) {
        destination[i] = source[i];
    }
}

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

std::string humanReadableSize(size_t bytes) {
    char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.0f KB", bytes / 1024.0);
    } else if (bytes < 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.0f MB", bytes / (1024.0 * 1024.0));
    } else {
        snprintf(buf, sizeof(buf), "%.0f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return std::string(buf);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <1|2> <read|write> [0|1 for one-way base GPU]\n";
        return 1;
    }

    int mode = std::stoi(argv[1]);
    std::string op = argv[2];
    int base_gpu = 0;
    if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

    // data size
    std::vector<size_t> data_sizes = {
        4ULL * 1024,                  // 4 KB
        32ULL * 1024,                 // 32 KB
        128ULL * 1024,                 // 64 KB
        512ULL * 1024,                // 512 KB
        1ULL * 1024 * 1024,           // 1 MB
        100ULL * 1024 * 1024,         // 100 MB
        1ULL * 1024 * 1024 * 1024,     // 1 GB
        8ULL * 1024 * 1024 * 1024    // 8 GB
    };

    int *d_src0, *d_src1, *d_dst0, *d_dst1;
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, data_sizes.back()));
    checkCuda(cudaMalloc(&d_dst0, data_sizes.back()));
    checkCuda(cudaMemset(d_src0, 3, data_sizes.back()));
    checkCuda(cudaMemset(d_dst0, 0, data_sizes.back()));

    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, data_sizes.back()));
    checkCuda(cudaMalloc(&d_dst1, data_sizes.back()));
    checkCuda(cudaMemset(d_src1, 3, data_sizes.back()));
    checkCuda(cudaMemset(d_dst1, 0, data_sizes.back()));

    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01) { checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1, 0)); }
    if (canAccess10) { checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0, 0)); }

    cudaStream_t stream0, stream1;
    cudaEvent_t start0, stop0, start1, stop1;
    checkCuda(cudaSetDevice(0)); checkCuda(cudaStreamCreate(&stream0));
    checkCuda(cudaEventCreate(&start0)); checkCuda(cudaEventCreate(&stop0));
    checkCuda(cudaSetDevice(1)); checkCuda(cudaStreamCreate(&stream1));
    checkCuda(cudaEventCreate(&start1)); checkCuda(cudaEventCreate(&stop1));

    printf("Press Enter to start the benchmark...\n");
    getchar();

    for (int si = 0; si < data_sizes.size(); si++) {
        size_t DATA_SIZE = data_sizes[si];
        size_t numElements = DATA_SIZE / sizeof(int);
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
                    if (dev == 0) copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_dst0, d_src1, numElements);
                    else copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_dst1, d_src0, numElements);
                } else {
                    if (dev == 0) copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_dst1, d_src0, numElements);
                    else copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_dst0, d_src1, numElements);
                }

                checkCuda(cudaEventRecord(stop, stream));
                checkCuda(cudaEventSynchronize(stop));
                float ms = 0.0f;
                checkCuda(cudaEventElapsedTime(&ms, start, stop));
                total_time0 += ms / 1000.0;
            } else {
                checkCuda(cudaSetDevice(0)); checkCuda(cudaEventRecord(start0, stream0));
                checkCuda(cudaSetDevice(1)); checkCuda(cudaEventRecord(start1, stream1));

                checkCuda(cudaSetDevice(0));
                if (op == "read") copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream0>>>(d_dst0, d_src1, numElements);
                else copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream0>>>(d_dst1, d_src0, numElements);

                checkCuda(cudaSetDevice(1));
                if (op == "read") copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_dst1, d_src0, numElements);
                else copyKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_dst0, d_src1, numElements);

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
                    << "DataSize: " << humanReadableSize(DATA_SIZE) << ", "
                    << "1-way, base GPU " << base_gpu
                    << ", Operation: " << op
                    << ", Avg Throughput: " << throughput << " GB/s\n";
        } else {
            double avg0 = total_time0 / REPEAT;
            double avg1 = total_time1 / REPEAT;
            double throughput0 = (DATA_SIZE / 1.0e9) / avg0;
            double throughput1 = (DATA_SIZE / 1.0e9) / avg1;
            std::cout << std::fixed << std::setprecision(2)
                    << "DataSize: " << humanReadableSize(DATA_SIZE) << ", "
                    << "2-way, GPU0->GPU1: " << throughput0 << " GB/s, "
                    << "GPU1->GPU0: " << throughput1 << " GB/s\n";
        }

    }

    checkCuda(cudaSetDevice(0)); cudaFree(d_src0); cudaFree(d_dst0); cudaStreamDestroy(stream0); cudaEventDestroy(start0); cudaEventDestroy(stop0);
    checkCuda(cudaSetDevice(1)); cudaFree(d_src1); cudaFree(d_dst1); cudaStreamDestroy(stream1); cudaEventDestroy(start1); cudaEventDestroy(stop1);

    return 0;
}
