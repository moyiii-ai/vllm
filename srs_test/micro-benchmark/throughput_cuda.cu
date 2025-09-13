#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdio>

#define GRID_SIZE 256
#define BLOCK_SIZE 256
#define REPEAT 50

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Convert bytes to human-readable string (integer units)
std::string humanReadableSizeInteger(size_t bytes) {
    char buf[64];
    if (bytes < 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024ULL * 1024ULL) {
        snprintf(buf, sizeof(buf), "%llu KB", bytes / 1024ULL);
    } else if (bytes < 1024ULL * 1024ULL * 1024ULL) {
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

    int mode = std::stoi(argv[1]);           // 1: one-way, 2: two-way
    std::string op = argv[2];                // read or write
    int base_gpu = 0;
    if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

    // Fixed data sizes (bytes)
    std::vector<size_t> data_sizes = {
        4ULL * 1024ULL,                    // 4 KB
        32ULL * 1024ULL,                   // 32 KB
        128ULL * 1024ULL,                  // 128 KB
        512ULL * 1024ULL,                  // 512 KB
        1ULL * 1024ULL * 1024ULL,          // 1 MB
        100ULL * 1024ULL * 1024ULL,        // 100 MB
        1ULL * 1024ULL * 1024ULL * 1024ULL,// 1 GB
        8ULL * 1024ULL * 1024ULL * 1024ULL // 8 GB
    };

    size_t max_size = data_sizes.back();

    uint8_t *d_src0=nullptr, *d_src1=nullptr, *d_dst0=nullptr, *d_dst1=nullptr;

    // Allocate memory on both GPUs
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&d_src0, max_size));
    checkCuda(cudaMalloc(&d_dst0, max_size));
    checkCuda(cudaMemset(d_src0, 5, max_size));
    checkCuda(cudaMemset(d_dst0, 0, max_size));

    checkCuda(cudaSetDevice(1));
    checkCuda(cudaMalloc(&d_src1, max_size));
    checkCuda(cudaMalloc(&d_dst1, max_size));
    checkCuda(cudaMemset(d_src1, 5, max_size));
    checkCuda(cudaMemset(d_dst1, 0, max_size));

    // Enable peer access
    int can01=0, can10=0;
    cudaDeviceCanAccessPeer(&can01, 0,1);
    cudaDeviceCanAccessPeer(&can10, 1,0);
    if(can01){ checkCuda(cudaSetDevice(0)); checkCuda(cudaDeviceEnablePeerAccess(1,0)); }
    if(can10){ checkCuda(cudaSetDevice(1)); checkCuda(cudaDeviceEnablePeerAccess(0,0)); }

    printf("Press Enter to start the memcpy benchmark...\n");
    getchar();

    for(size_t si=0; si<data_sizes.size(); ++si){
        size_t DATA_SIZE = data_sizes[si];
        double total_time0=0.0, total_time1=0.0;

        for(int iter=0; iter<REPEAT; ++iter){
            if(mode==1){
                // One-way
                int dst_gpu = (base_gpu==0)?1:0;
                cudaEvent_t start, stop;
                checkCuda(cudaSetDevice(base_gpu));
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&stop));
                checkCuda(cudaEventRecord(start));

                if(op=="read"){
                    // read: copy from peer to local GPU
                    checkCuda(cudaMemcpyPeerAsync(
                        (base_gpu==0)?d_dst0:d_dst1, base_gpu,
                        (base_gpu==0)?d_src1:d_src0, dst_gpu,
                        DATA_SIZE, 0
                    ));
                }else{
                    // write: copy from local GPU to peer
                    checkCuda(cudaMemcpyPeerAsync(
                        (base_gpu==0)?d_dst1:d_dst0, dst_gpu,
                        (base_gpu==0)?d_src0:d_src1, base_gpu,
                        DATA_SIZE, 0
                    ));
                }

                checkCuda(cudaEventRecord(stop));
                checkCuda(cudaEventSynchronize(stop));
                float ms=0.0f;
                checkCuda(cudaEventElapsedTime(&ms, start, stop));
                if(base_gpu==0) total_time0+=ms/1000.0; else total_time1+=ms/1000.0;

                cudaEventDestroy(start); cudaEventDestroy(stop);

            }else{
                // Two-way
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

                // Launch peer memcpy according to operation
                checkCuda(cudaSetDevice(0));
                checkCuda(cudaEventRecord(start0, stream0));
                if(op=="read")
                    checkCuda(cudaMemcpyPeerAsync(d_dst0, 0, d_src1, 1, DATA_SIZE, stream0));
                else
                    checkCuda(cudaMemcpyPeerAsync(d_dst1, 1, d_src0, 0, DATA_SIZE, stream0));
                checkCuda(cudaEventRecord(stop0, stream0));

                checkCuda(cudaSetDevice(1));
                checkCuda(cudaEventRecord(start1, stream1));
                if(op=="read")
                    checkCuda(cudaMemcpyPeerAsync(d_dst1, 1, d_src0, 0, DATA_SIZE, stream1));
                else
                    checkCuda(cudaMemcpyPeerAsync(d_dst0, 0, d_src1, 1, DATA_SIZE, stream1));
                checkCuda(cudaEventRecord(stop1, stream1));

                checkCuda(cudaSetDevice(0)); checkCuda(cudaEventSynchronize(stop0));
                checkCuda(cudaSetDevice(1)); checkCuda(cudaEventSynchronize(stop1));

                float ms0=0.0f, ms1=0.0f;
                checkCuda(cudaEventElapsedTime(&ms0, start0, stop0));
                checkCuda(cudaEventElapsedTime(&ms1, start1, stop1));
                total_time0+=ms0/1000.0;
                total_time1+=ms1/1000.0;

                // Cleanup
                checkCuda(cudaSetDevice(0)); cudaEventDestroy(start0); cudaEventDestroy(stop0); cudaStreamDestroy(stream0);
                checkCuda(cudaSetDevice(1)); cudaEventDestroy(start1); cudaEventDestroy(stop1); cudaStreamDestroy(stream1);
            }
        }

        // Output results
        std::cout << std::fixed << std::setprecision(2);
        if(mode==1){
            double avg_time = (base_gpu==0)?total_time0/REPEAT:total_time1/REPEAT;
            double throughput = (double)DATA_SIZE/1.0e9/avg_time;
            std::cout << "DataSize: "<<humanReadableSizeInteger(DATA_SIZE)
                      << ", 1-way, base GPU "<<base_gpu
                      << ", "<<op<<" Avg Throughput: "<<throughput<<" GB/s\n";
        }else{
            double avg0=total_time0/REPEAT, avg1=total_time1/REPEAT;
            double tp0=(double)DATA_SIZE/1.0e9/avg0;
            double tp1=(double)DATA_SIZE/1.0e9/avg1;
            std::cout << "DataSize: "<<humanReadableSizeInteger(DATA_SIZE)
                      << ", 2-way, GPU0->GPU1 "<<op<<" Avg: "<<tp0<<" GB/s, "
                      << "GPU1->GPU0 "<<op<<" Avg: "<<tp1<<" GB/s\n";
        }
    }

    // Cleanup
    checkCuda(cudaSetDevice(0)); cudaFree(d_src0); cudaFree(d_dst0);
    checkCuda(cudaSetDevice(1)); cudaFree(d_src1); cudaFree(d_dst1);

    return 0;
}
