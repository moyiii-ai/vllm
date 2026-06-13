#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdio>
#include <chrono>

#define GRID_SIZE 256
#define BLOCK_SIZE 256

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
    snprintf(buf, sizeof(buf), "%llu GB",
             bytes / (1024ULL * 1024ULL * 1024ULL));
  }
  return std::string(buf);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <1|2> <read|write> [base GPU 0|1 for 1-way]" << std::endl;
    return 1;
  }

  int mode = std::stoi(argv[1]);  // 1: one-way, 2: two-way
  std::string op = argv[2];       // read or write
  int base_gpu = 0;
  if (mode == 1 && argc >= 4) base_gpu = std::stoi(argv[3]);

  // Fixed data sizes (bytes)
  std::vector<size_t> data_sizes = {
      1ULL * 1024ULL * 1024ULL  // 1 MB
  };

  size_t max_size = data_sizes.back();

  uint8_t *d_src0 = nullptr, *d_src1 = nullptr, *d_dst0 = nullptr,
          *d_dst1 = nullptr;

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
  int can01 = 0, can10 = 0;
  cudaDeviceCanAccessPeer(&can01, 0, 1);
  cudaDeviceCanAccessPeer(&can10, 1, 0);
  if (can01) {
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaDeviceEnablePeerAccess(1, 0));
  }
  if (can10) {
    checkCuda(cudaSetDevice(1));
    checkCuda(cudaDeviceEnablePeerAccess(0, 0));
  }

  printf("Press Enter to start the memcpy benchmark...\n");
  getchar();

  for (size_t si = 0; si < data_sizes.size(); ++si) {
    size_t DATA_SIZE = data_sizes[si];
    double total_time0 = 0.0, total_time1 = 0.0;

    int repeat = 100;
    if (DATA_SIZE < (1ULL << 30)) {  // less than 1GB
      repeat = 20000;
    }

    for (int iter = 0; iter < repeat; ++iter) {
      if (mode == 1) {
        // One-way
        int dst_gpu = (base_gpu == 0) ? 1 : 0;
        auto start = std::chrono::high_resolution_clock::now();

        if (op == "read") {
          checkCuda(cudaMemcpyPeer((base_gpu == 0) ? d_dst0 : d_dst1, base_gpu,
                                   (base_gpu == 0) ? d_src1 : d_src0, dst_gpu,
                                   DATA_SIZE));
        } else {
          checkCuda(cudaMemcpyPeer((base_gpu == 0) ? d_dst1 : d_dst0, dst_gpu,
                                   (base_gpu == 0) ? d_src0 : d_src1, base_gpu,
                                   DATA_SIZE));
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        if (base_gpu == 0)
          total_time0 += elapsed;
        else
          total_time1 += elapsed;

      } else {
        // Two-way
        auto start0 = std::chrono::high_resolution_clock::now();
        checkCuda(cudaSetDevice(0));
        if (op == "read")
          checkCuda(cudaMemcpyPeer(d_dst0, 0, d_src1, 1, DATA_SIZE));
        else
          checkCuda(cudaMemcpyPeer(d_dst1, 1, d_src0, 0, DATA_SIZE));
        auto end0 = std::chrono::high_resolution_clock::now();

        auto start1 = std::chrono::high_resolution_clock::now();
        checkCuda(cudaSetDevice(1));
        if (op == "read")
          checkCuda(cudaMemcpyPeer(d_dst1, 1, d_src0, 0, DATA_SIZE));
        else
          checkCuda(cudaMemcpyPeer(d_dst0, 0, d_src1, 1, DATA_SIZE));
        auto end1 = std::chrono::high_resolution_clock::now();

        total_time0 += std::chrono::duration<double>(end0 - start0).count();
        total_time1 += std::chrono::duration<double>(end1 - start1).count();
      }
    }

    // Output results
    std::cout << std::fixed << std::setprecision(2);
    if (mode == 1) {
      double avg_time =
          (base_gpu == 0) ? total_time0 / repeat : total_time1 / repeat;
      double throughput = (double)DATA_SIZE / 1.0e9 / avg_time;
      std::cout << "DataSize: " << humanReadableSizeInteger(DATA_SIZE)
                << ", 1-way, base GPU " << base_gpu << ", " << op
                << " Avg Throughput: " << throughput << " GB/s\n";
    } else {
      double avg0 = total_time0 / repeat, avg1 = total_time1 / repeat;
      double tp0 = (double)DATA_SIZE / 1.0e9 / avg0;
      double tp1 = (double)DATA_SIZE / 1.0e9 / avg1;
      std::cout << "DataSize: " << humanReadableSizeInteger(DATA_SIZE)
                << ", 2-way, GPU0->GPU1 " << op << " Avg: " << tp0 << " GB/s, "
                << "GPU1->GPU0 " << op << " Avg: " << tp1 << " GB/s\n";
    }
  }

  // Cleanup
  checkCuda(cudaSetDevice(0));
  cudaFree(d_src0);
  cudaFree(d_dst0);
  checkCuda(cudaSetDevice(1));
  cudaFree(d_src1);
  cudaFree(d_dst1);

  return 0;
}
