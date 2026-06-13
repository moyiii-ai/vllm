#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(cmd)                                                \
  do {                                                                 \
    cudaError_t err = cmd;                                             \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

int main() {
  const size_t TRANSFER_SIZE = 32ULL * 1024ULL * 1024ULL;  // 32MB
  const int NUM_ITERATIONS = 1000;
  const int GPU2_ID = 2;
  const int GPU1_ID = 1;
  const int GPU0_ID = 0;

  // Allocate source buffer on GPU2
  CUDA_CHECK(cudaSetDevice(GPU2_ID));
  void* d_src_gpu2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src_gpu2, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_src_gpu2, 0xAA, TRANSFER_SIZE));

  // Allocate destination buffer on GPU1
  CUDA_CHECK(cudaSetDevice(GPU1_ID));
  void* d_dst_gpu1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dst_gpu1, TRANSFER_SIZE));

  // Allocate destination buffer on GPU0
  CUDA_CHECK(cudaSetDevice(GPU0_ID));
  void* d_dst_gpu0 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dst_gpu0, TRANSFER_SIZE));

  // Allocate destination buffer on CPU
  void* h_dst_cpu = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_dst_cpu, TRANSFER_SIZE));

  // Create streams for each transfer
  cudaStream_t stream_gpu2_to_gpu1;
  cudaStream_t stream_gpu2_to_gpu0;
  cudaStream_t stream_gpu2_to_cpu;
  CUDA_CHECK(cudaStreamCreate(&stream_gpu2_to_gpu1));
  CUDA_CHECK(cudaStreamCreate(&stream_gpu2_to_gpu0));
  CUDA_CHECK(cudaStreamCreate(&stream_gpu2_to_cpu));

  // Create events for timing
  cudaEvent_t start_gpu2_to_gpu1, stop_gpu2_to_gpu1;
  cudaEvent_t start_gpu2_to_gpu0, stop_gpu2_to_gpu0;
  cudaEvent_t start_gpu2_to_cpu, stop_gpu2_to_cpu;
  CUDA_CHECK(cudaEventCreate(&start_gpu2_to_gpu1));
  CUDA_CHECK(cudaEventCreate(&stop_gpu2_to_gpu1));
  CUDA_CHECK(cudaEventCreate(&start_gpu2_to_gpu0));
  CUDA_CHECK(cudaEventCreate(&stop_gpu2_to_gpu0));
  CUDA_CHECK(cudaEventCreate(&start_gpu2_to_cpu));
  CUDA_CHECK(cudaEventCreate(&stop_gpu2_to_cpu));

  // Enable peer access bidirectionally
  int can_access_1_to_2 = 0;
  int can_access_2_to_1 = 0;
  int can_access_0_to_2 = 0;
  int can_access_2_to_0 = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_1_to_2, GPU1_ID, GPU2_ID));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_2_to_1, GPU2_ID, GPU1_ID));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_0_to_2, GPU0_ID, GPU2_ID));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_2_to_0, GPU2_ID, GPU0_ID));

  // Enable peer access from GPU1 to GPU2
  if (can_access_1_to_2) {
    CUDA_CHECK(cudaSetDevice(GPU1_ID));
    cudaError_t err = cudaDeviceEnablePeerAccess(GPU2_ID, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      std::cerr << "Warning: Could not enable peer access from GPU1 to GPU2" << std::endl;
    }
  }

  // Enable peer access from GPU2 to GPU1
  if (can_access_2_to_1) {
    CUDA_CHECK(cudaSetDevice(GPU2_ID));
    cudaError_t err = cudaDeviceEnablePeerAccess(GPU1_ID, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      std::cerr << "Warning: Could not enable peer access from GPU2 to GPU1" << std::endl;
    }
  }

  // Enable peer access from GPU0 to GPU2
  if (can_access_0_to_2) {
    CUDA_CHECK(cudaSetDevice(GPU0_ID));
    cudaError_t err = cudaDeviceEnablePeerAccess(GPU2_ID, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      std::cerr << "Warning: Could not enable peer access from GPU0 to GPU2" << std::endl;
    }
  }

  // Enable peer access from GPU2 to GPU0
  if (can_access_2_to_0) {
    CUDA_CHECK(cudaSetDevice(GPU2_ID));
    cudaError_t err = cudaDeviceEnablePeerAccess(GPU0_ID, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      std::cerr << "Warning: Could not enable peer access from GPU2 to GPU0" << std::endl;
    }
  }

  std::cout << "Starting baseline transfers:" << std::endl;
  std::cout << "  GPU2 -> GPU1: " << TRANSFER_SIZE / (1024 * 1024) << " MB" << std::endl;
  std::cout << "  GPU2 -> GPU0: " << TRANSFER_SIZE / (1024 * 1024) << " MB" << std::endl;
  std::cout << "  GPU2 -> CPU:  " << TRANSFER_SIZE / (1024 * 1024) << " MB" << std::endl;
  std::cout << "  Iterations: " << NUM_ITERATIONS << std::endl;
  std::cout << std::endl;

  // Initialize total time accumulators
  float total_time_gpu2_to_gpu1 = 0.0f;
  float total_time_gpu2_to_gpu0 = 0.0f;
  float total_time_gpu2_to_cpu = 0.0f;

  // Perform transfers with independent timing for each iteration
  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    // Transfer 1: GPU2 -> GPU1 (device context set to GPU1)
    CUDA_CHECK(cudaSetDevice(GPU1_ID));
    CUDA_CHECK(cudaEventRecord(start_gpu2_to_gpu1, stream_gpu2_to_gpu1));
    CUDA_CHECK(cudaMemcpyPeerAsync(d_dst_gpu1, GPU1_ID, d_src_gpu2, GPU2_ID, 
                                    TRANSFER_SIZE, stream_gpu2_to_gpu1));
    CUDA_CHECK(cudaEventRecord(stop_gpu2_to_gpu1, stream_gpu2_to_gpu1));

    // Transfer 2: GPU2 -> GPU0 (device context set to GPU0)
    CUDA_CHECK(cudaSetDevice(GPU0_ID));
    CUDA_CHECK(cudaEventRecord(start_gpu2_to_gpu0, stream_gpu2_to_gpu0));
    CUDA_CHECK(cudaMemcpyPeerAsync(d_dst_gpu0, GPU0_ID, d_src_gpu2, GPU2_ID, 
                                    TRANSFER_SIZE, stream_gpu2_to_gpu0));
    CUDA_CHECK(cudaEventRecord(stop_gpu2_to_gpu0, stream_gpu2_to_gpu0));

    // Transfer 3: GPU2 -> CPU
    CUDA_CHECK(cudaSetDevice(GPU2_ID));
    CUDA_CHECK(cudaEventRecord(start_gpu2_to_cpu, stream_gpu2_to_cpu));
    CUDA_CHECK(cudaMemcpyAsync(h_dst_cpu, d_src_gpu2, TRANSFER_SIZE, 
                                cudaMemcpyDeviceToHost, stream_gpu2_to_cpu));
    CUDA_CHECK(cudaEventRecord(stop_gpu2_to_cpu, stream_gpu2_to_cpu));

    // Synchronize streams and accumulate time for this iteration
    CUDA_CHECK(cudaStreamSynchronize(stream_gpu2_to_gpu1));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_gpu2_to_gpu1, stop_gpu2_to_gpu1));
    total_time_gpu2_to_gpu1 += elapsed_ms;

    CUDA_CHECK(cudaStreamSynchronize(stream_gpu2_to_gpu0));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_gpu2_to_gpu0, stop_gpu2_to_gpu0));
    total_time_gpu2_to_gpu0 += elapsed_ms;

    CUDA_CHECK(cudaStreamSynchronize(stream_gpu2_to_cpu));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_gpu2_to_cpu, stop_gpu2_to_cpu));
    total_time_gpu2_to_cpu += elapsed_ms;
  }

  // Output results
  std::cout << "=== Results ===" << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "GPU2 -> GPU1 total time: " << total_time_gpu2_to_gpu1 << " ms" << std::endl;
  std::cout << "GPU2 -> GPU0 total time: " << total_time_gpu2_to_gpu0 << " ms" << std::endl;
  std::cout << "GPU2 -> CPU  total time: " << total_time_gpu2_to_cpu << " ms" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaSetDevice(GPU2_ID));
  CUDA_CHECK(cudaFree(d_src_gpu2));

  CUDA_CHECK(cudaSetDevice(GPU1_ID));
  CUDA_CHECK(cudaFree(d_dst_gpu1));

  CUDA_CHECK(cudaSetDevice(GPU0_ID));
  CUDA_CHECK(cudaFree(d_dst_gpu0));

  CUDA_CHECK(cudaFreeHost(h_dst_cpu));

  CUDA_CHECK(cudaStreamDestroy(stream_gpu2_to_gpu1));
  CUDA_CHECK(cudaStreamDestroy(stream_gpu2_to_gpu0));
  CUDA_CHECK(cudaStreamDestroy(stream_gpu2_to_cpu));

  CUDA_CHECK(cudaEventDestroy(start_gpu2_to_gpu1));
  CUDA_CHECK(cudaEventDestroy(stop_gpu2_to_gpu1));
  CUDA_CHECK(cudaEventDestroy(start_gpu2_to_gpu0));
  CUDA_CHECK(cudaEventDestroy(stop_gpu2_to_gpu0));
  CUDA_CHECK(cudaEventDestroy(start_gpu2_to_cpu));
  CUDA_CHECK(cudaEventDestroy(stop_gpu2_to_cpu));

  return 0;
}
