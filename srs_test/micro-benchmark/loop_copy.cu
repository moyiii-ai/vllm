#include <cuda_runtime.h>
#include <semaphore.h>
#include <fcntl.h>
#include <signal.h>

#include <iostream>
#include <chrono>

volatile bool stop_requested = false;

// Signal handler for Ctrl+C
void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    printf("Ctrl+C pressed\nStopping memcpy measurement...\n");
    stop_requested = true;
  }
}

// CUDA error checking macro
#define CUDA_CHECK(cmd)                                                \
  do {                                                                 \
    cudaError_t err = cmd;                                             \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;           \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// Copy kernel
#define GRID_SIZE 256
#define BLOCK_SIZE 256
__global__ void copyKernel(int* destination, const int* source,
                           size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < numElements; i += stride) {
    destination[i] = source[i];
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <read|write> <gpu_id> [sync]"
              << std::endl;
    return 1;
  }

  std::string mode(argv[1]);
  int gpu_id = std::stoi(argv[2]);
  bool synchronized = (argc == 4 && std::string(argv[3]) == "sync");
  int peer_gpu = (gpu_id == 0) ? 1 : 0;

  signal(SIGINT, sig_int_handler);

  constexpr size_t TRANSFER_SIZE = 8ULL * 1024ULL * 1024ULL * 1024ULL;  // 8GB
  size_t numElements = TRANSFER_SIZE / sizeof(int);

  // Check peer access
  int can_access = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, gpu_id, peer_gpu));
  if (!can_access) {
    std::cerr << "GPU " << gpu_id << " cannot access peer GPU " << peer_gpu
              << std::endl;
    return 1;
  }

  // Enable peer access
  CUDA_CHECK(cudaSetDevice(gpu_id));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_gpu, 0));

  // Allocate memory on self GPU
  int* d_self = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_self, 1, TRANSFER_SIZE));

  // Allocate memory on peer GPU
  CUDA_CHECK(cudaSetDevice(peer_gpu));
  int* d_peer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peer, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_peer, 0, TRANSFER_SIZE));

  // Switch back to current GPU
  CUDA_CHECK(cudaSetDevice(gpu_id));

  // Setup synchronization if requested
  sem_t* sem_self = nullptr;
  sem_t* sem_peer = nullptr;
  if (synchronized) {
    const char* sem_self_name = (gpu_id == 0) ? "/sem0_ready" : "/sem1_ready";
    const char* sem_peer_name = (gpu_id == 0) ? "/sem1_ready" : "/sem0_ready";

    sem_self = sem_open(sem_self_name, O_CREAT, 0644, 0);
    sem_peer = sem_open(sem_peer_name, O_CREAT, 0644, 0);

    if (sem_self == SEM_FAILED || sem_peer == SEM_FAILED) {
      perror("sem_open");
      return 1;
    }

    sem_post(sem_self);  // signal self ready
    sem_wait(sem_peer);  // wait for peer ready
  }

  std::cout << "Start running " << mode << " test on GPU " << gpu_id
            << " ... Press Ctrl+C to stop." << std::endl;

  size_t total_bytes = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Main kernel loop
  while (!stop_requested) {
    if (mode == "read") {
      copyKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_self, d_peer, numElements);
    } else if (mode == "write") {
      copyKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_peer, d_self, numElements);
    } else {
      std::cerr << "Invalid mode: " << mode << std::endl;
      break;
    }
    total_bytes += TRANSFER_SIZE;
  }

  // Wait for all kernels to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = gb / elapsed_s.count();

  // Output results
  std::cerr << "Transferred " << gb << " GB in " << elapsed_s.count()
            << " s. Throughput: " << throughput << " GB/s" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(d_self));
  CUDA_CHECK(cudaFree(d_peer));

  if (synchronized) {
    sem_close(sem_self);
    sem_close(sem_peer);
  }

  return 0;
}
