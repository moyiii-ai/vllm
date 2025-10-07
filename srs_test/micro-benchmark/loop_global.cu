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
    printf("Ctrl+C pressed\nStopping kernel measurement...\n");
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

#define GRID_SIZE 256
#define BLOCK_SIZE 256

// ================================================================
// Peer write kernel
__global__ void peerWriteKernelV4(uint32_t* __restrict__ dst_peer_u32,
                                  const uint32_t* __restrict__ src_local_u32,
                                  size_t n_vec4, unsigned long long* checksum) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = GRID_SIZE * blockDim.x;
  unsigned long long acc = 0ull;

  for (size_t i = tid; i < n_vec4; i += stride) {
    const uint32_t* src_ptr = src_local_u32 + (i << 2);
    uint32_t* dst_ptr = dst_peer_u32 + (i << 2);
    uint32_t x0, x1, x2, x3;

    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                 : "l"(src_ptr)
                 : "memory");
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

// ================================================================
// Peer read kernel
__global__ void peerReadKernelV4(uint32_t* __restrict__ dst_local_u32,
                                 const uint32_t* __restrict__ src_peer_u32,
                                 size_t n_vec4, unsigned long long* checksum) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = GRID_SIZE * blockDim.x;
  unsigned long long acc = 0ull;

  for (size_t i = tid; i < n_vec4; i += stride) {
    const uint32_t* src_ptr = src_peer_u32 + (i << 2);
    uint32_t* dst_ptr = dst_local_u32 + (i << 2);
    uint32_t x0, x1, x2, x3;

    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                 : "l"(src_ptr)
                 : "memory");
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

// ================================================================
// Main
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
  size_t numElements = TRANSFER_SIZE / sizeof(uint32_t);
  size_t n_vec4 = numElements / 4;

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
  uint32_t* d_self = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_self, 1, TRANSFER_SIZE));

  // Allocate memory on peer GPU
  CUDA_CHECK(cudaSetDevice(peer_gpu));
  uint32_t* d_peer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peer, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_peer, 0, TRANSFER_SIZE));

  // Allocate checksum on local GPU
  CUDA_CHECK(cudaSetDevice(gpu_id));
  unsigned long long* d_checksum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));

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

    sem_post(sem_self);
    sem_wait(sem_peer);
  }

  std::cout << "Start running " << mode << " test on GPU " << gpu_id
            << " ... Press Ctrl+C to stop." << std::endl;

  size_t total_bytes = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Main kernel loop
  while (!stop_requested) {
    if (mode == "read") {
      peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_self, d_peer, n_vec4,
                                                  d_checksum);
    } else if (mode == "write") {
      peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_peer, d_self, n_vec4,
                                                   d_checksum);
    } else {
      std::cerr << "Invalid mode: " << mode << std::endl;
      break;
    }
    total_bytes += TRANSFER_SIZE;
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = gb / elapsed_s.count();

  std::cerr << "Transferred " << gb << " GB in " << elapsed_s.count()
            << " s. Throughput: " << throughput << " GB/s" << std::endl;

  CUDA_CHECK(cudaFree(d_self));
  CUDA_CHECK(cudaFree(d_peer));
  CUDA_CHECK(cudaFree(d_checksum));

  if (synchronized) {
    sem_close(sem_self);
    sem_close(sem_peer);
  }

  return 0;
}
