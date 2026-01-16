#include <cuda_runtime.h>
#include <semaphore.h>
#include <fcntl.h>
#include <signal.h>

#include <iostream>
#include <string>
#include <chrono>
#include <cctype>
#include <thread>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    printf("Ctrl+C pressed\nStopping kernel measurement...\n");
    stop_requested = true;
  }
}

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

std::string format_size(size_t size) {
  double d = static_cast<double>(size);
  const char* unit = "B";
  if (d >= 1024) {
    d /= 1024;
    unit = "KB";
  }
  if (d >= 1024) {
    d /= 1024;
    unit = "MB";
  }
  if (d >= 1024) {
    d /= 1024;
    unit = "GB";
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "%.2f %s", d, unit);
  return std::string(buf);
}

size_t parse_size(const std::string& s) {
  size_t n = 0;
  size_t i = 0;
  while (i < s.size() && std::isdigit(s[i])) {
    n = n * 10 + (s[i] - '0');
    i++;
  }
  std::string unit = s.substr(i);
  for (auto& c : unit) c = std::toupper(c);
  if (unit == "KB") return n * 1024ULL;
  if (unit == "MB") return n * 1024ULL * 1024ULL;
  if (unit == "GB") return n * 1024ULL * 1024ULL * 1024ULL;
  return n;
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <read|write> <gpu_id> [sync] [size]"
              << std::endl;
    return 1;
  }

  std::string mode(argv[1]);
  int gpu_id = std::stoi(argv[2]);

  bool synchronized = false;
  size_t TRANSFER_SIZE = 8ULL * 1024ULL * 1024ULL * 1024ULL;

  for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "sync")
      synchronized = true;
    else
      TRANSFER_SIZE = parse_size(argv[i]);
  }

  size_t numElements = TRANSFER_SIZE / sizeof(uint32_t);
  size_t n_vec4 = numElements / 4;
  int peer_gpu = (gpu_id == 0) ? 1 : 0;

  signal(SIGINT, sig_int_handler);

  int can_access = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, gpu_id, peer_gpu));
  if (!can_access) {
    std::cerr << "GPU " << gpu_id << " cannot access peer GPU " << peer_gpu
              << std::endl;
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(gpu_id));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_gpu, 0));

  uint32_t* d_self = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_self, 1, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(peer_gpu));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu_id, 0));
  uint32_t* d_peer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peer, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_peer, 0, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(gpu_id));
  unsigned long long* d_checksum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));

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
            << " size=" << format_size(TRANSFER_SIZE)
            << " ... Press Ctrl+C to stop.\n";

  // QPS control
  double max_throughput_GB_s = 30.0;
  double qps_limit = 1.0;
  if (TRANSFER_SIZE > 1024 * 1024) {
    double size_GB =
        static_cast<double>(TRANSFER_SIZE) / (1024.0 * 1024 * 1024);
    qps_limit = max_throughput_GB_s / size_GB;
  }
  double interval_s = 1.0 / qps_limit;

  size_t total_bytes = 0;
  auto start_t = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    auto loop_start = std::chrono::high_resolution_clock::now();

    if (mode == "read")
      peerReadKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_self, d_peer, n_vec4,
                                                  d_checksum);
    else if (mode == "write")
      peerWriteKernelV4<<<GRID_SIZE, BLOCK_SIZE>>>(d_peer, d_self, n_vec4,
                                                   d_checksum);

    total_bytes += TRANSFER_SIZE;

    if (TRANSFER_SIZE > 1024 * 1024) {
      auto loop_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = loop_end - loop_start;
      double sleep_s = interval_s - elapsed.count();
      if (sleep_s > 0)
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_t = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_t - start_t;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double tp = gb / elapsed.count();

  std::cerr << "Transferred " << gb << " GB in " << elapsed.count()
            << " s. Throughput: " << tp << " GB/s\n";

  CUDA_CHECK(cudaFree(d_self));
  CUDA_CHECK(cudaFree(d_peer));
  CUDA_CHECK(cudaFree(d_checksum));

  if (synchronized) {
    sem_close(sem_self);
    sem_close(sem_peer);
  }

  return 0;
}
