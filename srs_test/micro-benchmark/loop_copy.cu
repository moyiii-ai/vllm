#include <cuda_runtime.h>
#include <semaphore.h>
#include <fcntl.h>
#include <signal.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cctype>
#include <thread>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    std::printf("Ctrl+C pressed\nStopping memcpy measurement...\n");
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

#define GRID_SIZE 256   // number of blocks; best if multiple of SMs
#define BLOCK_SIZE 256  // this is the number of threads per block

__global__ void copyKernel(int* destination, const int* source,
                           size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < numElements; i += stride) {
    destination[i] = source[i];
  }
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
  std::snprintf(buf, sizeof(buf), "%.2f %s", d, unit);
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

// Improved readability: declare as a regular function for clarity
void print_gpu_info(const char* which, const cudaDeviceProp& prop, int id) {
  std::cout << which << " GPU (ID " << id << "): " << prop.name
            << ", Compute Capability: " << prop.major << "." << prop.minor
            << ", PCIe Address: "
            << std::hex << std::setw(2) << std::setfill('0') << prop.pciBusID
            << ":" << std::setw(2) << std::setfill('0') << prop.pciDeviceID
            << "." << prop.pciDomainID
            << std::dec << std::endl;
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

  size_t numElements = TRANSFER_SIZE / sizeof(int);
  int peer_gpu = (gpu_id == 0) ? 1 : 0;

  // Print name, compute capability, and PCIe address for source and peer GPUs

  cudaDeviceProp prop_source, prop_peer;
  CUDA_CHECK(cudaGetDeviceProperties(&prop_source, gpu_id));
  CUDA_CHECK(cudaGetDeviceProperties(&prop_peer, peer_gpu));

  print_gpu_info("Source", prop_source, gpu_id);
  print_gpu_info("Peer", prop_peer, peer_gpu);


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

  int* d_self = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_self, 1, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(peer_gpu));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu_id, 0));
  int* d_peer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peer, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_peer, 0, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(gpu_id));

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
            << " ... Press Ctrl+C to stop." << std::endl;

  // QPS control
  double max_throughput_GB_s = 40.0;
  double qps_limit = 1.0;
  if (TRANSFER_SIZE > 1024 * 1024) {
    double size_GB =
        static_cast<double>(TRANSFER_SIZE) / (1024.0 * 1024 * 1024);
    qps_limit = max_throughput_GB_s / size_GB;
  }
  double interval_s = 1.0 / qps_limit;

  size_t total_bytes = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    auto loop_start = std::chrono::high_resolution_clock::now();

    if (mode == "read")
      copyKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_self, d_peer, numElements);
    else if (mode == "write")
      copyKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_peer, d_self, numElements);

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

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = gb / elapsed_s.count();

  std::cerr << "Transferred " << gb << " GB in " << elapsed_s.count()
            << " s. Throughput: " << throughput << " GB/s\n";

  CUDA_CHECK(cudaFree(d_self));
  CUDA_CHECK(cudaFree(d_peer));

  if (synchronized) {
    sem_close(sem_self);
    sem_close(sem_peer);
  }

  return 0;
}
