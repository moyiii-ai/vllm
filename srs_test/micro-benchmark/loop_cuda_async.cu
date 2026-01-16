#include <cuda_runtime.h>
#include <semaphore.h>
#include <fcntl.h>
#include <signal.h>

#include <iostream>
#include <chrono>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    printf("Ctrl+C pressed\nStopping memcpy measurement...\n");
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

static inline size_t parse_size(const std::string& s) {
  size_t n = 0;
  char unit = '\0';

  if (sscanf(s.c_str(), "%zu%c", &n, &unit) < 1) {
    std::cerr << "Invalid size format: " << s << std::endl;
    exit(1);
  }

  if (unit == 'K' || unit == 'k') return n * 1024ULL;
  if (unit == 'M' || unit == 'm') return n * 1024ULL * 1024ULL;
  if (unit == 'G' || unit == 'g') return n * 1024ULL * 1024ULL * 1024ULL;

  return n;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <read|write> <gpu_id> [sync] [data_size]" << std::endl;
    return 1;
  }

  std::string mode(argv[1]);
  int gpu_id = std::stoi(argv[2]);

  bool synchronized = false;
  size_t TRANSFER_SIZE = 32ULL * 1024ULL;

  if (argc >= 4) {
    if (std::string(argv[3]) == "sync") {
      synchronized = true;
      if (argc == 5) {
        TRANSFER_SIZE = parse_size(argv[4]);
      }
    } else {
      TRANSFER_SIZE = parse_size(argv[3]);
    }
  }

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

  uint8_t* d_self = nullptr;
  CUDA_CHECK(cudaMalloc(&d_self, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_self, 1, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(peer_gpu));
  uint8_t* d_peer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_peer, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_peer, 0, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(gpu_id));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

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

  auto format_size = [](size_t size) {
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
  };

  std::cout << "Start running " << mode << " test on GPU " << gpu_id
            << " size=" << format_size(TRANSFER_SIZE)
            << " ... Press Ctrl+C to stop." << std::endl;

  size_t total_bytes = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    if (mode == "read") {
      CUDA_CHECK(cudaMemcpyPeerAsync(d_self, gpu_id, d_peer, peer_gpu,
                                     TRANSFER_SIZE, stream));
    } else if (mode == "write") {
      CUDA_CHECK(cudaMemcpyPeerAsync(d_peer, peer_gpu, d_self, gpu_id,
                                     TRANSFER_SIZE, stream));
    } else {
      std::cerr << "Invalid mode: " << mode << std::endl;
      break;
    }
    total_bytes += TRANSFER_SIZE;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = gb / elapsed_s.count();

  std::cerr << "Transferred " << gb << " GB in " << elapsed_s.count()
            << " s. Throughput: " << throughput << " GB/s" << std::flush;

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_self));
  CUDA_CHECK(cudaFree(d_peer));

  if (synchronized) {
    sem_close(sem_self);
    sem_close(sem_peer);
  }

  return 0;
}