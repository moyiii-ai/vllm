#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <thread>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    std::printf("Ctrl+C pressed\nStopping GPU-to-GPU DMA...\n");
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

void print_gpu_info(const char* which, const cudaDeviceProp& prop, int id) {
  std::cout << which << " GPU (ID " << id << "): " << prop.name
            << ", Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 6) {
    std::cerr << "Usage: " << argv[0] << " <read|write> <source_gpu_id> <dest_gpu_id> [size] [iterations]"
              << std::endl;
    std::cerr << "Example: " << argv[0] << " write 4 3 32MB 1000  # GPU4 -> GPU3, 32MB, 1000 iters" << std::endl;
    std::cerr << "         " << argv[0] << " read 4 3 1GB         # GPU3 -> GPU4 (GPU4 reads), run until Ctrl+C" << std::endl;
    return 1;
  }

  std::string mode(argv[1]);
  if (mode != "read" && mode != "write") {
    std::cerr << "Error: mode must be 'read' or 'write'" << std::endl;
    return 1;
  }

  int source_gpu = std::stoi(argv[2]);
  int dest_gpu = std::stoi(argv[3]);
  size_t TRANSFER_SIZE = 1ULL * 1024ULL * 1024ULL * 1024ULL; // 1GB default
  size_t max_iterations = 0;  // 0 = run until Ctrl+C

  if (argc >= 5) {
    TRANSFER_SIZE = parse_size(argv[4]);
  }
  if (argc >= 6) {
    max_iterations = static_cast<size_t>(std::stoull(argv[5]));
  }

  signal(SIGINT, sig_int_handler);

  // For read mode, source and dest are swapped conceptually
  // read mode: GPU2 -> GPU0/1, so GPU0/1 reads from GPU2
  // write mode: GPU0/1 -> GPU2, so GPU0/1 writes to GPU2
  int actual_source, actual_dest;
  if (mode == "write") {
    actual_source = source_gpu;  // GPU0/1
    actual_dest = dest_gpu;      // GPU2
  } else {
    actual_source = dest_gpu;    // GPU2 (source of data)
    actual_dest = source_gpu;    // GPU0/1 (destination of data)
  }

  // Check peer access capability in both directions
  int can_access_src_to_dst = 0;
  int can_access_dst_to_src = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_src_to_dst, actual_source, actual_dest));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_dst_to_src, actual_dest, actual_source));
  
  if (!can_access_src_to_dst && !can_access_dst_to_src) {
    std::cerr << "GPUs " << actual_source << " and " << actual_dest 
              << " cannot access each other via peer access" << std::endl;
    return 1;
  }

  // Print GPU information
  cudaDeviceProp prop_source, prop_dest;
  CUDA_CHECK(cudaGetDeviceProperties(&prop_source, actual_source));
  CUDA_CHECK(cudaGetDeviceProperties(&prop_dest, actual_dest));
  print_gpu_info("Source", prop_source, actual_source);
  print_gpu_info("Destination", prop_dest, actual_dest);

  // Enable peer access from source to destination
  CUDA_CHECK(cudaSetDevice(actual_source));
  cudaError_t err_src = cudaDeviceEnablePeerAccess(actual_dest, 0);
  if (err_src != cudaSuccess && err_src != cudaErrorPeerAccessAlreadyEnabled) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err_src) << " at " 
              << __FILE__ << ":" << __LINE__ << std::endl;
    exit(EXIT_FAILURE);
  }

  // Enable peer access from destination to source (bidirectional)
  CUDA_CHECK(cudaSetDevice(actual_dest));
  cudaError_t err_dst = cudaDeviceEnablePeerAccess(actual_source, 0);
  if (err_dst != cudaSuccess && err_dst != cudaErrorPeerAccessAlreadyEnabled) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err_dst) << " at " 
              << __FILE__ << ":" << __LINE__ << std::endl;
    exit(EXIT_FAILURE);
  }

  // Allocate buffers
  CUDA_CHECK(cudaSetDevice(actual_source));
  int* d_source = nullptr;
  CUDA_CHECK(cudaMalloc(&d_source, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_source, actual_source + 1, TRANSFER_SIZE));

  CUDA_CHECK(cudaSetDevice(actual_dest));
  int* d_dest = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dest, TRANSFER_SIZE));
  CUDA_CHECK(cudaMemset(d_dest, 0, TRANSFER_SIZE));

  // Set device context for the operation
  // For write: source_gpu (GPU0/1) writes to dest_gpu (GPU2)
  // For read: source_gpu (GPU0/1) reads from dest_gpu (GPU2)
  // In both cases, operation is initiated from source_gpu context
  CUDA_CHECK(cudaSetDevice(source_gpu));

  std::cout << "Starting GPU" << actual_source << "->GPU" << actual_dest 
            << " (cudaMemcpyPeer, " << mode << ", initiated from GPU" << source_gpu << ")"
            << ", size=" << format_size(TRANSFER_SIZE)
            << ", iterations=" << (max_iterations ? std::to_string(max_iterations) : "until Ctrl+C")
            << std::endl;

  size_t total_bytes = 0;
  size_t iteration = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    // Use cudaMemcpyPeer for DMA copy between GPUs
    // Operation is initiated from source_gpu context
    CUDA_CHECK(cudaMemcpyPeer(d_dest, actual_dest, d_source, actual_source, TRANSFER_SIZE));
    
    total_bytes += TRANSFER_SIZE;
    iteration++;
    if (max_iterations > 0 && iteration >= max_iterations)
      break;

    // Print progress every 100 iterations (only when running until Ctrl+C)
    if (max_iterations == 0 && iteration % 100 == 0) {
      CUDA_CHECK(cudaDeviceSynchronize());
      auto current_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_s = current_time - start_time;
      double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
      double throughput = gb / elapsed_s.count();
      // std::cout << " ... " << iteration << " iters, " << std::fixed << std::setprecision(2) << throughput << " GB/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = (elapsed_s.count() > 0) ? (gb / elapsed_s.count()) : 0.0;
  double elapsed_ms = elapsed_s.count() * 1000.0;

  std::cout << "\n=== GPU-GPU cudaMemcpyPeer Results ===" << std::endl;
  std::cout << "GPU" << actual_source << "->GPU" << actual_dest << " (" << mode << ")" << std::endl;
  std::cout << "Iterations: " << iteration << std::endl;
  std::cout << "Total transferred: " << std::fixed << std::setprecision(3) << gb << " GB" << std::endl;
  std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput: " << std::fixed << std::setprecision(3) << throughput << " GB/s" << std::endl;

  // Free buffers
  CUDA_CHECK(cudaSetDevice(actual_source));
  CUDA_CHECK(cudaFree(d_source));
  
  CUDA_CHECK(cudaSetDevice(actual_dest));
  CUDA_CHECK(cudaFree(d_dest));

  return 0;
}
