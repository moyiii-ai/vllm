#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <thread>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    std::printf("Ctrl+C pressed\nStopping CPU-GPU DMA...\n");
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
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <read|write> <gpu_id> [size] [iterations]"
              << std::endl;
    std::cerr << "Example: " << argv[0] << " write 2 32MB 1000  # GPU2 -> CPU, 32MB, 1000 iters" << std::endl;
    std::cerr << "         " << argv[0] << " read 2 1GB        # CPU -> GPU2, 1GB, run until Ctrl+C" << std::endl;
    return 1;
  }

  std::string mode(argv[1]);
  if (mode != "read" && mode != "write") {
    std::cerr << "Error: mode must be 'read' or 'write'" << std::endl;
    return 1;
  }

  int gpu_id = std::stoi(argv[2]);
  size_t TRANSFER_SIZE = 1ULL * 1024ULL * 1024ULL * 1024ULL; // 1GB default
  size_t max_iterations = 0;  // 0 = run until Ctrl+C

  if (argc >= 4) {
    TRANSFER_SIZE = parse_size(argv[3]);
  }
  if (argc >= 5) {
    max_iterations = static_cast<size_t>(std::stoull(argv[4]));
  }

  signal(SIGINT, sig_int_handler);

  // Print GPU information
  cudaDeviceProp prop_gpu;
  CUDA_CHECK(cudaGetDeviceProperties(&prop_gpu, gpu_id));
  print_gpu_info("GPU", prop_gpu, gpu_id);

  // Allocate host (CPU) memory
  int* h_data = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_data, TRANSFER_SIZE));
  
  // Allocate device (GPU) memory
  CUDA_CHECK(cudaSetDevice(gpu_id));
  int* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, TRANSFER_SIZE));

  size_t numElements = TRANSFER_SIZE / sizeof(int);
  
  if (mode == "write") {
    // Write mode: GPU -> CPU (GPU writes/copies to CPU)
    CUDA_CHECK(cudaMemset(d_data, 0xDEADBEEF, TRANSFER_SIZE));
    for (size_t i = 0; i < numElements; i++) {
      h_data[i] = 0;
    }
  } else {
    // Read mode: CPU -> GPU (CPU reads/copies to GPU)
    for (size_t i = 0; i < numElements; i++) {
      h_data[i] = 0xDEADBEEF;
    }
    CUDA_CHECK(cudaMemset(d_data, 0, TRANSFER_SIZE));
  }

  // Set device to GPU for the operation (required for CPU-GPU transfers)
  CUDA_CHECK(cudaSetDevice(gpu_id));

  std::cout << "Starting " << (mode == "write" ? "GPU->CPU" : "CPU->GPU") << gpu_id << " DMA " << mode
            << " (cudaMemcpy), size=" << format_size(TRANSFER_SIZE)
            << ", iterations=" << (max_iterations ? std::to_string(max_iterations) : "until Ctrl+C")
            << std::endl;

  size_t total_bytes = 0;
  size_t iteration = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    if (mode == "write") {
      // GPU -> CPU
      CUDA_CHECK(cudaMemcpy(h_data, d_data, TRANSFER_SIZE, cudaMemcpyDeviceToHost));
    } else {
      // CPU -> GPU
      CUDA_CHECK(cudaMemcpy(d_data, h_data, TRANSFER_SIZE, cudaMemcpyHostToDevice));
    }
    
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

  std::cout << "\n=== CPU-GPU cudaMemcpy Results ===" << std::endl;
  std::cout << (mode == "write" ? "GPU->CPU" : "CPU->GPU") << gpu_id << " (" << mode << ")" << std::endl;
  std::cout << "Iterations: " << iteration << std::endl;
  std::cout << "Total transferred: " << std::fixed << std::setprecision(3) << gb << " GB" << std::endl;
  std::cout << "Time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput: " << std::fixed << std::setprecision(3) << throughput << " GB/s" << std::endl;

  CUDA_CHECK(cudaFreeHost(h_data));
  CUDA_CHECK(cudaFree(d_data));

  return 0;
}
