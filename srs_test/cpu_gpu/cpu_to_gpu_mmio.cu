#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <thread>
#include <cstring>

volatile bool stop_requested = false;

void sig_int_handler(int signum) {
  if (signum == SIGINT && !stop_requested) {
    std::printf("Ctrl+C pressed\nStopping CPU-to-GPU MMIO write...\n");
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
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <dest_gpu_id> [size]"
              << std::endl;
    std::cerr << "Example: " << argv[0] << " 2 1GB" << std::endl;
    return 1;
  }

  int dest_gpu = std::stoi(argv[1]);
  size_t TRANSFER_SIZE = 1ULL * 1024ULL * 1024ULL * 1024ULL; // 1GB default

  if (argc == 3) {
    TRANSFER_SIZE = parse_size(argv[2]);
  }

  signal(SIGINT, sig_int_handler);

  // Print GPU information
  cudaDeviceProp prop_dest;
  CUDA_CHECK(cudaGetDeviceProperties(&prop_dest, dest_gpu));
  print_gpu_info("Destination", prop_dest, dest_gpu);

  // Set device
  CUDA_CHECK(cudaSetDevice(dest_gpu));

  // Allocate mapped host memory using cudaHostAllocMapped
  // This creates a pinned memory region that is mapped into the GPU's address space
  int* host_ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&host_ptr, TRANSFER_SIZE, cudaHostAllocMapped));

  // Get the device pointer that maps to this host pointer
  // This allows CPU writes to host_ptr to be directly visible to GPU via PCIe MMIO
  int* dev_ptr = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));

  size_t numElements = TRANSFER_SIZE / sizeof(int);
  
  // Initialize host memory (this will trigger MMIO writes to GPU)
  for (size_t i = 0; i < numElements; i++) {
    host_ptr[i] = 0xDEADBEEF; // Marker pattern
  }

  std::cout << "Starting CPU->GPU MMIO write to GPU" << dest_gpu
            << ", size=" << format_size(TRANSFER_SIZE)
            << " ... Press Ctrl+C to stop." << std::endl;
  std::cout << "Using mapped pinned memory - CPU writes to host_ptr trigger PCIe MMIO writes to GPU" << std::endl;
  std::cout << "Host pointer: " << std::hex << host_ptr << std::dec << std::endl;
  std::cout << "Device pointer: " << std::hex << dev_ptr << std::dec << std::endl;

  size_t total_bytes = 0;
  size_t iteration = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  while (!stop_requested) {
    // Write to mapped host memory from CPU side
    // CPU writes to host_ptr will immediately flow to GPU via PCIe MMIO (if hardware supports write combining)
    int pattern = 0xDEADBEEF + (iteration % 256);
    
    // Write element by element to ensure each write triggers MMIO
    for (size_t i = 0; i < numElements; i++) {
      host_ptr[i] = pattern;
    }
    
    // Memory barrier to ensure writes are completed
    __sync_synchronize();
    
    total_bytes += TRANSFER_SIZE;
    iteration++;

    // Print progress every 100 iterations
    if (iteration % 100 == 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_s = current_time - start_time;
      double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
      double throughput = gb / elapsed_s.count();
      std::cout << "CPU->GPU" << dest_gpu << " (MMIO)"
                << ": " << iteration << " iterations, "
                << format_size(total_bytes) << " transferred, "
                << std::fixed << std::setprecision(2) << throughput << " GB/s" << std::endl;
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_s = end_time - start_time;
  double gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
  double throughput = gb / elapsed_s.count();

  std::cerr << "\n=== Final Statistics ===" << std::endl;
  std::cerr << "CPU->GPU" << dest_gpu << " (MMIO)" << std::endl;
  std::cerr << "Total iterations: " << iteration << std::endl;
  std::cerr << "Total transferred: " << gb << " GB" << std::endl;
  std::cerr << "Time elapsed: " << elapsed_s.count() << " s" << std::endl;
  std::cerr << "Average throughput: " << throughput << " GB/s" << std::endl;

  CUDA_CHECK(cudaFreeHost(host_ptr));

  return 0;
}
