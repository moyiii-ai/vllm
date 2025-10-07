#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>

#define CHECK(cmd) do {                                \
    cudaError_t e = cmd;                               \
    if (e != cudaSuccess) {                            \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) \
                  << " (code " << e << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                            \
    }                                                  \
} while(0)

struct DeviceContext {
    int devId;
    char* buf_send;
    char* buf_recv;
};

// Decide repeat count based on message size
int chooseRepeat(size_t bytes) {
    if (bytes <= (1u<<20))   return 2000;   // <= 1 MB
    if (bytes <= (8u<<20))   return 800;
    if (bytes <= (64u<<20))  return 200;
    if (bytes <= (256u<<20)) return 80;
    return 40;
}

// Run one direction copy and measure bandwidth
double runSingleDirection(int srcDev, char* dst, char* src,
                          size_t bytes, int repeat) {
    CHECK(cudaSetDevice(srcDev));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeat; i++) {
        CHECK(cudaMemcpyPeerAsync(dst, (srcDev^1), src, srcDev, bytes, stream));
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(stream));

    double totalGB = double(bytes) * repeat / 1e9;
    double totalSec = ms / 1e3;
    return totalGB / totalSec;
}

// Bidirectional write test
void runBiWrite(const DeviceContext& d0, const DeviceContext& d1,
                const std::vector<size_t>& sizes, bool print) {
    for (auto bytes : sizes) {
        const int repeat = chooseRepeat(bytes);
        double bw01 = 0.0, bw10 = 0.0;
        std::thread t0([&]{ bw01 = runSingleDirection(d0.devId, d1.buf_recv, d0.buf_send, bytes, repeat); });
        std::thread t1([&]{ bw10 = runSingleDirection(d1.devId, d0.buf_recv, d1.buf_send, bytes, repeat); });
        t0.join(); t1.join();
        if (print) {
            std::cout << "Write: dev0->dev1 | Size = " << (bytes / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw01 << " GB/s\n";
            std::cout << "Write: dev1->dev0 | Size = " << (bytes / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw10 << " GB/s\n";
        }
    }
}

// Bidirectional read test
void runBiRead(const DeviceContext& d0, const DeviceContext& d1,
               const std::vector<size_t>& sizes, bool print) {
    for (auto bytes : sizes) {
        const int repeat = chooseRepeat(bytes);
        double bw01 = 0.0, bw10 = 0.0;
        std::thread t0([&]{ bw01 = runSingleDirection(d1.devId, d0.buf_recv, d1.buf_send, bytes, repeat); });
        std::thread t1([&]{ bw10 = runSingleDirection(d0.devId, d1.buf_recv, d0.buf_send, bytes, repeat); });
        t0.join(); t1.join();
        if (print) {
            std::cout << "Read: dev0<-dev1 | Size = " << (bytes / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw01 << " GB/s\n";
            std::cout << "Read: dev1<-dev0 | Size = " << (bytes / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw10 << " GB/s\n";
        }
    }
}

// Run with warmup wrapper
void runWithWarmup(const char* title,
                   void (*passFn)(const DeviceContext&, const DeviceContext&, const std::vector<size_t>&, bool),
                   const DeviceContext& d0, const DeviceContext& d1,
                   const std::vector<size_t>& sizes) {
    // warmup (not print)
    passFn(d0, d1, sizes, false);
    // real run with output
    std::cout << title << "\n";
    passFn(d0, d1, sizes, true);
}

// Run tests based on type
void runTestsByType(const std::string& testType,
                   const DeviceContext& d0, const DeviceContext& d1,
                   const std::vector<size_t>& various_sizes,
                   const std::vector<size_t>& decode_sizes,
                   const std::vector<size_t>& prefill_sizes) {
    if (testType == "write") {
        // Only run write tests
        runWithWarmup("=== Bidirectional Write (various sizes) ===", runBiWrite, d0, d1, various_sizes);
        std::cout << "\n\n";
        runWithWarmup("=== Bidirectional Write (All-Reduce decode sizes) ===", runBiWrite, d0, d1, decode_sizes);
        std::cout << "\n\n";
        runWithWarmup("=== Bidirectional Write (All-Reduce prefill sizes) ===", runBiWrite, d0, d1, prefill_sizes);
    } else if (testType == "read") {
        // Only run read tests
        runWithWarmup("=== Bidirectional Read  (various sizes) ===", runBiRead,  d0, d1, various_sizes);
        std::cout << "\n\n";
        runWithWarmup("=== Bidirectional Read  (All-Reduce decode sizes) ===", runBiRead,  d0, d1, decode_sizes);
        std::cout << "\n\n";
        runWithWarmup("=== Bidirectional Read  (All-Reduce prefill sizes) ===", runBiRead,  d0, d1, prefill_sizes);
    } else {
        // Run all tests (default behavior)
        runWithWarmup("=== Bidirectional Write (various sizes) ===", runBiWrite, d0, d1, various_sizes);
        runWithWarmup("=== Bidirectional Read  (various sizes) ===", runBiRead,  d0, d1, various_sizes);
        std::cout << "\n\n";

        runWithWarmup("=== Bidirectional Write (All-Reduce decode sizes) ===", runBiWrite, d0, d1, decode_sizes);
        runWithWarmup("=== Bidirectional Read  (All-Reduce decode sizes) ===", runBiRead,  d0, d1, decode_sizes);
        std::cout << "\n\n";

        runWithWarmup("=== Bidirectional Write (All-Reduce prefill sizes) ===", runBiWrite, d0, d1, prefill_sizes);
        runWithWarmup("=== Bidirectional Read  (All-Reduce prefill sizes) ===", runBiRead,  d0, d1, prefill_sizes);
    }
}

void initDevice(DeviceContext& ctx, size_t cap) {
    CHECK(cudaSetDevice(ctx.devId));
    CHECK(cudaMalloc(&ctx.buf_send, cap));
    CHECK(cudaMalloc(&ctx.buf_recv, cap));
    CHECK(cudaMemset(ctx.buf_send, 0, cap));
    CHECK(cudaMemset(ctx.buf_recv, 0, cap));
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string testType;
    if (argc == 2) {
        testType = argv[1];
        if (testType != "write" && testType != "read") {
            std::cerr << "Invalid argument. Use: " << argv[0] << " [write|read]\n";
            return EXIT_FAILURE;
        }
    } else if (argc > 2) {
        std::cerr << "Too many arguments. Use: " << argv[0] << " [write|read]\n";
        return EXIT_FAILURE;
    }

    // Message sizes
    std::vector<size_t> various_sizes = { size_t(1)<<20, size_t(8)<<20, size_t(64)<<20, size_t(256)<<20 };
    std::vector<size_t> decode_sizes  = { size_t(8)<<10, size_t(64)<<10 };   // 8KB, 64KB
    std::vector<size_t> prefill_sizes = { size_t(8)<<20, size_t(16)<<20 };   // 8MB, 16MB

    size_t max_needed = 0;
    for (auto v: various_sizes) max_needed = std::max(max_needed, v);
    for (auto v: decode_sizes)  max_needed = std::max(max_needed, v);
    for (auto v: prefill_sizes) max_needed = std::max(max_needed, v);

    DeviceContext d0{0}, d1{1};
    initDevice(d0, max_needed);
    initDevice(d1, max_needed);

    // Enable peer access
    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceEnablePeerAccess(0, 0));

    // Run tests based on type
    runTestsByType(testType, d0, d1, various_sizes, decode_sizes, prefill_sizes);

    // Cleanup
    CHECK(cudaFree(d0.buf_send));
    CHECK(cudaFree(d0.buf_recv));
    CHECK(cudaFree(d1.buf_send));
    CHECK(cudaFree(d1.buf_recv));
    return 0;
}
    