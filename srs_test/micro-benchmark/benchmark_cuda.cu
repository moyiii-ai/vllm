// micro_benchmark.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdio>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__)      \
                      << " (code " << err__ << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                           \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// Per-GPU context: two buffers (send/recv) to avoid buffer conflicts in bidirectional tests.
struct DeviceContext {
    int devId;
    char* buf_send = nullptr;
    char* buf_recv = nullptr;
};

// Allocate per-GPU buffers with given capacity (bytes).
void initDevice(DeviceContext& ctx, size_t capacityBytes) {
    CHECK(cudaSetDevice(ctx.devId));
    CHECK(cudaMalloc(&ctx.buf_send, capacityBytes));
    CHECK(cudaMalloc(&ctx.buf_recv, capacityBytes));
}

// Free per-GPU buffers.
void cleanupDevice(DeviceContext& ctx) {
    CHECK(cudaSetDevice(ctx.devId));
    if (ctx.buf_send) CHECK(cudaFree(ctx.buf_send));
    if (ctx.buf_recv) CHECK(cudaFree(ctx.buf_recv));
}

// Single-direction P2P copy loop (srcDev -> dstDev), timed with CUDA events.
// Returns average bandwidth in GB/s over 'iters' copies of 'bytes'.
double runSingleDirection(const void* src, int srcDev, void* dst, int dstDev,
                          size_t bytes, int iters) {
    CHECK(cudaSetDevice(srcDev));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Timed loop
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        CHECK(cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, stream));
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(stream));

    // Avg time per copy (ms)
    const double avg_ms = ms / static_cast<double>(iters);
    // GB/s = (bytes per copy) / (seconds per copy)
    const double gb = static_cast<double>(bytes) / 1e9;
    const double secs = avg_ms / 1000.0;
    return gb / secs;
}

// Helper: run a lambda twice: first as warm-up (no print), then real (print).
template <typename F>
void runWithSingleWarmup(F&& f) {
    f(false);  // warm-up only
    f(true);   // measured run with printing
}

// Bidirectional "write" test (semantics: each GPU writes to the peer).
// Two threads run in parallel: dev0: send->dev1.recv, dev1: send->dev0.recv
void runSimultaneousWrite(const DeviceContext& ctx0, const DeviceContext& ctx1,
                          size_t bytes, int iters, bool print) {
    double bw01 = 0.0, bw10 = 0.0;

    std::thread t0([&]{
        bw01 = runSingleDirection(ctx0.buf_send, ctx0.devId,
                                  ctx1.buf_recv, ctx1.devId,
                                  bytes, iters);
    });
    std::thread t1([&]{
        bw10 = runSingleDirection(ctx1.buf_send, ctx1.devId,
                                  ctx0.buf_recv, ctx0.devId,
                                  bytes, iters);
    });
    t0.join();
    t1.join();

    if (print) {
        std::cout << "Write: dev0->dev1 | Size = " << (bytes / 1024.0 / 1024.0)
                  << " MB | Bandwidth = " << bw01 << " GB/s\n";
        std::cout << "Write: dev1->dev0 | Size = " << (bytes / 1024.0 / 1024.0)
                  << " MB | Bandwidth = " << bw10 << " GB/s\n";
    }
}

// Bidirectional "read" test (semantics: each GPU pulls data from the peer).
// Two threads run in parallel: dev0 reads from dev1 (src=dev1.send -> dst=dev0.recv), and vice versa.
void runSimultaneousRead(const DeviceContext& ctx0, const DeviceContext& ctx1,
                         size_t bytes, int iters, bool print) {
    double bw0reads = 0.0, bw1reads = 0.0;

    std::thread t0([&]{
        bw0reads = runSingleDirection(ctx1.buf_send, ctx1.devId,
                                      ctx0.buf_recv, ctx0.devId,
                                      bytes, iters);
    });
    std::thread t1([&]{
        bw1reads = runSingleDirection(ctx0.buf_send, ctx0.devId,
                                      ctx1.buf_recv, ctx1.devId,
                                      bytes, iters);
    });
    t0.join();
    t1.join();

    if (print) {
        std::cout << "Read: dev0<-dev1 | Size = " << (bytes / 1024.0 / 1024.0)
                  << " MB | Bandwidth = " << bw0reads << " GB/s\n";
        std::cout << "Read: dev1<-dev0 | Size = " << (bytes / 1024.0 / 1024.0)
                  << " MB | Bandwidth = " << bw1reads << " GB/s\n";
    }
}

// ===== Approximate all-reduce scale for LLaMA-3.1-8B =====
// We model only the message size/iteration pattern, not a real collective.
// Settings below are approximate, intended for relative bandwidth comparison.
struct AllReduceConfig {
    int   hidden_dim     = 4096;  // LLaMA-8B approx
    int   num_layers     = 32;
    int   ops_per_layer  = 2;     // typical: 2 all-reduce per layer (attn out + MLP out)
    int   dtype_bytes    = 2;     // FP16/BF16
    int   decode_tokens  = 128;   // number of generated tokens to simulate
    int   prefill_seqlen = 2048;  // typical prefill length
};

// Simulate decode: per-copy size = hidden_dim * dtype_bytes (≈8KB).
// Iterations = decode_tokens * num_layers * ops_per_layer.
void simulateDecode(const DeviceContext& ctx0, const DeviceContext& ctx1,
                    const AllReduceConfig& cfg) {
    const size_t bytes_per_copy = static_cast<size_t>(cfg.hidden_dim) * cfg.dtype_bytes; // ~8KB
    const int iters = cfg.decode_tokens * cfg.num_layers * cfg.ops_per_layer;

    runWithSingleWarmup([&](bool print){
        if (print) {
            std::cout << "\n=== Simulated All-Reduce Scale: Decode ===\n";
            std::cout << "per-copy size ≈ " << (bytes_per_copy / 1024.0) << " KB"
                      << ", iterations = " << iters << "\n";
            std::cout << "-- Bidirectional Write --\n";
        }
        runSimultaneousWrite(ctx0, ctx1, bytes_per_copy, iters, print);
        if (print) std::cout << "-- Bidirectional Read --\n";
        runSimultaneousRead(ctx0, ctx1, bytes_per_copy, iters, print);
    });
}

// Simulate prefill: per-copy size = seqlen * hidden_dim * dtype_bytes (≈16MB for 2048×4096×2).
// Iterations = num_layers * ops_per_layer (per forward pass).
void simulatePrefill(const DeviceContext& ctx0, const DeviceContext& ctx1,
                     const AllReduceConfig& cfg) {
    const size_t bytes_per_copy =
        static_cast<size_t>(cfg.prefill_seqlen) * cfg.hidden_dim * cfg.dtype_bytes; // ~16MB
    const int iters = cfg.num_layers * cfg.ops_per_layer; // per forward

    runWithSingleWarmup([&](bool print){
        if (print) {
            std::cout << "\n=== Simulated All-Reduce Scale: Prefill ===\n";
            std::cout << "per-copy size ≈ " << (bytes_per_copy / 1024.0 / 1024.0) << " MB"
                      << ", iterations = " << iters << "\n";
            std::cout << "-- Bidirectional Write --\n";
        }
        runSimultaneousWrite(ctx0, ctx1, bytes_per_copy, iters, print);
        if (print) std::cout << "-- Bidirectional Read --\n";
        runSimultaneousRead(ctx0, ctx1, bytes_per_copy, iters, print);
    });
}

int main() {
    // Pick two devices (0 and 1 by default).
    int numDevices = 0;
    CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 2) {
        std::cerr << "Need at least 2 GPUs for this benchmark.\n";
        return 0;
    }
    const int dev0 = 0, dev1 = 1;

    int canAccess01 = 0, canAccess10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&canAccess01, dev0, dev1));
    CHECK(cudaDeviceCanAccessPeer(&canAccess10, dev1, dev0));
    if (!canAccess01 || !canAccess10) {
        std::cerr << "Peer access not supported between GPUs " << dev0
                  << " and " << dev1 << "\n";
        return 0;
    }

    // Enable peer access on both devices.
    CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    // Allocate per-GPU buffers sized for the largest test (max of base sizes and prefill size).
    std::vector<size_t> base_sizes = { size_t(1) << 20, size_t(8) << 20, size_t(64) << 20, size_t(256) << 20 };

    AllReduceConfig cfg; // defaults as defined above
    const size_t prefill_bytes = static_cast<size_t>(cfg.prefill_seqlen) * cfg.hidden_dim * cfg.dtype_bytes;

    size_t max_needed = 0;
    for (auto b : base_sizes) max_needed = std::max(max_needed, b);
    max_needed = std::max(max_needed, prefill_bytes);

    DeviceContext ctx0{dev0}, ctx1{dev1};
    initDevice(ctx0, max_needed);
    initDevice(ctx1, max_needed);

    // ===== Base tests: bidirectional write/read for multiple sizes =====
    runWithSingleWarmup([&](bool print){
        if (print) std::cout << "=== Simultaneous Write Test ===\n";
        for (auto bytes : base_sizes) {
            // More iters for small sizes to improve timing stability
            const int iters = (bytes <= (8u << 20)) ? 500 : 100;
            runSimultaneousWrite(ctx0, ctx1, bytes, iters, print);
        }
    });

    runWithSingleWarmup([&](bool print){
        if (print) std::cout << "\n=== Simultaneous Read Test ===\n";
        for (auto bytes : base_sizes) {
            const int iters = (bytes <= (8u << 20)) ? 500 : 100;
            runSimultaneousRead(ctx0, ctx1, bytes, iters, print);
        }
    });

    // ===== Simulate LLaMA-8B all-reduce scales =====
    simulateDecode (ctx0, ctx1, cfg);   // decode: ~8KB per copy, many iterations
    simulatePrefill(ctx0, ctx1, cfg);   // prefill: ~16MB per copy, fewer iterations

    cleanupDevice(ctx0);
    cleanupDevice(ctx1);
    return 0;
}
