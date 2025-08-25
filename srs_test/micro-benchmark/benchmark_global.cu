// benchmark_global_ldst_v4.cu
// Bidirectional P2P benchmark using SM-driven ld/st.volatile.global.v4.u32
// - Keeps previous "various sizes" bidirectional read/write tests
// - Adds whole-pass warm-up (run entire suite once, discard), then measured pass
// - Adds "All-Reduce-like" sizes (prefill/decode) and measures bidirectional read/write
//
// Build example (A100):
//   nvcc -O2 -arch=compute_80 -code=sm_80 -o benchmark_global_ldst_v4 benchmark_global_ldst_v4.cu
//
// Notes:
// - This benchmark uses inline PTX vectorized 16B loads/stores to better mimic NCCL's style.
// - Uses UVA + peer access; each GPU has two local buffers: send (source) and recv (destination).
// - Each kernel loops with "repeat" to increase runtime and stabilize timing.
// - We accumulate a checksum and use memory clobbers to avoid DCE and force real traffic.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <cstring>

#define CHECK(cmd) do { \
    cudaError_t e__ = (cmd); \
    if (e__ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e__) \
                  << " (code " << int(e__) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

struct DeviceContext {
    int devId;
    // Local buffers:
    // - buf_send: data this GPU will read from (local) or send to peer
    // - buf_recv: data this GPU will write into (local) or receive from peer
    void* buf_send = nullptr;  // 16B aligned
    void* buf_recv = nullptr;  // 16B aligned
};

// Simple filler to touch memory (float)
__global__ void fillPatternF32(float* p, size_t n, float base) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t idx = i; idx < n; idx += stride) {
        p[idx] = base + float(idx % 1024) * 0.001f;
    }
}

// Kernel: peer READ (this GPU loads from peer and stores into its local buffer)
// Vectorized 16B transactions using ld.volatile.global.v4.u32 and st.volatile.global.v4.u32
__global__ void peerReadKernelV4(uint32_t* __restrict__ dst_local_u32,
                                 const uint32_t* __restrict__ src_peer_u32,
                                 size_t n_vec4, int repeat, unsigned long long* checksum)
{
    // n_vec4 is the number of 16-byte vectors (v4.u32)
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    // Per-thread checksum to avoid DCE
    unsigned long long acc = 0ull;

    // Pointers as vector-of-4 u32 (16B per element)
    for (int r = 0; r < repeat; ++r) {
        for (size_t i = tid; i < n_vec4; i += stride) {
            // Compute 16B base addresses (as u32* + 4*i)
            const uint32_t* src_ptr = src_peer_u32 + (i << 2);
            uint32_t* dst_ptr       = dst_local_u32 + (i << 2);

            // Load 16B from peer (volatile)
            uint32_t x0, x1, x2, x3;
            asm volatile(
                "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "l"(src_ptr)
                : "memory");

            // Store 16B into local (volatile write-through to be conservative)
            asm volatile(
                "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
                : "memory");

            // Mix into checksum
            acc += (unsigned long long)x0 + x1 + x2 + x3;
        }
        __syncthreads();
    }

    // Ensure global visibility (system scope)
    __threadfence_system();

    // Reduce per-thread contribution
    if (tid == 0) {
        atomicAdd(checksum, acc);
    }
}

// Kernel: peer WRITE (this GPU loads from its local buffer and stores into peer buffer)
__global__ void peerWriteKernelV4(uint32_t* __restrict__ dst_peer_u32,
                                  const uint32_t* __restrict__ src_local_u32,
                                  size_t n_vec4, int repeat, unsigned long long* checksum)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    unsigned long long acc = 0ull;

    for (int r = 0; r < repeat; ++r) {
        for (size_t i = tid; i < n_vec4; i += stride) {
            const uint32_t* src_ptr = src_local_u32 + (i << 2);
            uint32_t* dst_ptr       = dst_peer_u32  + (i << 2);

            // Load 16B locally (volatile to avoid caching artifacts)
            uint32_t x0, x1, x2, x3;
            asm volatile(
                "ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                : "l"(src_ptr)
                : "memory");

            // Store 16B to peer (volatile)
            asm volatile(
                "st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(dst_ptr), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
                : "memory");

            acc += (unsigned long long)x0 + x1 + x2 + x3;
        }
        __syncthreads();
    }

    __threadfence_system();

    if (tid == 0) {
        atomicAdd(checksum, acc);
    }
}

static inline void initDevice(DeviceContext& ctx, size_t maxBytes) {
    CHECK(cudaSetDevice(ctx.devId));
    // Allocate 16B-aligned buffers (cudaMalloc is sufficiently aligned)
    CHECK(cudaMalloc(&ctx.buf_send, maxBytes));
    CHECK(cudaMalloc(&ctx.buf_recv, maxBytes));

    // Touch memory with a pattern (float-wise) to avoid zero-pages / lazy mapping effects
    const size_t nFloat = maxBytes / sizeof(float);
    const int threads = 256, blocks = 256;
    fillPatternF32<<<blocks, threads>>>((float*)ctx.buf_send, nFloat, 1.0f + ctx.devId * 10.0f);
    fillPatternF32<<<blocks, threads>>>((float*)ctx.buf_recv, nFloat, 0.0f);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

static inline void cleanupDevice(DeviceContext& ctx) {
    CHECK(cudaSetDevice(ctx.devId));
    if (ctx.buf_send) CHECK(cudaFree(ctx.buf_send));
    if (ctx.buf_recv) CHECK(cudaFree(ctx.buf_recv));
}

// Return GB/s for a single kernel launch (read variant)
double runOneRead(int currentDev,
                  void* localDst, const void* peerSrc,
                  size_t bytes, int repeat,
                  int blocks, int threads)
{
    CHECK(cudaSetDevice(currentDev));

    // Count number of 16B vectors
    size_t n_vec4 = bytes / 16;
    // Device checksum buffer
    unsigned long long* d_chk = nullptr;
    CHECK(cudaMalloc(&d_chk, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_chk, 0, sizeof(unsigned long long)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    peerReadKernelV4<<<blocks, threads>>>(
        (uint32_t*)localDst, (const uint32_t*)peerSrc, n_vec4, repeat, d_chk);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    unsigned long long hostChk = 0;
    CHECK(cudaMemcpy(&hostChk, d_chk, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    (void)hostChk;

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_chk));

    // Cross-GPU traffic (bytes over the link) = bytes * repeat (one remote load per 16B)
    const double gb = (double(bytes) * repeat) / 1e9;
    const double sec = ms / 1e3;
    return gb / (sec > 0 ? sec : 1e-12);
}

// Return GB/s for a single kernel launch (write variant)
double runOneWrite(int currentDev,
                   void* peerDst, const void* localSrc,
                   size_t bytes, int repeat,
                   int blocks, int threads)
{
    CHECK(cudaSetDevice(currentDev));

    size_t n_vec4 = bytes / 16;
    unsigned long long* d_chk = nullptr;
    CHECK(cudaMalloc(&d_chk, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_chk, 0, sizeof(unsigned long long)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    peerWriteKernelV4<<<blocks, threads>>>(
        (uint32_t*)peerDst, (const uint32_t*)localSrc, n_vec4, repeat, d_chk);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    unsigned long long hostChk = 0;
    CHECK(cudaMemcpy(&hostChk, d_chk, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    (void)hostChk;

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_chk));

    // Cross-GPU traffic (bytes over the link) = bytes * repeat (one remote store per 16B)
    const double gb = (double(bytes) * repeat) / 1e9;
    const double sec = ms / 1e3;
    return gb / (sec > 0 ? sec : 1e-12);
}

// Heuristic to choose repeat count so each kernel runs ~10–30ms for stability.
int chooseRepeat(size_t bytes) {
    // Tune these as needed. The smaller the message, the larger the repeat.
    if (bytes <= (1u<<20))   return 2000;   // 1 MB
    if (bytes <= (8u<<20))   return 800;
    if (bytes <= (64u<<20))  return 200;
    if (bytes <= (256u<<20)) return 80;
    return 40;
}

// Run a whole pass (all sizes) for bidirectional READ; if print=false, treat as warm-up.
void runPassBiRead(const DeviceContext& d0, const DeviceContext& d1,
                   const std::vector<size_t>& sizes, bool print)
{
    const int blocks = 256, threads = 256;
    for (auto bytes : sizes) {
        // Require 16B multiple
        size_t bytes16 = (bytes / 16) * 16;
        if (bytes16 == 0) continue;
        const int repeat = chooseRepeat(bytes16);
        double bw0 = 0.0, bw1 = 0.0;
        std::thread t0([&]{ bw0 = runOneRead(d0.devId, d0.buf_recv, d1.buf_send, bytes16, repeat, blocks, threads); });
        std::thread t1([&]{ bw1 = runOneRead(d1.devId, d1.buf_recv, d0.buf_send, bytes16, repeat, blocks, threads); });
        t0.join(); t1.join();
        if (print) {
            std::cout << "Read: dev0<-dev1 | Size = " << (bytes16 / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw0 << " GB/s\n";
            std::cout << "Read: dev1<-dev0 | Size = " << (bytes16 / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw1 << " GB/s\n";
        }
    }
}

// Run a whole pass (all sizes) for bidirectional WRITE; if print=false, treat as warm-up.
void runPassBiWrite(const DeviceContext& d0, const DeviceContext& d1,
                    const std::vector<size_t>& sizes, bool print)
{
    const int blocks = 256, threads = 256;
    for (auto bytes : sizes) {
        size_t bytes16 = (bytes / 16) * 16;
        if (bytes16 == 0) continue;
        const int repeat = chooseRepeat(bytes16);
        double bw01 = 0.0, bw10 = 0.0;
        std::thread t0([&]{ bw01 = runOneWrite(d0.devId, d1.buf_recv, d0.buf_send, bytes16, repeat, blocks, threads); });
        std::thread t1([&]{ bw10 = runOneWrite(d1.devId, d0.buf_recv, d1.buf_send, bytes16, repeat, blocks, threads); });
        t0.join(); t1.join();
        if (print) {
            std::cout << "Write: dev0->dev1 | Size = " << (bytes16 / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw01 << " GB/s\n";
            std::cout << "Write: dev1->dev0 | Size = " << (bytes16 / 1024.0 / 1024.0)
                      << " MB | Bandwidth = " << bw10 << " GB/s\n";
        }
    }
}

// Convenience to run: warm-up pass then measured pass
void runWithWarmup(const char* title,
                   void (*passFn)(const DeviceContext&, const DeviceContext&, const std::vector<size_t>&, bool),
                   const DeviceContext& d0, const DeviceContext& d1,
                   const std::vector<size_t>& sizes)
{
    passFn(d0, d1, sizes, /*print=*/false);
    std::cout << title << "\n";
    passFn(d0, d1, sizes, /*print=*/true);
}

int main() {
    // Require at least two devices
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 GPUs.\n";
        return 0;
    }
    const int dev0 = 0, dev1 = 1;

    // Check and enable P2P
    int can01 = 0, can10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CHECK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    if (!can01 || !can10) {
        std::cerr << "Peer access not supported between " << dev0 << " and " << dev1 << "\n";
        return 0;
    }
    CHECK(cudaSetDevice(dev0));
    (void)cudaDeviceEnablePeerAccess(dev1, 0); // ignore if already enabled
    CHECK(cudaSetDevice(dev1));
    (void)cudaDeviceEnablePeerAccess(dev0, 0);

    // Baseline sizes (bytes): 1MB, 8MB, 64MB, 256MB
    std::vector<size_t> sizes_baseline = {
        size_t(1)  << 20,
        size_t(8)  << 20,
        size_t(64) << 20,
        size_t(256)<< 20
    };

    // All-Reduce-like sizes (bytes):
    // - decode (per token per rank) tends to be small (tens of KB); choose 64KB, 256KB
    // - prefill (sequence/batch aggregated) can be multi-MB; choose 8MB, 16MB
    // You can adjust these to your exact TP topology and dtype.
    std::vector<size_t> sizes_allreduce_decode = {
        size_t(8) << 10,   // 8 KB
        size_t(64)  << 10,  // 64 KB
    };
    std::vector<size_t> sizes_allreduce_prefill = {
        size_t(8)  << 20,   // 8 MB
        size_t(16) << 20    // 16 MB
    };

    // Allocate maximum buffer we will use (align to largest)
    size_t maxBytes = 0;
    auto updMax = [&](const std::vector<size_t>& v){ for (auto b : v) if (b > maxBytes) maxBytes = b; };
    updMax(sizes_baseline);
    updMax(sizes_allreduce_decode);
    updMax(sizes_allreduce_prefill);

    DeviceContext d0{dev0}, d1{dev1};
    initDevice(d0, maxBytes);
    initDevice(d1, maxBytes);

    // 1) Baseline: Bidirectional WRITE, then READ (each has a whole-pass warm-up)
    runWithWarmup("=== Bidirectional Write (various sizes) ===", runPassBiWrite, d0, d1, sizes_baseline);
    runWithWarmup("=== Bidirectional Read  (various sizes) ===", runPassBiRead , d0, d1, sizes_baseline);
    printf("\n\n");
    // 2) All-Reduce-like: decode sizes
    runWithWarmup("=== Bidirectional Write (All-Reduce decode sizes) ===", runPassBiWrite, d0, d1, sizes_allreduce_decode);
    runWithWarmup("=== Bidirectional Read  (All-Reduce decode sizes) ===", runPassBiRead , d0, d1, sizes_allreduce_decode);
    printf("\n\n");
    // 3) All-Reduce-like: prefill sizes
    runWithWarmup("=== Bidirectional Write (All-Reduce prefill sizes) ===", runPassBiWrite, d0, d1, sizes_allreduce_prefill);
    runWithWarmup("=== Bidirectional Read  (All-Reduce prefill sizes) ===", runPassBiRead , d0, d1, sizes_allreduce_prefill);

    cleanupDevice(d0);
    cleanupDevice(d1);
    return 0;
}
