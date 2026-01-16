// Simple NCCL in-node in-place-style all-gather benchmark on 2 GPUs (0 and 1).
// "In-place" here means each rank sends from its slice within the final receive buffer:
//   sendbuff = recvbuff + rank * count
//   recvbuff size = count * nranks
// We measure per-GPU latency and throughput separately using CUDA events on each device.

#include <cstdio>
#include <cstdlib>
#include <pthread.h>

#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(cmd)                                                         \
  do {                                                                          \
    cudaError_t e = cmd;                                                        \
    if (e != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA failure '%s' at %s:%d\n",                           \
              cudaGetErrorString(e), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

#define CHECK_NCCL(cmd)                                                         \
  do {                                                                          \
    ncclResult_t r = cmd;                                                       \
    if (r != ncclSuccess) {                                                     \
      fprintf(stderr, "NCCL failure '%s' at %s:%d\n",                           \
              ncclGetErrorString(r), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

struct ThreadArgs {
  int rank;
  int nranks;
  ncclUniqueId id;
  size_t count_per_rank;   // number of elements per rank
  int iters;
};

__global__ void init_buffer(float* buf, size_t offset, size_t n, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[offset + idx] = value;
  }
}

void* run_rank(void* arg) {
  ThreadArgs* args = reinterpret_cast<ThreadArgs*>(arg);
  int rank = args->rank;
  int nranks = args->nranks;
  size_t count_per_rank = args->count_per_rank;
  int iters = args->iters;

  // Explicitly bind each rank to one GPU: 0 and 1
  CHECK_CUDA(cudaSetDevice(rank));

  // Create CUDA stream and NCCL communicator
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, nranks, args->id, rank));

  size_t total_count = count_per_rank * nranks;
  size_t bytes_per_iter = total_count * sizeof(float);

  float* d_buf = nullptr;
  CHECK_CUDA(cudaMalloc(&d_buf, bytes_per_iter));

  // Initialize this rank's slice with its rank id as value
  size_t offset = rank * count_per_rank;
  int threads = 256;
  int blocks = static_cast<int>((count_per_rank + threads - 1) / threads);
  init_buffer<<<blocks, threads, 0, stream>>>(d_buf, offset, count_per_rank,
                                              static_cast<float>(rank));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Warmup iterations (not timed)
  int warmup = 5;
  for (int i = 0; i < warmup; ++i) {
    CHECK_NCCL(ncclAllGather(
        /*sendbuff=*/d_buf + offset,
        /*recvbuff=*/d_buf,
        /*sendcount=*/count_per_rank,
        /*datatype=*/ncclFloat,
        /*comm=*/comm,
        /*stream=*/stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Timed iterations
  cudaEvent_t start_evt, stop_evt;
  CHECK_CUDA(cudaEventCreate(&start_evt));
  CHECK_CUDA(cudaEventCreate(&stop_evt));

  CHECK_CUDA(cudaEventRecord(start_evt, stream));
  for (int i = 0; i < iters; ++i) {
    CHECK_NCCL(ncclAllGather(
        /*sendbuff=*/d_buf + offset,
        /*recvbuff=*/d_buf,
        /*sendcount=*/count_per_rank,
        /*datatype=*/ncclFloat,
        /*comm=*/comm,
        /*stream=*/stream));
  }
  CHECK_CUDA(cudaEventRecord(stop_evt, stream));
  CHECK_CUDA(cudaEventSynchronize(stop_evt));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));

  // Average latency per all-gather (ms) for this GPU
  double avg_ms = elapsed_ms / iters;
  // Throughput per GPU in GB/s: based on input data size for fair comparison
  double seconds = avg_ms / 1e3;
  double input_bytes = count_per_rank * sizeof(float);
  double input_gbps = (input_bytes / 1e9) / seconds;

  printf("[Rank %d] count_per_rank=%zu, total_elems=%zu, avg_latency=%.3f ms, throughput=%.3f GB/s\n",
         rank, count_per_rank, total_count, avg_ms, input_gbps);
  fflush(stdout);

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start_evt));
  CHECK_CUDA(cudaEventDestroy(stop_evt));
  CHECK_CUDA(cudaFree(d_buf));
  ncclCommDestroy(comm);
  CHECK_CUDA(cudaStreamDestroy(stream));

  return nullptr;
}

int main(int argc, char* argv[]) {
  // Default: 16M elements per rank (~64 MB if float32) and 100 iterations
  size_t count_per_rank = 1 << 24;
  int iters = 100;

  if (argc >= 2) {
    count_per_rank = static_cast<size_t>(atoll(argv[1]));
  }
  if (argc >= 3) {
    iters = atoi(argv[2]);
  }

  const int nranks = 2;

  // Ensure there are at least 2 GPUs available and print GPU info
  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));
  if (ndev < nranks) {
    fprintf(stderr, "Error: need at least %d GPUs, but only %d found.\n",
            nranks, ndev);
    return EXIT_FAILURE;
  }

  // Print GPU information for each rank
  printf("=== GPU Information ===\n");
  for (int i = 0; i < nranks; ++i) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
    printf("GPU %d: %s (Compute Capability: %d.%d)\n", 
           i, prop.name, prop.major, prop.minor);
  }
  printf("=======================\n\n");

  // Create NCCL unique ID (shared by ranks in this process)
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  pthread_t threads[nranks];
  ThreadArgs args[nranks];

  for (int rank = 0; rank < nranks; ++rank) {
    args[rank].rank = rank;
    args[rank].nranks = nranks;
    args[rank].id = id;
    args[rank].count_per_rank = count_per_rank;
    args[rank].iters = iters;

    int ret = pthread_create(&threads[rank], nullptr, run_rank, &args[rank]);
    if (ret != 0) {
      fprintf(stderr, "Error creating pthread for rank %d (ret=%d)\n", rank, ret);
      return EXIT_FAILURE;
    }
  }

  for (int rank = 0; rank < nranks; ++rank) {
    pthread_join(threads[rank], nullptr);
  }

  return EXIT_SUCCESS;
}

