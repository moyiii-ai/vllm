# vLLM-Related Scripts Overview

This document provides a categorized overview of scripts for starting vLLM servers, clients, and other auxiliary tasks.

## Scenario 2: Test Execution Steps
1. Start the vLLM server
    * For a single-CPU/no-DP scenario, use `server_lmcache.sh`.
    * For DP + LMCache, use `server_dp_lmcache.sh`.
    * Keep the server running until all tests are completed.
2. Warmup
Run `client_benchmark_warmup.sh` and wait for it to finish.
3. Start the tests
    * If you want to measure PCIe counters for each QPS, run `client_benchmark_single.sh` and adjust the request rate for each run.
    * If you want to rerun all tests in one command, run `client_benchmark_multiple.sh`.

## vLLM Server Startup Scripts

All server-related scripts start with the prefix `server`, and are used to initialize vLLM servers with different configurations:

1. server_tp.sh
2. server_lmcache.sh
3. server_tp_lmcache.sh
4. server_dp_lmcache.sh

## vLLM Client Startup Scripts
All client-related scripts begin with the prefix `client`, and are used to interact with the running vLLM server:
1. **client_simple_query.py**: A minimal script for basic vLLM inference requests (validates server connectivity & simple generation).
2. **client_query_twice.py**: An LMCache project example: sends the same long prompt (from random_text.txt) twice to compare TTFT between cold and warm queries.
3. **client_benchmark_serving.sh**: Runs the vLLM project's `benchmark_serving.py`, with results saved to `benchmark.log`.
4. **client_benchmark_warmup.sh**: Executes a preliminary vLLM benchmark with --warmup flag and output length 1, designed to populate LMCache with all prompts from the dataset for subsequent performance testing.
5. **client_benchmark_single.sh**: Executes a single run of vLLM's benchmark with a fixed request rate, generating a single set of results in JSON format.
6. **client_benchmark_multiple.sh**: Executes multiple runs of vLLM's benchmark, using a predefined list of request rates, with results saved in JSON format.

## Other Auxiliary Scripts
Scripts for supplementary tasks like benchmarking and text preparation:

1. simple_benchmark.sh: 
    * Initializes a basic vLLM server (without Parallelism or LMCache).
    * Uses vllm benchmark_serving.py (an official vLLM tool) to measure key serving metrics (e.g., throughput, TTFT, TPOT, ITL).

2. text_generator.py: 
    * Generates random text and calculates its token count using the Llama-3.1-8B tokenizer.
    * Serves as a dependency for client_query_twice.py: the generated long text is used as a shared common prefix for the client’s two consecutive queries.

3. narrativeqa.jsonl: The LongBench-aligned NarrativeQA dataset file used in vLLM benchmark scripts to provide prompts for inference performance testing.



# vLLM Server/Client Benchmark Scripts
This folder contains scripts to run vLLM servers and clients under different deployment configurations (Data Parallelism, Tensor Parallelism, single GPU, LMCache and RDMA).

## Prerequisites
- Environment with **vLLM v0.10.1** and **LMCache v0.3.3** installed (compatibility with other versions is not guaranteed).
- Adjust script parameters (e.g., target GPU IDs, request rate, dataset name) before execution to match your hardware/experimental setup.

## Basic Workflow
1. Start the target vLLM server.
2. Launch the corresponding client script:
   - The client will exit automatically after completion and write results to log files.
   - The server must be stopped manually (e.g., with `Ctrl+C`).

---

## Experimental Configurations & Usage
Below are step-by-step instructions for running benchmarks across different setups:

### 1. Compare DP vs. Single GPU (with LMCache)
#### 1.1 Single GPU + LMCache
1. Start the server: `./server_lmcache.sh`
2. Warm up LMCache (preload all prompts): `./client_benchmark_warmup.sh` (wait for completion).
3. Run the benchmark: `./client_benchmark_single.sh`

#### 1.2 DP + LMCache
> Note: We observed unexpected behavior with vLLM's `--data-parallel-size 2` flag, so we manually launch separate servers on two GPUs instead.
1. Start two DP servers (each on a separate GPU):
   - `./server_dp0.sh` (GPU 0)
   - `./server_dp1.sh` (GPU 1)
2. Warm up both servers simultaneously: `./client_benchmark_warmup_dp.sh`
3. Run dual-client benchmarks (send traffic to both servers): `./client_benchmark_dp.sh`

### 2. Load Balance: DP + LMCache + RDMA
1. Start DP servers: `./server_dp0.sh` (GPU 0) and `./server_dp1.sh` (GPU 1).
2. Warm up LMCache on both servers: `./client_benchmark_warmup_dp.sh`.
3. Start the RDMA server (on the same GPU as `server_dp0`): `./rdma_replay/bin/rdma_server`.
4. Launch the RDMA client and DP benchmark **simultaneously**:
   - `./rdma_replay/bin/rdma_client`
   - `./client_benchmark_dp.sh`

### 3. Compare TP vs. Single GPU
#### 3.1 Single GPU
1. Start the server (choose one based on LMCache usage):
   - Without LMCache: `./server_simple.sh`
   - With LMCache: `./server_lmcache.sh`
2. Run the benchmark: `./client_benchmark_tp.sh`

#### 3.2 TP
1. Start the TP server (choose one based on LMCache usage):
   - Without LMCache: `./server_tp.sh`
   - With LMCache: `./server_tp_lmcache.sh`
2. Run the benchmark: `./client_benchmark_tp.sh`

---

## Results & Data Analysis
- **Raw Experimental Results**: The original benchmark results are stored in the `original_result` folder.
- **Dataset Naming Rule**: Folders named with `alpaca` correspond to the Alpaca dataset; any folder not containing `alpaca` in its name uses the NarrativeQA dataset by default.
- **Analysis & Plot Scripts**: Data analysis and visualization scripts are provided in the same directory:
  - `DP_plot.py`: For analyzing Data Parallelism benchmark results
  - `TP_plot.py`: For analyzing Tensor Parallelism benchmark results
  - `LB_plot.py`: For analyzing Load Balance benchmark results (DP + LMCache + RDMA)
- **Note**: These scripts read data from specific subfolders under `original_result`. Due to potential changes in folder structure or naming conventions, you may need to modify the file paths in the scripts before execution to match your local setup.

---

## Important Notes
- **Environment Compatibility**: These scripts depend on vLLM's folder structure and environment variables. For best results, use them in the context of the repository:  
  `https://github.com/moyiii-ai/vllm/tree/main/srs_test`
- **Script Dependencies**: Client scripts reference `benchmark_serving_xingyu.py`, which must be placed in the `benchmarks/` directory at the root of the vLLM repository.