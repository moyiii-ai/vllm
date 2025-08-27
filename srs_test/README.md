# vLLM-Related Scripts Overview

This document provides a categorized overview of scripts for starting vLLM servers, clients, and other auxiliary tasks.

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
