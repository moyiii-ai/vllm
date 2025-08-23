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
1. client_query_twice.py
2. client_long_doc.py

## Other Auxiliary Scripts
Scripts for supplementary tasks like benchmarking and text preparation:

1. simple_benchmark.sh: 
    * Initializes a basic vLLM server (without Tensor Parallelism or LMCache).
    * Uses vllm benchmark_serving.py (an official vLLM tool) to measure key serving metrics (e.g., throughput, TTFT, TPOT, ITL).

2. text_generator.py: 
    * Generates random text and calculates its token count using the Llama-3.1-8B tokenizer.
    * Serves as a dependency for client_query_twice.py: the generated long text is used as a shared common prefix for the client’s two consecutive queries.
