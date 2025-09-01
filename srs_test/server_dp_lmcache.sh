#!/bin/bash

export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=100.0
export LMCACHE_USE_EXPERIMENTAL=True

vllm serve \
    meta-llama/Llama-3.1-8B-Instruct \
    --data-parallel-size 2 \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'