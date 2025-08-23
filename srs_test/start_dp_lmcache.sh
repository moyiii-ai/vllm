#!/bin/bash

export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0
# export LMCACHE_CONFIG_FILE="cpu-offload.yaml"
export LMCACHE_USE_EXPERIMENTAL=True

vllm serve \
    meta-llama/Llama-3.1-8B \
    --data-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'