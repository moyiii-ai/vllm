#!/bin/bash

export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0
# export LMCACHE_CONFIG_FILE="cpu-offload.yaml"
export LMCACHE_USE_EXPERIMENTAL=True

vllm serve Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 1 \
    --expert-parallel-size 2 \
    --dtype float16 \
    --max-model-len 32768 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
