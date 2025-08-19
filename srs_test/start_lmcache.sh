#!/bin/bash

export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0
# export LMCACHE_CONFIG_FILE="cpu-offload.yaml"
export LMCACHE_USE_EXPERIMENTAL=True

vllm serve \
    Qwen/Qwen-7B \
    --max-model-len 8192 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --trust-remote-code
