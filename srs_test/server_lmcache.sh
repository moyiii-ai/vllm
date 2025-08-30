#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=200.0
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_USE_LAYERWISE=True

vllm serve \
    meta-llama/Llama-3.1-8B \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'