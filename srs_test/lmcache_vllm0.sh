LMCACHE_CONFIG_FILE=lmcache_config.yaml \
CUDA_VISIBLE_DEVICES=0 \
vllm serve meta-llama/Llama-3.1-8B \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --port 8000 --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'