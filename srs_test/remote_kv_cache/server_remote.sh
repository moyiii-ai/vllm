LMCACHE_CONFIG_FILE="mooncake-config.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve \
meta-llama/Llama-3.1-8B-Instruct \
--kv-transfer-config \
'{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  