#!/bin/bash
# Start one embedded MooncakeStoreConnector vLLM worker.
# Usage:
#   ./start_server.sh local              # default GPU_PAIR=a100-l40s → GPU0 A100
#   ./start_server.sh --l40s-l40s local  # local on GPU1 L40S
#   ./start_server.sh remote             # GPU2 L40S / NUMA1 / h2 / mlx5_3 / :8001
#   ./start_server.sh local                   # default: Put pin ON
#   ./start_server.sh --no-prefer-segment local  # disable Put pin
#   BACKGROUND=1 or --bg to detach.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ROLE=""
BACKGROUND="${BACKGROUND:-0}"
# 1 = pin Put to this role's segment (default); 0 / --no-prefer-segment to disable.
PREFER_SEGMENT="${PREFER_SEGMENT:-1}"
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s ;;
    --bg) BACKGROUND=1 ;;
    --no-prefer-segment) PREFER_SEGMENT=0 ;;
    --prefer-segment) PREFER_SEGMENT=1 ;;
    local|remote|gpu0|0|a100|gpu1|1|gpu2|2)
      if [[ -z "$ROLE" ]]; then ROLE="$arg"; else ARGS+=("$arg"); fi
      ;;
    *) ARGS+=("$arg") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
ROLE="${ROLE:-local}"

case "${PREFER_SEGMENT,,}" in
  1|true|yes|on) PREFER_SEGMENT=1 ;;
  0|false|no|off) PREFER_SEGMENT=0 ;;
  *)
    echo "ERROR: PREFER_SEGMENT must be 0/1/true/false (got: $PREFER_SEGMENT)"
    exit 1
    ;;
esac

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda
mkdir -p "$LOG_DIR"

case "$ROLE" in
  local|gpu0|0|a100|gpu1|1)
    ROLE=local
    NS="$H1_NS"
    NUMA=0
    GPU="$LOCAL_GPU"
    RDMA_BDF="$H1_RDMA_BDF"
    HOSTNAME_IP="$H1_IP"
    SEGMENT="$H1_SEGMENT"
    PORT="$VLLM_PORT_LOCAL"
    CFG="$MOONCAKE_CONFIG_LOCAL"
    LOG="$LOG_DIR/vllm_local.log"
    PIDF="$LOG_DIR/vllm_local.pid"
    TAILF="$LOG_DIR/vllm_local_tail.pid"
    ;;
  remote|gpu2|2)
    ROLE=remote
    NS="$H2_NS"
    NUMA=1
    GPU="$REMOTE_GPU"
    RDMA_BDF="$H2_RDMA_BDF"
    HOSTNAME_IP="$H2_IP"
    SEGMENT="$H2_SEGMENT"
    PORT="$VLLM_PORT_REMOTE"
    CFG="$MOONCAKE_CONFIG_REMOTE"
    LOG="$LOG_DIR/vllm_remote.log"
    PIDF="$LOG_DIR/vllm_remote.pid"
    TAILF="$LOG_DIR/vllm_remote_tail.pid"
    ;;
  *)
    echo "Usage: $0 [--l40s-l40s|--a100-l40s] [--prefer-segment|--no-prefer-segment] {local|remote} [--bg]"
    echo "  GPU_PAIR=$GPU_PAIR  local=GPU$LOCAL_GPU ($LOCAL_GPU_NAME)  remote=GPU$REMOTE_GPU ($REMOTE_GPU_NAME)"
    echo "  PREFER_SEGMENT=$PREFER_SEGMENT (default 1; --no-prefer-segment or PREFER_SEGMENT=0 to disable Put pin)"
    exit 1
    ;;
esac

# Re-resolve mlx5_* by PCI BDF and verify visible in this role's netns.
resolve_rdma_names || exit 1
if [[ "$ROLE" == "local" ]]; then
  RDMA="$H1_RDMA"
else
  RDMA="$H2_RDMA"
fi
verify_rdma_device "$RDMA" "$RDMA_BDF" "$NS" || exit 1

export MOONCAKE_CONFIG_PATH="$CFG"
write_mooncake_config "$CFG" "$RDMA" "$HOSTNAME_IP"

# Normalize ENABLE_CROSS_LAYERS_BLOCKS → JSON true/false for kv_connector_extra_config.
case "${ENABLE_CROSS_LAYERS_BLOCKS,,}" in
  1|true|yes|on) CROSS_LAYERS_JSON=true ;;
  0|false|no|off|"") CROSS_LAYERS_JSON=false ;;
  *)
    echo "ERROR: ENABLE_CROSS_LAYERS_BLOCKS must be 0/1/true/false (got: $ENABLE_CROSS_LAYERS_BLOCKS)"
    exit 1
    ;;
esac
# Display / logging copy (pretty). Actual argv is built with escaped quotes below.
KV_TRANSFER_CONFIG="{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"enable_cross_layers_blocks\":${CROSS_LAYERS_JSON}}}"

if ! lsmod | grep -q '^nvidia_peermem'; then
  echo "WARNING: nvidia_peermem not loaded (sudo modprobe nvidia_peermem)."
fi

VLLM_BIN="$(command -v vllm)"
: >"$LOG"

PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping vLLM $ROLE ..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  if ip netns list 2>/dev/null | awk '{print $1}' | grep -qx "$NS"; then
    sudo ip netns pids "$NS" 2>/dev/null | while read -r p; do
      cmd="$(ps -o args= -p "$p" 2>/dev/null || true)"
      if echo "$cmd" | grep -Eq "vllm|VLLM"; then
        # Only kill serve on this role's port when possible.
        if echo "$cmd" | grep -Eq -- "--port[= ]?$PORT|$PORT"; then
          sudo kill -9 "$p" 2>/dev/null || true
        elif echo "$cmd" | grep -Eq 'vllm serve'; then
          sudo kill -9 "$p" 2>/dev/null || true
        fi
      fi
    done
  fi
  rm -f "$PIDF" "$TAILF"
  echo "vLLM $ROLE stopped."
}

echo "=== vLLM $ROLE (GPU_PAIR=$GPU_PAIR) ==="
echo "  netns=$NS  NUMA=$NUMA  CUDA_VISIBLE_DEVICES=$GPU  RDMA=$RDMA  host=$HOSTNAME_IP  port=$PORT"
if [[ "$ROLE" == "local" ]]; then
  echo "  local GPU$LOCAL_GPU ($LOCAL_GPU_NAME)"
  warn_if_local_gpu_pcie_slow
else
  echo "  remote GPU$REMOTE_GPU ($REMOTE_GPU_NAME)"
fi
echo "  CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "  DRAM contribution=$GLOBAL_SEGMENT_SIZE  max_model_len=$MAX_MODEL_LEN  block_size=$BLOCK_SIZE"
if [[ "$PREFER_SEGMENT" == "1" ]]; then
  echo "  TE/segment pin: REQUESTER_LOCAL_HOSTNAME=$SEGMENT  PREFERRED_SEGMENT=$SEGMENT"
else
  echo "  TE/segment pin: REQUESTER_LOCAL_HOSTNAME=$SEGMENT  PREFERRED_SEGMENT=<unset>"
fi
echo "  MC_STORE_MEMCPY=$MC_STORE_MEMCPY"
echo "  ENABLE_CROSS_LAYERS_BLOCKS=$CROSS_LAYERS_JSON"
echo "  model=$MODEL"
echo "  config=$CFG"
echo "  kv_transfer_config=$KV_TRANSFER_CONFIG"

# Optional Put pin. Explicit unset avoids inheriting a parent export via sudo -E.
if [[ "$PREFER_SEGMENT" == "1" ]]; then
  PREFERRED_SEGMENT_EXPORT="export MOONCAKE_PREFERRED_SEGMENT='$SEGMENT'"
else
  PREFERRED_SEGMENT_EXPORT="unset MOONCAKE_PREFERRED_SEGMENT"
fi

netns_run "$NS" numactl --cpunodebind="$NUMA" --membind="$NUMA" bash -c "
  export CUDA_DEVICE_ORDER='$CUDA_DEVICE_ORDER'
  export CUDA_VISIBLE_DEVICES=$GPU
  export PYTHONHASHSEED=0
  export MOONCAKE_CONFIG_PATH='$CFG'
  export MOONCAKE_REQUESTER_LOCAL_HOSTNAME='$SEGMENT'
  $PREFERRED_SEGMENT_EXPORT
  export MC_STORE_MEMCPY='$MC_STORE_MEMCPY'
  export HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN:-}'
  export HF_TOKEN='${HF_TOKEN:-}'
  export HF_HUB_OFFLINE='$HF_HUB_OFFLINE'
  export TRANSFORMERS_OFFLINE='$TRANSFORMERS_OFFLINE'
  exec $VLLM_BIN serve '$MODEL' \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --block-size $BLOCK_SIZE \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\":\"MooncakeStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"enable_cross_layers_blocks\":$CROSS_LAYERS_JSON}}'
" >"$LOG" 2>&1 &
VLLM_PID=$!
PIDS+=("$VLLM_PID")
echo "$VLLM_PID" >"$PIDF"

echo "Launching (wrapper pid $VLLM_PID). API in $NS: http://127.0.0.1:$PORT"
echo "Streaming $LOG (Ctrl+C or ./stop_all.sh to stop)."

if [[ "$BACKGROUND" == "1" ]]; then
  echo "BACKGROUND=1: detaching."
  exit 0
fi

trap cleanup INT TERM EXIT
tail -n +1 -F "$LOG" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")
echo "$TAIL_PID" >"$TAILF"
wait "$TAIL_PID"
