#!/bin/bash
# Convenience: start master + both vLLMs in background.
# Leaves processes detached; use ./stop_all.sh to tear down.
#
# Usage:
#   ./start_all.sh                     # default: Put pin ON (each role → own segment)
#   ./start_all.sh --l40s-l40s         # L40S local (GPU1) + L40S remote (GPU2)
#   ./start_all.sh --no-prefer-segment # disable Put pin (random segment allocation)
#   GPU_PAIR=l40s-l40s ./start_all.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1 = pin Put to each role's own segment (default); 0 / --no-prefer-segment to disable.
PREFER_SEGMENT="${PREFER_SEGMENT:-1}"
SERVER_EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s ;;
    --no-prefer-segment) PREFER_SEGMENT=0 ;;
    --prefer-segment) PREFER_SEGMENT=1 ;;
  esac
done

case "${PREFER_SEGMENT,,}" in
  1|true|yes|on) PREFER_SEGMENT=1 ;;
  0|false|no|off) PREFER_SEGMENT=0 ;;
  *)
    echo "ERROR: PREFER_SEGMENT must be 0/1/true/false (got: $PREFER_SEGMENT)"
    exit 1
    ;;
esac
# Always pass explicit flag so start_server.sh does not re-apply its own default.
if [[ "$PREFER_SEGMENT" == "1" ]]; then
  SERVER_EXTRA+=(--prefer-segment)
else
  SERVER_EXTRA+=(--no-prefer-segment)
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

echo "Requesting sudo ..."
sudo -v

if ! lsmod | grep -q '^nvidia_peermem'; then
  echo "Loading nvidia_peermem ..."
  sudo modprobe nvidia_peermem
fi

echo "=== GPU_PAIR=$GPU_PAIR ==="
echo "  local:  GPU$LOCAL_GPU ($LOCAL_GPU_NAME)  remote: GPU$REMOTE_GPU ($REMOTE_GPU_NAME)"
echo "  PREFER_SEGMENT=$PREFER_SEGMENT"
resolve_rdma_names || exit 1
echo "  H1 RDMA: $H1_RDMA ($H1_RDMA_BDF)  H2 RDMA: $H2_RDMA ($H2_RDMA_BDF)"
warn_if_local_gpu_pcie_slow

# Tear down without printing queue report (fresh logs come after restart).
SKIP_REPORT=1 bash "$SCRIPT_DIR/stop_all.sh" || true
reset_phase_markers
BACKGROUND=1 bash "$SCRIPT_DIR/start_store.sh"
# Export resolved names so BACKGROUND child shells see the same mlx5_*.
export H1_RDMA H2_RDMA H1_RDMA_BDF H2_RDMA_BDF
BACKGROUND=1 bash "$SCRIPT_DIR/start_server.sh" "${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"}" local
BACKGROUND=1 bash "$SCRIPT_DIR/start_server.sh" "${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"}" remote

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"

echo
echo "Stack ready (GPU_PAIR=$GPU_PAIR, PREFER_SEGMENT=$PREFER_SEGMENT):"
echo "  master:  $MASTER_IP:$MASTER_PORT (NUMA$MASTER_NUMA)"
echo "  local:   http://127.0.0.1:$VLLM_PORT_LOCAL  (inside $H1_NS, GPU$LOCAL_GPU $LOCAL_GPU_NAME)"
echo "  remote:  http://127.0.0.1:$VLLM_PORT_REMOTE (inside $H2_NS, GPU$REMOTE_GPU $REMOTE_GPU_NAME)"
echo "  logs:    $LOG_DIR/vllm_local.log  $LOG_DIR/vllm_remote.log"
echo "           (BACKGROUND: no live tail — use:  tail -F $LOG_DIR/vllm_local.log)"
if [[ "$GPU_PAIR" == "l40s-l40s" ]]; then
  echo "  pex-mon: -d $PEX_MON_DEVICE -p $PEX_MON_PORTS  (112=gpu, 128=cpu)"
fi
echo "Next: ./smoke_warmup.sh && ./run_smoke.sh   # or ./warmup.sh for full split"
echo "      (pass --l40s-l40s to smoke scripts too if you use that pair)"
echo "Then: ./stop_all.sh   # prints max Running/Waiting from the logs above"
