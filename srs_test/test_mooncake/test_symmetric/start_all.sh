#!/bin/bash
# Convenience: start master + both vLLMs in background.
#
# Usage:
#   ./start_all.sh                         # default GPU_PAIR=l40s-l40s, Put pin ON
#   ./start_all.sh --a100-l40s
#   ./start_all.sh --no-prefer-segment
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

if ! grep -qx nvidia_peermem < <(lsmod | awk '{print $1}'); then
  echo "Loading nvidia_peermem ..."
  sudo modprobe nvidia_peermem
fi

echo "=== GPU_PAIR=$GPU_PAIR (symmetric MultiConnector) ==="
echo "  local:  GPU$LOCAL_GPU ($LOCAL_GPU_NAME) Offloading ${OFFLOADING_CPU_BYTES}B + Mooncake $GLOBAL_SEGMENT_SIZE pin=$H1_SEGMENT"
echo "  remote: GPU$REMOTE_GPU ($REMOTE_GPU_NAME) Offloading ${OFFLOADING_CPU_BYTES}B + Mooncake $GLOBAL_SEGMENT_SIZE pin=$H2_SEGMENT"
echo "  PREFER_SEGMENT=$PREFER_SEGMENT"
resolve_rdma_names || exit 1
echo "  H1 RDMA: $H1_RDMA ($H1_RDMA_BDF)  H2 RDMA: $H2_RDMA ($H2_RDMA_BDF)"
warn_if_local_gpu_pcie_slow
warn_if_remote_gpu_pcie_slow

SKIP_REPORT=1 bash "$SCRIPT_DIR/stop_all.sh" || true
reset_phase_markers
BACKGROUND=1 bash "$SCRIPT_DIR/start_store.sh"
export H1_RDMA H2_RDMA H1_RDMA_BDF H2_RDMA_BDF
BACKGROUND=1 bash "$SCRIPT_DIR/start_server.sh" "${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"}" local
BACKGROUND=1 bash "$SCRIPT_DIR/start_server.sh" "${SERVER_EXTRA[@]+"${SERVER_EXTRA[@]}"}" remote

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"

echo
echo "Stack ready (GPU_PAIR=$GPU_PAIR, PREFER_SEGMENT=$PREFER_SEGMENT):"
echo "  master:  $MASTER_IP:$MASTER_PORT (NUMA$MASTER_NUMA)"
echo "  local:   http://127.0.0.1:$VLLM_PORT_LOCAL  ($H1_NS, GPU$LOCAL_GPU $LOCAL_GPU_NAME)"
echo "  remote:  http://127.0.0.1:$VLLM_PORT_REMOTE ($H2_NS, GPU$REMOTE_GPU $REMOTE_GPU_NAME)"
echo "  logs:    $LOG_DIR/vllm_local.log  $LOG_DIR/vllm_remote.log"
if [[ "$GPU_PAIR" == "l40s-l40s" ]]; then
  echo "  pex-mon: -d $PEX_MON_DEVICE -p $PEX_MON_PORTS  (112=gpu, 128=cpu; NUMA0 L40S switch only)"
fi
echo "Next: ./warmup.sh && ./run_measure.sh"
echo "Then: ./stop_all.sh"
