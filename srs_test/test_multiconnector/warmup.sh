#!/bin/bash
# Warmup for MultiConnector Flow:
#   local half  -> Node A (OffloadingConnector stores to A's CPU)
#   remote half -> Node B (MooncakeStore stores to B's embedded segment)
#
# Datasets: reuse test_mooncake narrativeqa_{local,remote}.jsonl
#
# Usage:
#   ./warmup.sh              # both halves in parallel
#   ./warmup.sh local
#   ./warmup.sh remote
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REQUEST_RATE="${REQUEST_RATE:-2}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

ROLE="${1:-both}"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets at:"
  echo "  $DATASET_LOCAL"
  echo "  $DATASET_REMOTE"
  echo "Build them in test_mooncake:"
  echo "  cd $MOONCAKE_TEST_DIR && PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v
echo "=== warmup config (MultiConnector Flow) ==="
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE"
echo "  local  -> Node A Offloading (no max_offload_tokens; allow store)"
echo "  remote -> Node B Mooncake segment $H2_SEGMENT"

warmup_one() {
  local role="$1"
  # Warmup must populate stores — do NOT set max_offload_tokens.
  unset BENCH_EXTRA_BODY || true
  if [[ "$role" == "local" ]]; then
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    phase_mark start warmup
    run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET_LOCAL" "warmup_local"
    phase_mark end warmup
  else
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
    phase_mark start warmup
    run_bench_on "$H2_NS" "$VLLM_PORT_REMOTE" "$DATASET_REMOTE" "warmup_remote"
    phase_mark end warmup
  fi
}

case "$ROLE" in
  local|remote)
    warmup_one "$ROLE"
    ;;
  both|all)
    echo "=== warmup BOTH (parallel) ==="
    echo "  local : $DATASET_LOCAL  -> :$VLLM_PORT_LOCAL (GPU$LOCAL_GPU $LOCAL_GPU_NAME / Offloading)"
    echo "  remote: $DATASET_REMOTE -> :$VLLM_PORT_REMOTE (GPU$REMOTE_GPU $REMOTE_GPU_NAME / Mooncake)"
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
    unset BENCH_EXTRA_BODY || true
    phase_mark start warmup
    run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET_LOCAL" "warmup_local" &
    pid_a=$!
    run_bench_on "$H2_NS" "$VLLM_PORT_REMOTE" "$DATASET_REMOTE" "warmup_remote" &
    pid_b=$!
    ec=0
    wait "$pid_a" || ec=1
    wait "$pid_b" || ec=1
    phase_mark end warmup
    if [[ "$ec" != "0" ]]; then
      echo "ERROR: one warmup client failed"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [both|local|remote]"
    exit 1
    ;;
esac

echo
echo "Warmup done."
echo "  Node A Offloading should hold local-half KV (H2D path on measure)."
echo "  Node B Mooncake should hold remote-half KV (GDRDMA path on measure)."
echo "  Check master:  grep -E 'Mem Storage|Keys:' $LOG_DIR/master.log | tail"
