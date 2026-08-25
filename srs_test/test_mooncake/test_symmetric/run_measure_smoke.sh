#!/bin/bash
# After warmup_smoke.sh: two sequential completions on the LOCAL server.
#
#   1) local-dataset  prompt -> expect Offloading CPU→GPU (H2D from NUMA0 DRAM)
#   2) remote-dataset prompt -> expect Mooncake load_get (GDR from NUMA1),
#                               and Offloading GPU→CPU ≈ 0 (max_offload_tokens=0)
#
# Same deploy as start_all.sh. Measure's round-robin mix cannot attribute a
# 10s KV dump to one dataset; this isolates each path onto its own dump.
#
# Usage:
#   ./run_measure_smoke.sh                 # same line idx as warmup default
#   ./run_measure_smoke.sh 3 5
#   ./run_measure_smoke.sh --a100-l40s
#   LOCAL_IDX=2 REMOTE_IDX=4 ./run_measure_smoke.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FILTERED=()
for arg in "$@"; do
  case "$arg" in
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s ;;
    *) FILTERED+=("$arg") ;;
  esac
done
set -- "${FILTERED[@]+"${FILTERED[@]}"}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

LOCAL_IDX="${1:-${LOCAL_IDX:-1}}"
REMOTE_IDX="${2:-${REMOTE_IDX:-1}}"
SMOKE_DIR="${LOG_DIR}/smoke_measure"
mkdir -p "$SMOKE_DIR"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets. Build them in test_mooncake:"
  echo "  cd $MOONCAKE_TEST_DIR && PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

# Same guard as run_measure.sh: Offloading must not re-cache GDR Gets.
EXTRA_BODY='{"kv_transfer_params":{"max_offload_tokens":0}}'

echo "Requesting sudo for netns ..."
sudo -v

echo "=== run_measure_smoke (sequential, local server only, GPU_PAIR=$GPU_PAIR) ==="
echo "  target:  GPU$LOCAL_GPU ($LOCAL_GPU_NAME) :$VLLM_PORT_LOCAL ($H1_NS)"
echo "  req 1:   local  dataset line=$LOCAL_IDX  -> expect Offloading CPU→GPU"
echo "  req 2:   remote dataset line=$REMOTE_IDX -> expect Mooncake load_get, no Offloading GPU→CPU"
echo "  guard:   $EXTRA_BODY"
warn_if_local_gpu_pcie_slow
echo "  out_len: $OUT_LEN"
echo "  logs:    $SMOKE_DIR/"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"

smoke_write_body "$DATASET_LOCAL" "$LOCAL_IDX" \
  "$SMOKE_DIR/local_body.json" "$SMOKE_DIR/local_meta.txt" "$EXTRA_BODY"
smoke_write_body "$DATASET_REMOTE" "$REMOTE_IDX" \
  "$SMOKE_DIR/remote_body.json" "$SMOKE_DIR/remote_meta.txt" "$EXTRA_BODY"

echo "Draining the previous KV-metrics window (~12s) so dumps stay isolated ..."
sleep 12

phase_mark start measure_smoke
ec=0

echo
echo "=== 1/2 local-half on local server (expect CPU DRAM → GPU) ==="
before="$(log_line_count "$LOG_DIR/vllm_local.log")"
if ! smoke_curl "local" "$H1_NS" "$VLLM_PORT_LOCAL" "$SMOKE_DIR/local_body.json" "$SMOKE_DIR"; then
  ec=1
fi
wait_kv_transfer_metrics "$LOG_DIR/vllm_local.log" "$before" 25 \
  "$SMOKE_DIR/local_kv_metrics.txt" || true
echo "--- KV dump after local-half ---"
if [[ -s "$SMOKE_DIR/local_kv_metrics.txt" ]]; then
  summarize_kv_metrics "$SMOKE_DIR/local_kv_metrics.txt" local_h2d || ec=1
else
  echo "  (missing — check $LOG_DIR/vllm_local.log)"
  ec=1
fi

echo
echo "=== 2/2 remote-half on local server (expect GDR, no Offloading store) ==="
before="$(log_line_count "$LOG_DIR/vllm_local.log")"
if ! smoke_curl "remote" "$H1_NS" "$VLLM_PORT_LOCAL" "$SMOKE_DIR/remote_body.json" "$SMOKE_DIR"; then
  ec=1
fi
wait_kv_transfer_metrics "$LOG_DIR/vllm_local.log" "$before" 25 \
  "$SMOKE_DIR/remote_kv_metrics.txt" || true
echo "--- KV dump after remote-half ---"
if [[ -s "$SMOKE_DIR/remote_kv_metrics.txt" ]]; then
  summarize_kv_metrics "$SMOKE_DIR/remote_kv_metrics.txt" remote_gdr || ec=1
else
  echo "  (missing — check $LOG_DIR/vllm_local.log)"
  ec=1
fi

phase_mark end measure_smoke

echo
echo "=== run_measure_smoke summary ==="
for tag in local remote; do
  if [[ -f "$SMOKE_DIR/${tag}_time.txt" ]]; then
    echo -n "  $tag: "
    tr '\n' ' ' <"$SMOKE_DIR/${tag}_time.txt"
    echo
  fi
done
echo "  dumps: $SMOKE_DIR/local_kv_metrics.txt"
echo "         $SMOKE_DIR/remote_kv_metrics.txt"
echo "  vLLM:  $LOG_DIR/vllm_local.log"

if [[ "$ec" != "0" ]]; then
  echo "FAIL: one request or KV-path check did not match."
  exit 1
fi
echo "PASS: local-half from CPU, remote-half from Mooncake GDR, no Offloading re-cache."
