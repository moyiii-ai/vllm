#!/bin/bash
# Prime one prompt per server (same MultiConnector deploy as start_all.sh).
#
#   local  dataset line -> local  vLLM  (Offloading + Mooncake Put to H1_SEGMENT)
#   remote dataset line -> remote vLLM  (Offloading + Mooncake Put to H2_SEGMENT)
#
# Do NOT set max_offload_tokens here — both connectors must store.
# Sequential so each side's KV Transfer dump is easy to read.
#
# Usage:
#   ./warmup_smoke.sh                 # line 1 + line 1
#   ./warmup_smoke.sh 3 5             # local line 3, remote line 5 (1-based)
#   ./warmup_smoke.sh --a100-l40s
#   LOCAL_IDX=2 REMOTE_IDX=4 ./warmup_smoke.sh
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
WARM_DIR="${LOG_DIR}/smoke_warmup"
mkdir -p "$WARM_DIR"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets. Build them in test_mooncake:"
  echo "  cd $MOONCAKE_TEST_DIR && PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v

echo "=== warmup_smoke (one curl per server, GPU_PAIR=$GPU_PAIR) ==="
echo "  local : $DATASET_LOCAL  line=$LOCAL_IDX -> :$VLLM_PORT_LOCAL ($H1_NS, GPU$LOCAL_GPU $LOCAL_GPU_NAME / $H1_SEGMENT)"
echo "  remote: $DATASET_REMOTE line=$REMOTE_IDX -> :$VLLM_PORT_REMOTE ($H2_NS, GPU$REMOTE_GPU $REMOTE_GPU_NAME / $H2_SEGMENT)"
echo "  expect: Offloading GPU→CPU + Mooncake save_put on each side (no max_offload_tokens)"
warn_if_local_gpu_pcie_slow
warn_if_remote_gpu_pcie_slow
echo "  out_len: $OUT_LEN"
echo "  logs:    $WARM_DIR/"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"

smoke_write_body "$DATASET_LOCAL" "$LOCAL_IDX" "$WARM_DIR/local_body.json" "$WARM_DIR/local_meta.txt"
smoke_write_body "$DATASET_REMOTE" "$REMOTE_IDX" "$WARM_DIR/remote_body.json" "$WARM_DIR/remote_meta.txt"

phase_mark start warmup_smoke

local_before="$(log_line_count "$LOG_DIR/vllm_local.log")"
echo "Warming local (GPU$LOCAL_GPU) ..."
smoke_curl "local" "$H1_NS" "$VLLM_PORT_LOCAL" "$WARM_DIR/local_body.json" "$WARM_DIR"
wait_kv_transfer_metrics "$LOG_DIR/vllm_local.log" "$local_before" 25 \
  "$WARM_DIR/local_kv_metrics.txt" || true
echo "--- local server KV dump ---"
if [[ -s "$WARM_DIR/local_kv_metrics.txt" ]]; then
  summarize_kv_metrics "$WARM_DIR/local_kv_metrics.txt" warmup_save || true
else
  echo "  (missing — check $LOG_DIR/vllm_local.log)"
fi

remote_before="$(log_line_count "$LOG_DIR/vllm_remote.log")"
echo
echo "Warming remote (GPU$REMOTE_GPU) ..."
smoke_curl "remote" "$H2_NS" "$VLLM_PORT_REMOTE" "$WARM_DIR/remote_body.json" "$WARM_DIR"
wait_kv_transfer_metrics "$LOG_DIR/vllm_remote.log" "$remote_before" 25 \
  "$WARM_DIR/remote_kv_metrics.txt" || true
echo "--- remote server KV dump ---"
if [[ -s "$WARM_DIR/remote_kv_metrics.txt" ]]; then
  summarize_kv_metrics "$WARM_DIR/remote_kv_metrics.txt" warmup_save || true
else
  echo "  (missing — check $LOG_DIR/vllm_remote.log)"
fi

phase_mark end warmup_smoke

echo
echo "=== warmup_smoke summary ==="
ec=0
for tag in local remote; do
  if [[ -f "$WARM_DIR/${tag}_time.txt" ]]; then
    echo -n "  $tag: "
    tr '\n' ' ' <"$WARM_DIR/${tag}_time.txt"
    echo
  fi
  code="$(sed -n 's/^http_code=//p' "$WARM_DIR/${tag}_time.txt" | head -1)"
  if [[ "$code" != "200" ]]; then
    echo "ERROR: expected HTTP 200 from $tag (got ${code:-?})"
    ec=1
  fi
done
if [[ "$ec" != "0" ]]; then
  exit 1
fi

echo "Both OK. Local-line KV should sit in Offloading+Mooncake $H1_SEGMENT;"
echo "remote-line KV in Offloading+Mooncake $H2_SEGMENT."
echo "Next: ./run_measure_smoke.sh $LOCAL_IDX $REMOTE_IDX"
