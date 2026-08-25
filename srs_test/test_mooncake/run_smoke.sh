#!/bin/bash
# Curl smoke against local vLLM after warmup.
#
# Default: ONE local-dataset + ONE remote-dataset completion in parallel
# (CPU memcpy + RDMA mix). Pass local|remote to exercise a single path.
#
# After smoke_warmup.sh / warmup.sh (with segment pin), local-line KV should
# live in H1_SEGMENT (NUMA0) and remote-line KV in H2_SEGMENT (NUMA1).
#
# Usage:
#   ./run_smoke.sh                      # both, default GPU_PAIR=a100-l40s
#   ./run_smoke.sh --l40s-l40s local    # local-only on L40S GPU1
#   ./run_smoke.sh both                 # same as default both
#   ./run_smoke.sh 3 5                  # both, local line 3 + remote line 5
#   ./run_smoke.sh both 3 5
#   ./run_smoke.sh local
#   ./run_smoke.sh local 3
#   ./run_smoke.sh remote
#   ./run_smoke.sh remote 5
#   LOCAL_IDX=2 REMOTE_IDX=4 ./run_smoke.sh
#   MODE=local LOCAL_IDX=2 ./run_smoke.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${MODE:-both}"
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

if [[ $# -ge 1 ]]; then
  case "$1" in
    local|remote|both|all)
      MODE="$1"
      [[ "$MODE" == "all" ]] && MODE=both
      shift
      ;;
  esac
fi

LOCAL_IDX="${LOCAL_IDX:-1}"
REMOTE_IDX="${REMOTE_IDX:-1}"
case "$MODE" in
  both)
    LOCAL_IDX="${1:-$LOCAL_IDX}"
    REMOTE_IDX="${2:-$REMOTE_IDX}"
    ;;
  local)
    LOCAL_IDX="${1:-$LOCAL_IDX}"
    ;;
  remote)
    REMOTE_IDX="${1:-$REMOTE_IDX}"
    ;;
  *)
    echo "Usage: $0 [--l40s-l40s] [both|local|remote] [line_idx ...]  (see header)"
    exit 1
    ;;
esac

SMOKE_DIR="${LOG_DIR}/smoke_curl"
mkdir -p "$SMOKE_DIR"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets. Run:"
  echo "  PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v

echo "=== smoke (mode=$MODE, GPU_PAIR=$GPU_PAIR) ==="
echo "  target:   local vLLM GPU$LOCAL_GPU ($LOCAL_GPU_NAME) :$VLLM_PORT_LOCAL (netns $H1_NS)"
warn_if_local_gpu_pcie_slow
if [[ "$MODE" == "both" || "$MODE" == "local" ]]; then
  echo "  local:    $DATASET_LOCAL  line=$LOCAL_IDX  -> expect $H1_SEGMENT (NUMA0) / CPU memcpy"
fi
if [[ "$MODE" == "both" || "$MODE" == "remote" ]]; then
  echo "  remote:   $DATASET_REMOTE line=$REMOTE_IDX -> expect $H2_SEGMENT (NUMA1) / RDMA via $H1_RDMA"
fi
if [[ "$GPU_PAIR" == "l40s-l40s" ]]; then
  echo "  pex-mon:  -d $PEX_MON_DEVICE -p $PEX_MON_PORTS (112=gpu, 128=cpu on L40S switch)"
fi
echo "  out_len:  $OUT_LEN"
echo "  logs:     $SMOKE_DIR/"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"

write_body() {
  local dataset="$1" idx="$2" out_body="$3" out_meta="$4"
  python3 - "$dataset" "$idx" "$MODEL" "$OUT_LEN" "$out_body" "$out_meta" <<'PY'
import json, sys
dataset, idx_s, model, max_tokens, out_body, out_meta = sys.argv[1:7]
idx = int(idx_s)
prompt = None
with open(dataset, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i == idx:
            obj = json.loads(line)
            prompt = obj["prompt"]
            break
if prompt is None:
    raise SystemExit(f"ERROR: line {idx} not found in {dataset}")
body = {
    "model": model,
    "prompt": prompt,
    "max_tokens": int(max_tokens),
    "temperature": 0,
    "ignore_eos": True,
}
with open(out_body, "w", encoding="utf-8") as f:
    json.dump(body, f)
with open(out_meta, "w", encoding="utf-8") as f:
    f.write(f"dataset={dataset}\n")
    f.write(f"line={idx}\n")
    f.write(f"prompt_chars={len(prompt)}\n")
    f.write(f"max_tokens={max_tokens}\n")
print(f"wrote {out_body} (prompt_chars={len(prompt)})")
PY
}

curl_one() {
  local tag="$1" body="$2"
  local resp="$SMOKE_DIR/${tag}_resp.json"
  local http="$SMOKE_DIR/${tag}_http.txt"
  local timing="$SMOKE_DIR/${tag}_time.txt"
  # Run inside h1 so we hit the netns-local vLLM :8000.
  # -w writes HTTP code + wall time after the body.
  netns_run "$H1_NS" curl -sS \
    -X POST "http://127.0.0.1:${VLLM_PORT_LOCAL}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary @"$body" \
    -o "$resp" \
    -w "http_code=%{http_code}\ntime_total_s=%{time_total}\n" \
    >"$http" 2>"$SMOKE_DIR/${tag}_curl.err" || true
  # Merge curl -w into timing for easy reading.
  cat "$http" >"$timing"
  local code
  code="$(sed -n 's/^http_code=//p' "$timing" | head -1)"
  local secs
  secs="$(sed -n 's/^time_total_s=//p' "$timing" | head -1)"
  if [[ "$code" == "200" ]]; then
    echo "[$tag] OK http=$code time=${secs}s -> $resp"
  else
    echo "[$tag] FAIL http=${code:-?} time=${secs:-?} (see $SMOKE_DIR/${tag}_*)"
    if [[ -s "$SMOKE_DIR/${tag}_curl.err" ]]; then
      sed -n '1,5p' "$SMOKE_DIR/${tag}_curl.err"
    fi
    if [[ -s "$resp" ]]; then
      python3 -c "import pathlib; t=pathlib.Path('$resp').read_text(errors='replace'); print(t[:400])"
    fi
  fi
}

TAGS=()
case "$MODE" in
  both)
    LOCAL_BODY="$SMOKE_DIR/local_body.json"
    REMOTE_BODY="$SMOKE_DIR/remote_body.json"
    write_body "$DATASET_LOCAL" "$LOCAL_IDX" "$LOCAL_BODY" "$SMOKE_DIR/local_meta.txt"
    write_body "$DATASET_REMOTE" "$REMOTE_IDX" "$REMOTE_BODY" "$SMOKE_DIR/remote_meta.txt"
    echo "Launching two curls in parallel ..."
    curl_one "local" "$LOCAL_BODY" &
    pid_local=$!
    curl_one "remote" "$REMOTE_BODY" &
    pid_remote=$!
    ec=0
    wait "$pid_local" || ec=1
    wait "$pid_remote" || ec=1
    TAGS=(local remote)
    ;;
  local)
    LOCAL_BODY="$SMOKE_DIR/local_body.json"
    write_body "$DATASET_LOCAL" "$LOCAL_IDX" "$LOCAL_BODY" "$SMOKE_DIR/local_meta.txt"
    echo "Launching local curl only ..."
    curl_one "local" "$LOCAL_BODY"
    ec=0
    TAGS=(local)
    ;;
  remote)
    REMOTE_BODY="$SMOKE_DIR/remote_body.json"
    write_body "$DATASET_REMOTE" "$REMOTE_IDX" "$REMOTE_BODY" "$SMOKE_DIR/remote_meta.txt"
    echo "Launching remote curl only ..."
    curl_one "remote" "$REMOTE_BODY"
    ec=0
    TAGS=(remote)
    ;;
esac

echo
echo "=== smoke summary (mode=$MODE) ==="
for tag in "${TAGS[@]}"; do
  if [[ -f "$SMOKE_DIR/${tag}_time.txt" ]]; then
    echo -n "  $tag: "
    tr '\n' ' ' <"$SMOKE_DIR/${tag}_time.txt"
    echo
  fi
done

if [[ "$ec" != "0" ]]; then
  echo "ERROR: one or more curls failed"
  exit 1
fi

for tag in "${TAGS[@]}"; do
  code="$(sed -n 's/^http_code=//p' "$SMOKE_DIR/${tag}_time.txt" | head -1)"
  if [[ "$code" != "200" ]]; then
    echo "ERROR: expected HTTP 200 from $tag (got ${code:-?})"
    exit 1
  fi
done

if [[ "$MODE" == "both" ]]; then
  echo "Both OK. Watch vLLM local log (Running/Waiting) and RNIC/pex while this ran."
else
  echo "OK ($MODE). Watch vLLM local log and RNIC/pex while this ran."
fi
echo "  tail -f $LOG_DIR/vllm_local.log"
echo "  rg 'load_get|External prefix' $LOG_DIR/vllm_local.log | tail"
