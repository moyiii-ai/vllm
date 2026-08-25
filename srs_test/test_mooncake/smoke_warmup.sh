#!/bin/bash
# Minimal warmup for run_smoke.sh: curl ONE local-dataset line to local vLLM
# and ONE remote-dataset line to remote vLLM so KV lands in each side's DRAM.
#
# Unlike warmup.sh (full split via vllm bench), this only primes the prompts
# that run_smoke.sh will later re-request concurrently on the local GPU.
#
# Usage:
#   ./smoke_warmup.sh                     # same defaults as run_smoke (line 1 + 1)
#   ./smoke_warmup.sh --l40s-l40s         # warmup against L40S+L40S stack
#   ./smoke_warmup.sh 3 5                 # local line 3 + remote line 5 (1-based)
#   LOCAL_IDX=2 REMOTE_IDX=4 ./smoke_warmup.sh
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
WARM_DIR="${LOG_DIR}/smoke_warmup_curl"
mkdir -p "$WARM_DIR"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets. Run:"
  echo "  PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v

echo "=== smoke_warmup (curl per side, GPU_PAIR=$GPU_PAIR) ==="
echo "  local : $DATASET_LOCAL  line=$LOCAL_IDX -> :$VLLM_PORT_LOCAL ($H1_NS, GPU$LOCAL_GPU $LOCAL_GPU_NAME / NUMA0 DRAM)"
echo "  remote: $DATASET_REMOTE line=$REMOTE_IDX -> :$VLLM_PORT_REMOTE ($H2_NS, GPU$REMOTE_GPU $REMOTE_GPU_NAME / NUMA1 DRAM)"
warn_if_local_gpu_pcie_slow
echo "  out_len: $OUT_LEN"
echo "  logs:    $WARM_DIR/"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"

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
  local tag="$1" ns="$2" port="$3" body="$4"
  local resp="$WARM_DIR/${tag}_resp.json"
  local http="$WARM_DIR/${tag}_http.txt"
  local timing="$WARM_DIR/${tag}_time.txt"
  netns_run "$ns" curl -sS \
    -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H "Content-Type: application/json" \
    --data-binary @"$body" \
    -o "$resp" \
    -w "http_code=%{http_code}\ntime_total_s=%{time_total}\n" \
    >"$http" 2>"$WARM_DIR/${tag}_curl.err" || true
  cat "$http" >"$timing"
  local code secs
  code="$(sed -n 's/^http_code=//p' "$timing" | head -1)"
  secs="$(sed -n 's/^time_total_s=//p' "$timing" | head -1)"
  if [[ "$code" == "200" ]]; then
    echo "[$tag] OK http=$code time=${secs}s ns=$ns :$port -> $resp"
  else
    echo "[$tag] FAIL http=${code:-?} time=${secs:-?} (see $WARM_DIR/${tag}_*)"
    if [[ -s "$WARM_DIR/${tag}_curl.err" ]]; then
      sed -n '1,5p' "$WARM_DIR/${tag}_curl.err"
    fi
    if [[ -s "$resp" ]]; then
      python3 -c "import pathlib; t=pathlib.Path('$resp').read_text(errors='replace'); print(t[:400])"
    fi
  fi
}

LOCAL_BODY="$WARM_DIR/local_body.json"
REMOTE_BODY="$WARM_DIR/remote_body.json"
write_body "$DATASET_LOCAL" "$LOCAL_IDX" "$LOCAL_BODY" "$WARM_DIR/local_meta.txt"
write_body "$DATASET_REMOTE" "$REMOTE_IDX" "$REMOTE_BODY" "$WARM_DIR/remote_meta.txt"

# Sequential: each side puts into its own embedded segment without contending.
echo "Warming local (GPU$LOCAL_GPU) ..."
curl_one "local" "$H1_NS" "$VLLM_PORT_LOCAL" "$LOCAL_BODY"
echo "Warming remote (GPU$REMOTE_GPU) ..."
curl_one "remote" "$H2_NS" "$VLLM_PORT_REMOTE" "$REMOTE_BODY"

echo
echo "=== smoke_warmup summary ==="
for tag in local remote; do
  if [[ -f "$WARM_DIR/${tag}_time.txt" ]]; then
    echo -n "  $tag: "
    tr '\n' ' ' <"$WARM_DIR/${tag}_time.txt"
    echo
  fi
done

local_code="$(sed -n 's/^http_code=//p' "$WARM_DIR/local_time.txt" | head -1)"
remote_code="$(sed -n 's/^http_code=//p' "$WARM_DIR/remote_time.txt" | head -1)"
if [[ "$local_code" != "200" || "$remote_code" != "200" ]]; then
  echo "ERROR: expected HTTP 200 from both (local=$local_code remote=$remote_code)"
  exit 1
fi

echo "Both OK. With segment pin, local-line KV should sit in $H1_SEGMENT (NUMA0)"
echo "and remote-line KV in $H2_SEGMENT (NUMA1). Confirm logs show matching"
echo "preferred_segment=... and segment_name=..."
echo "Next: ./run_smoke.sh $LOCAL_IDX $REMOTE_IDX"
echo "  rg 'Mem Storage|Keys:|save_put|load_get' $LOG_DIR/master.log $LOG_DIR/vllm_local.log $LOG_DIR/vllm_remote.log | tail"
