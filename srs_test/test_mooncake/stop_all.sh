#!/bin/bash
# Stop Mooncake master + both vLLM servers.
# After stop, prints peak Running/Waiting from logs (SKIP_REPORT=1 to skip).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"

stop_pidfile() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local pid
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping pid $pid ($f)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  fi
}

mkdir -p "$LOG_DIR"
for name in \
  vllm_local_tail vllm_remote_tail store_tail \
  vllm_local vllm_remote vllm \
  store master metadata
do
  stop_pidfile "$LOG_DIR/${name}.pid"
done

for ns in "$H1_NS" "$H2_NS"; do
  if ip netns list 2>/dev/null | awk '{print $1}' | grep -qx "$ns"; then
    sudo ip netns pids "$ns" 2>/dev/null | while read -r p; do
      cmd="$(ps -o args= -p "$p" 2>/dev/null || true)"
      if echo "$cmd" | grep -Eq 'mooncake_|vllm'; then
        echo "Killing $p in $ns: $cmd"
        sudo kill -9 "$p" 2>/dev/null || true
      fi
    done
  fi
done

pkill -f "tail -n \+1 -F $LOG_DIR/" 2>/dev/null || true
echo "Stopped."

# Analyze logs before the next start_server truncates them.
case "${SKIP_REPORT:-0}" in
  1|true|yes|on) ;;
  *)
    if [[ -x "$SCRIPT_DIR/report_server_stats.sh" ]] || [[ -f "$SCRIPT_DIR/report_server_stats.sh" ]]; then
      echo
      bash "$SCRIPT_DIR/report_server_stats.sh" || true
    fi
    ;;
esac
