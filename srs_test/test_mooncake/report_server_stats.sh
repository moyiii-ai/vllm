#!/bin/bash
# Parse vLLM Engine logs for peak Running / Waiting, split by warmup vs measure.
# Usage:
#   ./report_server_stats.sh              # local + remote under LOG_DIR
#   ./report_server_stats.sh path/to.log  # one or more explicit log files (no phase split)
#
# Phase windows (first match wins):
#   1) logs/phases.jsonl  written by warmup.sh / run_measure.sh
#   2) inferred from latest logs/warmup_*/ and logs/measure_*/ bench JSONs
#      (end=date field, start=end-duration)
#
# Note: vLLM samples ~every 10s (and skips INFO when idle), so brief peaks
# between samples can be missed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"

python3 - "$LOG_DIR" "$@" <<'PY'
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = Path(sys.argv[1])
EXTRA_LOGS = sys.argv[2:]

ENGINE_PAT = re.compile(
    r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs"
    r"(?:.*?GPU KV cache usage:\s*([\d.]+)%)?"
    r"(?:.*?External prefix cache hit rate:\s*([\d.]+)%)?",
)
TS_PAT = re.compile(r"INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


@dataclass
class Phase:
    name: str
    start: datetime
    end: datetime
    source: str = ""


@dataclass
class Acc:
    n: int = 0
    max_run: int = -1
    max_wait: int = -1
    sum_run: int = 0
    sum_wait: int = 0
    max_kv: float = -1.0
    last_ext: float | None = None
    peak_run_ts: str = ""
    peak_wait_ts: str = ""

    def add(self, running: int, waiting: int, kv: float | None, ext: float | None, ts: str) -> None:
        self.n += 1
        self.sum_run += running
        self.sum_wait += waiting
        if running > self.max_run:
            self.max_run, self.peak_run_ts = running, ts
        if waiting > self.max_wait:
            self.max_wait, self.peak_wait_ts = waiting, ts
        if kv is not None and kv > self.max_kv:
            self.max_kv = kv
        if ext is not None:
            self.last_ext = ext


def parse_log_ts(s: str, year_hint: int) -> datetime | None:
    try:
        return datetime.strptime(f"{year_hint}-{s}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_bench_date(s: str) -> datetime:
    # 20260804-035055
    return datetime.strptime(s, "%Y%m%d-%H%M%S")


def latest_bench_json(subdir: Path) -> Path | None:
    if not subdir.is_dir():
        return None
    best = None
    best_dt = None
    for p in subdir.glob("vllm-*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            dt = parse_bench_date(d["date"])
        except Exception:
            continue
        if best_dt is None or dt > best_dt:
            best, best_dt = p, dt
    return best


def phase_from_bench(name: str, *subdirs: Path) -> Phase | None:
    windows: list[tuple[datetime, datetime, str]] = []
    for sd in subdirs:
        p = latest_bench_json(sd)
        if p is None:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        end = parse_bench_date(d["date"])
        dur = float(d.get("duration") or 0.0)
        # small pad: engine samples are ~10s and bench date is end-of-run
        start = end - timedelta(seconds=dur + 15)
        end = end + timedelta(seconds=15)
        windows.append((start, end, str(p.name)))
    if not windows:
        return None
    start = min(w[0] for w in windows)
    end = max(w[1] for w in windows)
    src = ", ".join(w[2] for w in windows)
    return Phase(name=name, start=start, end=end, source=f"bench:{src}")


def load_phases(log_dir: Path) -> list[Phase]:
    marker = log_dir / "phases.jsonl"
    if marker.is_file():
        starts: dict[str, datetime] = {}
        ends: dict[str, datetime] = {}
        src = f"phases.jsonl"
        for line in marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = o.get("phase")
            event = o.get("event")
            ts_s = o.get("ts")  # "YYYY-MM-DD HH:MM:SS"
            if not phase or not event or not ts_s:
                continue
            try:
                ts = datetime.strptime(ts_s, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if event == "start":
                starts[phase] = ts
            elif event == "end":
                ends[phase] = ts
        out: list[Phase] = []
        for name in ("warmup", "measure"):
            if name in starts and name in ends and ends[name] >= starts[name]:
                out.append(Phase(name, starts[name], ends[name], src))
            elif name in starts and name not in ends:
                # still running / crashed before end mark
                out.append(Phase(name, starts[name], datetime.now(), src + " (open end)"))
        if out:
            return out

    # Fallback: infer from bench result JSONs
    phases: list[Phase] = []
    w = phase_from_bench("warmup", log_dir / "warmup_local", log_dir / "warmup_remote")
    m = phase_from_bench("measure", log_dir / "measure_local", log_dir / "measure_remote")
    if w:
        phases.append(w)
    if m:
        phases.append(m)
    return phases


def pick_phase(ts: datetime, phases: list[Phase]) -> str:
    for p in phases:
        if p.start <= ts <= p.end:
            return p.name
    return "other"


def print_acc(title: str, acc: Acc) -> None:
    print(f"  -- {title} --")
    if acc.n == 0:
        print("     no Engine samples in this window")
        return
    print(f"     samples:                  {acc.n}")
    print(
        f"     max Running (concurrent):  {acc.max_run}"
        + (f"  @ {acc.peak_run_ts}" if acc.peak_run_ts else "")
    )
    print(
        f"     max Waiting (queued):      {acc.max_wait}"
        + (f"  @ {acc.peak_wait_ts}" if acc.peak_wait_ts else "")
    )
    print(f"     avg Running (over samples): {acc.sum_run / acc.n:.2f}")
    print(f"     avg Waiting (over samples): {acc.sum_wait / acc.n:.2f}")
    if acc.max_kv >= 0:
        print(f"     max GPU KV cache usage:    {acc.max_kv:.1f}%")
    if acc.last_ext is not None:
        print(f"     last Ext prefix hit rate:  {acc.last_ext:.1f}%")


def analyze(label: str, path: Path, phases: list[Phase], split: bool) -> None:
    print(f"=== {label} ({path}) ===")
    if not path.is_file():
        print("  (log missing)")
        print()
        return

    year_hint = datetime.now().year
    if phases:
        year_hint = phases[0].start.year

    buckets: dict[str, Acc] = {
        "warmup": Acc(),
        "measure": Acc(),
        "other": Acc(),
        "all": Acc(),
    }

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ENGINE_PAT.search(line)
            if not m:
                continue
            running = int(m.group(1))
            waiting = int(m.group(2))
            kv = float(m.group(3)) if m.group(3) is not None else None
            ext = float(m.group(4)) if m.group(4) is not None else None
            ts_m = TS_PAT.search(line)
            ts_s = ts_m.group(1) if ts_m else ""
            ts = parse_log_ts(ts_s, year_hint) if ts_s else None
            buckets["all"].add(running, waiting, kv, ext, ts_s)
            if split and phases and ts is not None:
                buckets[pick_phase(ts, phases)].add(running, waiting, kv, ext, ts_s)
            elif split and phases:
                buckets["other"].add(running, waiting, kv, ext, ts_s)

    if buckets["all"].n == 0:
        print("  no Engine Running/Waiting samples found")
        print()
        return

    if split and phases:
        for name in ("warmup", "measure"):
            print_acc(name, buckets[name])
        if buckets["other"].n:
            print_acc("other (outside phase windows)", buckets["other"])
    else:
        print_acc("all samples", buckets["all"])
    print()


def main() -> None:
    phases = load_phases(LOG_DIR)
    print("=== vLLM queue stats (from Engine logs) ===")
    print("Note: ~10s sample interval; short spikes between samples may be missed.")
    if phases:
        print("Phases:")
        for p in phases:
            print(
                f"  {p.name:8s}  {p.start:%Y-%m-%d %H:%M:%S} .. {p.end:%Y-%m-%d %H:%M:%S}"
                f"  [{p.source}]"
            )
    else:
        print("Phases: (none found — showing unsplit totals)")
        print("  Tip: run ./warmup.sh / ./run_measure.sh (writes phases.jsonl),")
        print("       or keep bench JSONs under logs/warmup_* and logs/measure_*.")
    print()

    if EXTRA_LOGS:
        for i, log in enumerate(EXTRA_LOGS, 1):
            analyze(f"log#{i}", Path(log), phases, split=False)
        return

    analyze("local", LOG_DIR / "vllm_local.log", phases, split=True)
    analyze("remote", LOG_DIR / "vllm_remote.log", phases, split=True)


main()
PY
