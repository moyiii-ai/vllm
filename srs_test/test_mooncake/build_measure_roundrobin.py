#!/usr/bin/env python3
"""Build a measure dataset by round-robin interleaving local/remote split halves.

After prefer-segment warmup, narrativeqa_local KV lives on H1_SEGMENT (NUMA0)
and narrativeqa_remote KV on H2_SEGMENT (NUMA1). Interleaving so measure traffic
to the local vLLM alternates DRAM memcpy hits and RDMA Gets.

Order: L0, R0, L1, R1, ... until one side is exhausted, then append the rest.

Extra fields (kv_side, src_idx, measure_idx) are ignored by vllm bench custom
dataset loading as long as `prompt` is present.

Example:
  python build_measure_roundrobin.py
  python build_measure_roundrobin.py \\
      --local narrativeqa_local.jsonl \\
      --remote narrativeqa_remote.jsonl \\
      --out narrativeqa_measure_rr.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--local",
        type=Path,
        default=here / "narrativeqa_local.jsonl",
        help="Warmup/local-segment prompts (default: narrativeqa_local.jsonl).",
    )
    p.add_argument(
        "--remote",
        type=Path,
        default=here / "narrativeqa_remote.jsonl",
        help="Warmup/remote-segment prompts (default: narrativeqa_remote.jsonl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=here / "narrativeqa_measure_rr.jsonl",
        help="Round-robin output (default: narrativeqa_measure_rr.jsonl).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=here / "narrativeqa_measure_rr_report.json",
        help="Small JSON summary next to the dataset.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise SystemExit(f"{path}:{i}: missing 'prompt'")
            rows.append(obj)
    return rows


def interleave(
    local: list[dict[str, Any]], remote: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = max(len(local), len(remote))
    for i in range(n):
        if i < len(local):
            row = dict(local[i])
            row["kv_side"] = "local"
            row["src_idx"] = i + 1  # 1-based line in source file
            row["measure_idx"] = len(out) + 1
            out.append(row)
        if i < len(remote):
            row = dict(remote[i])
            row["kv_side"] = "remote"
            row["src_idx"] = i + 1
            row["measure_idx"] = len(out) + 1
            out.append(row)
    return out


def main() -> None:
    args = parse_args()
    if not args.local.is_file():
        raise SystemExit(f"missing --local {args.local}")
    if not args.remote.is_file():
        raise SystemExit(f"missing --remote {args.remote}")

    local = load_jsonl(args.local)
    remote = load_jsonl(args.remote)
    rows = interleave(local, remote)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # First 2*min(nL,nR) lines are strict L/R pairs; remainder is one side.
    n_pair = min(len(local), len(remote))
    report = {
        "local": str(args.local.resolve()),
        "remote": str(args.remote.resolve()),
        "out": str(args.out.resolve()),
        "num_local": len(local),
        "num_remote": len(remote),
        "num_out": len(rows),
        "num_strict_pairs": n_pair,
        "tail_side": (
            "local"
            if len(local) > len(remote)
            else ("remote" if len(remote) > len(local) else None)
        ),
        "tail_count": abs(len(local) - len(remote)),
        "pattern": "L0,R0,L1,R1,..., then remaining longer side",
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"wrote {args.out}  ({len(rows)} prompts)")
    print(
        f"  local={len(local)} remote={len(remote)} "
        f"strict_pairs={n_pair} tail={report['tail_side']}x{report['tail_count']}"
    )
    print(f"  report {args.report}")


if __name__ == "__main__":
    main()
