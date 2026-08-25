#!/usr/bin/env python3
"""Split NarrativeQA into two balanced halves with no story-level KV-block overlap.

NarrativeQA prompts share a short instruction, then a long Story, then a Question.
Many questions reuse the same Story. Random/request-count splits put the same
story on both sides, so Mooncake block hashes collide and warmup placement breaks.

This script:
  1) Groups prompts by Story (text before '\\n\\nQuestion:')
  2) Assigns each story group wholly to part A or B (greedy balance by #tokens)
  3) Verifies vLLM-style chained block hashes have no overlap beyond the shared
     instruction-only prefix blocks (which are unavoidable)

Example:
  PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py \\
      --input ../narrativeqa.jsonl \\
      --out-a narrativeqa_local.jsonl \\
      --out-b narrativeqa_remote.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


QUESTION_SPLIT = re.compile(r"\n\nQuestion:\s*", re.IGNORECASE)


@dataclass
class PromptRow:
    idx: int
    obj: dict[str, Any]
    prompt: str
    story_key: str
    n_tokens: int
    block_hashes: list[bytes]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "narrativeqa.jsonl",
    )
    p.add_argument(
        "--out-a",
        type=Path,
        default=Path(__file__).resolve().parent / "narrativeqa_local.jsonl",
    )
    p.add_argument(
        "--out-b",
        type=Path,
        default=Path(__file__).resolve().parent / "narrativeqa_remote.jsonl",
    )
    p.add_argument(
        "--model",
        default="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        help="Tokenizer / model id (or local snapshot path).",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="vLLM hash/scheduler block size (default 16).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "narrativeqa_split_report.json",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Only used to break ties when token totals are equal.",
    )
    return p.parse_args()


def resolve_model_path(model: str) -> str:
    """Prefer local HF snapshot when present (avoids re-download / hub auth)."""
    if Path(model).is_dir() and (Path(model) / "config.json").is_file():
        return model
    hub = Path.home() / ".cache/huggingface/hub"
    candidates = [
        hub / "models--RedHatAI--Meta-Llama-3.1-8B-Instruct-FP8-dynamic" / "snapshots",
        hub / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots",
    ]
    for snap_root in candidates:
        if "Llama-3.1-8B-Instruct" in model and snap_root.is_dir():
            kids = sorted(snap_root.iterdir())
            if kids:
                return str(kids[0])
    return model


def story_key(prompt: str) -> str:
    m = QUESTION_SPLIT.search(prompt)
    if not m:
        raise ValueError("prompt missing '\\n\\nQuestion:' separator")
    return prompt[: m.start()]


def global_common_prefix(texts: Sequence[str]) -> str:
    if not texts:
        return ""
    pref = texts[0]
    for t in texts[1:]:
        n = 0
        limit = min(len(pref), len(t))
        while n < limit and pref[n] == t[n]:
            n += 1
        pref = pref[:n]
        if not pref:
            break
    return pref


def load_hash_fn() -> Callable[[Any], bytes]:
    try:
        from vllm.utils.hashing import sha256 as vllm_sha256

        return vllm_sha256
    except ImportError:
        import hashlib
        import pickle

        def _sha256(obj: Any) -> bytes:
            return hashlib.sha256(
                pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            ).digest()

        return _sha256


def init_none_hash(hash_fn: Callable[[Any], bytes]) -> bytes:
    # Match vLLM: PYTHONHASHSEED set -> hash_fn(seed); else random (bad for us).
    seed = os.environ.get("PYTHONHASHSEED")
    if seed is None:
        print(
            "WARNING: PYTHONHASHSEED unset; set PYTHONHASHSEED=0 to match vLLM serve.",
            file=sys.stderr,
        )
        seed = "0"
        os.environ["PYTHONHASHSEED"] = seed
    return hash_fn(seed)


def hash_block_tokens(
    hash_fn: Callable[[Any], bytes],
    parent: bytes | None,
    token_ids: Sequence[int],
    none_hash: bytes,
) -> bytes:
    parent_h = none_hash if parent is None else parent
    return hash_fn((parent_h, tuple(token_ids), None))


def token_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    hash_fn: Callable[[Any], bytes],
    none_hash: bytes,
) -> list[bytes]:
    hashes: list[bytes] = []
    parent: bytes | None = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block = token_ids[start : start + block_size]
        h = hash_block_tokens(hash_fn, parent, block, none_hash)
        hashes.append(h)
        parent = h
    return hashes


def greedy_partition(
    group_tokens: dict[str, int],
    seed: int,
) -> tuple[set[str], set[str]]:
    """Assign each story to the currently lighter side (by token count)."""
    import random

    rng = random.Random(seed)
    items = sorted(group_tokens.items(), key=lambda kv: (-kv[1], kv[0]))
    # Shuffle equal-size ties stably via seed on equal token counts.
    # Already secondary-sorted by key; for equal tokens, rng.shuffle within buckets.
    buckets: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for k, t in items:
        buckets[t].append((k, t))
    ordered: list[tuple[str, int]] = []
    for t in sorted(buckets.keys(), reverse=True):
        bucket = buckets[t]
        rng.shuffle(bucket)
        ordered.extend(bucket)

    a: set[str] = set()
    b: set[str] = set()
    ta = tb = 0
    for key, tok in ordered:
        if ta <= tb:
            a.add(key)
            ta += tok
        else:
            b.add(key)
            tb += tok
    return a, b


def write_jsonl(path: Path, rows: Iterable[PromptRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.obj, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    hash_fn = load_hash_fn()
    none_hash = init_none_hash(hash_fn)
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} block_size={args.block_size}")

    raw: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    if not raw:
        print("ERROR: empty input", file=sys.stderr)
        return 1

    prompts = [r["prompt"] for r in raw]
    instr = global_common_prefix(prompts)
    instr_tokens = tok.encode(instr, add_special_tokens=True)
    # Full blocks covered solely by the shared instruction.
    n_shared_full_blocks = len(instr_tokens) // args.block_size
    shared_prefix_hashes = set(
        token_block_hashes(instr_tokens, args.block_size, hash_fn, none_hash)
    )

    print(
        f"Loaded {len(raw)} prompts; shared instruction "
        f"{len(instr)} chars / {len(instr_tokens)} toks "
        f"-> {n_shared_full_blocks} unavoidable shared full blocks"
    )

    rows: list[PromptRow] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for i, obj in enumerate(raw):
        prompt = obj["prompt"]
        sk = story_key(prompt)
        ids = tok.encode(prompt, add_special_tokens=True)
        hashes = token_block_hashes(ids, args.block_size, hash_fn, none_hash)
        row = PromptRow(
            idx=i,
            obj=obj,
            prompt=prompt,
            story_key=sk,
            n_tokens=len(ids),
            block_hashes=hashes,
        )
        rows.append(row)
        groups[sk].append(i)

    group_tokens = {
        sk: sum(rows[i].n_tokens for i in idxs) for sk, idxs in groups.items()
    }
    print(f"Story groups: {len(groups)}")

    set_a, set_b = greedy_partition(group_tokens, args.seed)
    rows_a = [rows[i] for sk in set_a for i in groups[sk]]
    rows_b = [rows[i] for sk in set_b for i in groups[sk]]
    # Stable order within each part: original file order
    rows_a.sort(key=lambda r: r.idx)
    rows_b.sort(key=lambda r: r.idx)

    tok_a = sum(r.n_tokens for r in rows_a)
    tok_b = sum(r.n_tokens for r in rows_b)
    imbalance = abs(tok_a - tok_b) / max(tok_a + tok_b, 1)

    hashes_a = {h for r in rows_a for h in r.block_hashes}
    hashes_b = {h for r in rows_b for h in r.block_hashes}
    overlap_all = hashes_a & hashes_b
    # Overlap that is NOT explained by the shared instruction-only blocks.
    # (Chained hashes after the story diverges should be unique per story.)
    content_overlap = overlap_all - shared_prefix_hashes

    # Extra check: any story key in both sets?
    story_overlap = set_a & set_b

    print("\n=== split summary ===")
    print(f"  A: {len(rows_a)} prompts, {len(set_a)} stories, {tok_a} tokens")
    print(f"  B: {len(rows_b)} prompts, {len(set_b)} stories, {tok_b} tokens")
    print(f"  token imbalance: {imbalance:.4%} (|A-B|/(A+B))")
    print(f"  unique block hashes A/B: {len(hashes_a)} / {len(hashes_b)}")
    print(f"  raw hash overlap: {len(overlap_all)}")
    print(f"  instruction-only shared hashes: {len(shared_prefix_hashes)}")
    print(f"  content hash overlap (should be 0): {len(content_overlap)}")
    print(f"  story-key overlap (should be 0): {len(story_overlap)}")

    ok = (not story_overlap) and (not content_overlap)
    if not ok:
        print("ERROR: residual content/story overlap; refusing to write outputs.", file=sys.stderr)
        if content_overlap:
            print(f"  sample content-overlap hashes: {list(content_overlap)[:3]!r}", file=sys.stderr)
        return 2

    write_jsonl(args.out_a, rows_a)
    write_jsonl(args.out_b, rows_b)

    report = {
        "input": str(args.input),
        "out_a": str(args.out_a),
        "out_b": str(args.out_b),
        "model": model_path,
        "block_size": args.block_size,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "num_prompts_total": len(rows),
        "num_story_groups": len(groups),
        "shared_instruction_chars": len(instr),
        "shared_instruction_tokens": len(instr_tokens),
        "unavoidable_shared_full_blocks": n_shared_full_blocks,
        "part_a": {
            "num_prompts": len(rows_a),
            "num_stories": len(set_a),
            "num_tokens": tok_a,
            "num_unique_block_hashes": len(hashes_a),
        },
        "part_b": {
            "num_prompts": len(rows_b),
            "num_stories": len(set_b),
            "num_tokens": tok_b,
            "num_unique_block_hashes": len(hashes_b),
        },
        "token_imbalance": imbalance,
        "raw_hash_overlap": len(overlap_all),
        "instruction_only_hash_overlap": len(overlap_all & shared_prefix_hashes),
        "content_hash_overlap": len(content_overlap),
        "story_key_overlap": len(story_overlap),
        "passed": True,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_a}")
    print(f"Wrote {args.out_b}")
    print(f"Wrote {args.report}")
    print("PASS: no story-level / content block-hash overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
