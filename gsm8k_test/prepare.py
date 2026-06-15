#!/usr/bin/env python3
"""Generate GSM8K benchmark JSONL from cached raw data in this directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLLM_ROOT))

from tests.evals.gsm8k.gsm8k_eval import (  # noqa: E402
    _build_gsm8k_prompts,
    load_gsm8k_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K custom dataset JSONL")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "gsm8k_200.jsonl",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        print("Raw train.jsonl / test.jsonl not found, downloading...")
        load_gsm8k_data()
    else:
        import tests.evals.gsm8k.gsm8k_eval as gsm8k_eval

        gsm8k_eval.download_and_cache_file = lambda url, filename=None: str(
            data_dir / Path(filename or url.split("/")[-1]).name
        )

    prompts, _ = _build_gsm8k_prompts(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
