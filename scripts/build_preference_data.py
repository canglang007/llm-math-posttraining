"""Build RFT and DPO records from answer-verified candidates.

Candidate JSONL schema:
  {"id","question","reference","solution","correct"?,"category"?,"level"?,"source"?}
`correct` is optional; when present it must agree with the public answer check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import read_jsonl, write_jsonl  # noqa: E402
from math_posttrain.preferences import build_dpo_records, build_rft_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--rft-output", type=Path, default=Path("data/processed/rft/candidates.jsonl"))
    parser.add_argument("--dpo-output", type=Path, default=Path("data/processed/dpo/candidates.jsonl"))
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    rft = build_rft_records(candidates, args.dataset)
    dpo = build_dpo_records(candidates, args.dataset)
    write_jsonl(args.rft_output, rft)
    write_jsonl(args.dpo_output, dpo)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "candidates": len(candidates),
                "rft_records": len(rft),
                "dpo_pairs": len(dpo),
                "rft_output": str(args.rft_output),
                "dpo_output": str(args.dpo_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
