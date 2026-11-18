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
    parser.add_argument(
        "--max-per-question",
        type=int,
        default=None,
        help="每题最多保留多少条正确解；1 表示按题去重（避免条数虚高于题数）。默认保留全部。",
    )
    parser.add_argument(
        "--ignore-correctness",
        action="store_true",
        help="A2 消融用：不判答案对错，所有候选都可作 RFT 训练目标（模拟「不用验证器」）。",
    )
    args = parser.parse_args()

    if args.ignore_correctness and args.max_per_question is None:
        print(
            "[warn] --ignore-correctness 未配 --max-per-question：数据量会是处理组的数倍，"
            "「验证」与「数据量」两个变量会混在一起",
            file=sys.stderr,
        )

    candidates = read_jsonl(args.candidates)
    rft = build_rft_records(
        candidates,
        args.dataset,
        max_per_question=args.max_per_question,
        ignore_correctness=args.ignore_correctness,
    )
    dpo = build_dpo_records(candidates, args.dataset)
    write_jsonl(args.rft_output, rft)
    write_jsonl(args.dpo_output, dpo)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "candidates": len(candidates),
                "candidate_questions": len({str(c.get("id", "")) for c in candidates}),
                "rft_records": len(rft),
                "rft_questions": len({str(r.get("id", "")) for r in rft}),
                "dpo_pairs": len(dpo),
                "max_per_question": args.max_per_question,
                "verified": not args.ignore_correctness,
                "rft_output": str(args.rft_output),
                "dpo_output": str(args.dpo_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
