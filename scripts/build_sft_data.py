"""Build SFT records from public GSM8K/MATH solutions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import load_examples, write_jsonl  # noqa: E402
from math_posttrain.sft import build_sft_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw_path = args.raw_dir / args.dataset / f"{args.split}.jsonl"
    examples = load_examples(raw_path, source=args.dataset, split=args.split)
    if args.limit:
        examples = examples[: args.limit]
    records = build_sft_records(examples, args.dataset)
    output = args.output or Path("data/processed/sft") / f"{args.dataset}-{args.split}.jsonl"
    write_jsonl(output, records)
    lengths = [len(record["messages"][-1]["content"]) for record in records]
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "input": len(examples),
                "output": len(records),
                "skipped": len(examples) - len(records),
                "mean_solution_chars": round(sum(lengths) / len(lengths), 2) if lengths else None,
                "output_path": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
