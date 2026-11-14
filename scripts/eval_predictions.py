"""Evaluate a prediction JSONL file with GSM8K/MATH answer rules."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import read_jsonl  # noqa: E402
from math_posttrain.evaluation import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    summary = evaluate(rows, args.dataset)
    payload = asdict(summary)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
