"""Build SFT records from public GSM8K/MATH solutions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import load_examples, write_jsonl  # noqa: E402
from math_posttrain.sft import build_sft_records  # noqa: E402
from math_posttrain.splits import carve_dev  # noqa: E402


def _stats(dataset: str, split: str, input_count: int, records: list[dict], path: Path) -> dict:
    lengths = [len(record["messages"][-1]["content"]) for record in records]
    return {
        "dataset": dataset,
        "split": split,
        "input": input_count,
        "output": len(records),
        "skipped": input_count - len(records),
        "mean_solution_chars": round(sum(lengths) / len(lengths), 2) if lengths else None,
        "output_path": str(path),
    }


def _raw_rows(examples: list) -> list[dict]:
    return [
        {
            "id": example.id,
            "question": example.question,
            "answer": example.answer,
            "solution": example.solution,
            "category": example.category,
            "level": example.level,
        }
        for example in examples
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--dev-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw_path = args.raw_dir / args.dataset / f"{args.split}.jsonl"
    examples = load_examples(raw_path, source=args.dataset, split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    if args.dev_size:
        if args.split != "train":
            raise SystemExit("--dev-size is only valid with --split train")
        train_ids, dev_ids = carve_dev([example.id for example in examples], dev_size=args.dev_size)
        by_id = {example.id: example for example in examples}
        train_examples = [by_id[item] for item in train_ids]
        dev_examples = [by_id[item] for item in dev_ids]
        train_records = build_sft_records(train_examples, args.dataset)
        dev_records = build_sft_records(dev_examples, args.dataset)
        train_output = args.output or Path("data/processed/sft") / f"{args.dataset}-train.jsonl"
        dev_output = args.dev_output or train_output.with_name(f"{args.dataset}-dev.jsonl")
        dev_raw_output = dev_output.with_name(f"{args.dataset}-dev.raw.jsonl")
        write_jsonl(train_output, train_records)
        write_jsonl(dev_output, dev_records)
        write_jsonl(dev_raw_output, _raw_rows(dev_examples))
        dev_stats = _stats(args.dataset, "dev", len(dev_examples), dev_records, dev_output)
        dev_stats["raw_output_path"] = str(dev_raw_output)
        print(json.dumps(
            {
                "train": _stats(args.dataset, "train", len(train_examples), train_records, train_output),
                "dev": dev_stats,
            },
            ensure_ascii=False,
        ))
        return

    records = build_sft_records(examples, args.dataset)
    output = args.output or Path("data/processed/sft") / f"{args.dataset}-{args.split}.jsonl"
    write_jsonl(output, records)
    print(json.dumps(_stats(args.dataset, args.split, len(examples), records, output), ensure_ascii=False))


if __name__ == "__main__":
    main()
