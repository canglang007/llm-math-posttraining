"""把采样产物展开成「候选级」JSONL，供 RFT / DPO 数据构造使用。

采样产物（run_sampling.py）是「每题一条」：
  {"id","question","reference","candidates": [text, text, ...]}

而 preferences.py 需要「每条候选一行」：
  {"id","question","reference","solution","correct","split","source"}

本脚本只做这一层展开 + 用公开答案打标，不产生任何新标签。

数据纪律：
- ``--split`` 必须显式指定。训练用数据只能来自 ``train``；
  ``dev`` 仅用于验证管线是否跑通，产物文件名会被强制加上 ``dev-`` 前缀，
  避免误当成训练数据使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import is_correct  # noqa: E402
from math_posttrain.data import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-per-question",
        type=int,
        default=None,
        help="每题最多保留几条候选（默认全部）。只保留正确候选的场景请用 --correct-only。",
    )
    parser.add_argument("--correct-only", action="store_true", help="只输出答案正确的候选（RFT 用）")
    parser.add_argument("--min-correct-per-question", type=int, default=1, help="每题至少要有几条正确候选才保留")
    args = parser.parse_args()

    rows = read_jsonl(args.sampling_file)
    records: list[dict] = []
    stats = {"questions": len(rows), "candidates": 0, "kept": 0, "correct": 0, "dropped_questions": 0}

    for row in rows:
        question_id = str(row.get("id", ""))
        question = str(row["question"])
        reference = str(row["reference"])
        solutions = [str(text) for text in row.get("candidates", [])]
        if args.max_per_question is not None:
            solutions = solutions[: args.max_per_question]

        labelled = []
        for solution in solutions:
            stats["candidates"] += 1
            correct = is_correct(solution, reference, args.dataset)
            if correct:
                stats["correct"] += 1
            labelled.append((solution, correct))

        correct_count = sum(1 for _, ok in labelled if ok)
        if correct_count < args.min_correct_per_question:
            stats["dropped_questions"] += 1
            continue

        for solution, correct in labelled:
            if args.correct_only and not correct:
                continue
            records.append(
                {
                    "id": question_id,
                    "question": question,
                    "reference": reference,
                    "solution": solution,
                    "correct": correct,
                    "split": args.split,
                    "source": args.dataset,
                }
            )
            stats["kept"] += 1

    if args.split == "dev" and not args.output.name.startswith("dev-"):
        args.output = args.output.with_name(f"dev-{args.output.name}")

    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                **stats,
                "split": args.split,
                "keep_rate": round(stats["kept"] / max(1, stats["candidates"]), 4),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
