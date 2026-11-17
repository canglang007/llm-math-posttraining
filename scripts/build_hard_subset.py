"""按难度筛题：从采样候选里挑出「模型有对有错」的题，供 GRPO/RL 训练。

为什么需要：
GRPO 的梯度来自「同一题多个候选之间的相对优势」。若某题 k 个候选全对或全错，
组内奖励标准差为 0、优势为 0，该题不产生任何梯度。

实测（120 题 × k=4，SFT checkpoint-400）：
    4/4 全对  88 题 (73%)   ← 无梯度
    3/4       11 题 (9%)
    2/4       12 题 (10%)
    1/4        3 题 (2.5%)
    0/4 全错   6 题 (5%)    ← 也无梯度
只有 26 题（22%）有区分度，这正是 `frac_reward_zero_std ≈ 0.67` 的成因。

用法：
    python scripts/build_hard_subset.py \
        --sampling-file experiments/sampling/gsm8k-train120-k4-t0.7.jsonl \
        --dataset gsm8k \
        --output data/processed/sft/gsm8k-train-hard26.raw.jsonl

输出是 raw 格式（id/question/answer），与 build_sft_data.py 的 raw 产物同构，
可直接作为 GRPO 的 prompt_file。
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-correct",
        type=int,
        default=1,
        help="正确候选下界（含）。默认 1：排除全错的题（模型完全不会，也难有梯度）。",
    )
    parser.add_argument(
        "--max-correct",
        type=int,
        default=None,
        help="正确候选上界（不含）。默认 = k，即排除全对的题。",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.sampling_file)
    kept: list[dict] = []
    dist: dict[int, int] = {}
    for row in rows:
        candidates = row.get("candidates", [])
        k = len(candidates)
        ok = sum(1 for text in candidates if is_correct(str(text), str(row["reference"]), args.dataset))
        dist[ok] = dist.get(ok, 0) + 1
        upper = k if args.max_correct is None else args.max_correct
        if args.min_correct <= ok < upper:
            kept.append({"id": str(row["id"]), "question": str(row["question"]), "answer": str(row["reference"])})

    write_jsonl(args.output, kept)
    total = sum(dist.values())
    print(
        json.dumps(
            {
                "questions": total,
                "correct_count_distribution": {f"{n}/{max(dist) if dist else 0}": c for n, c in sorted(dist.items())},
                "kept": len(kept),
                "kept_ratio": round(len(kept) / max(1, total), 3),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
