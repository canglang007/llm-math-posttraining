"""口径校准：逐题比对两条推理后端（HF vs vLLM）的预测是否一致。

换推理引擎时不能只看总分是否接近——必须回答三个问题：
1. 判分一致率多高？（同一题是否判出相同对/错）
2. 答案一致率多高？（抽取出的最终答案是否相同）
3. 翻转的题是哪些、方向如何？（HF 对 vLLM 错 / vLLM 对 HF 错）

用法：
    python scripts/compare_inference_backends.py \
        --baseline experiments/eval/sft-v3-ckpt400-dev200.jsonl \
        --candidate experiments/eval/sft-v3-ckpt400-dev200-vllm.jsonl \
        --dataset gsm8k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402
from math_posttrain.data import read_jsonl  # noqa: E402


def compare_rows(baseline: list[dict], candidate: list[dict], dataset: str) -> dict:
    """按 id 对齐两份预测，统计判分一致率、答案一致率与翻转明细。"""
    base_by_id = {str(row["id"]): row for row in baseline}
    cand_by_id = {str(row["id"]): row for row in candidate}
    common_ids = sorted(set(base_by_id) & set(cand_by_id))
    if not common_ids:
        raise ValueError("两份预测没有共同的 id，无法比对")

    stats = {
        "common": len(common_ids),
        "baseline_only": len(set(base_by_id) - set(cand_by_id)),
        "candidate_only": len(set(cand_by_id) - set(base_by_id)),
        "baseline_correct": 0,
        "candidate_correct": 0,
        "grade_agree": 0,
        "answer_agree": 0,
        "flips": [],
    }
    for qid in common_ids:
        base_row, cand_row = base_by_id[qid], cand_by_id[qid]
        reference = base_row.get("reference") or cand_row.get("reference") or ""
        base_ok = is_correct(base_row.get("prediction", ""), reference, dataset)
        cand_ok = is_correct(cand_row.get("prediction", ""), reference, dataset)
        stats["baseline_correct"] += int(base_ok)
        stats["candidate_correct"] += int(cand_ok)
        stats["grade_agree"] += int(base_ok == cand_ok)

        base_answer = extract_final_answer(base_row.get("prediction", ""), dataset)
        cand_answer = extract_final_answer(cand_row.get("prediction", ""), dataset)
        stats["answer_agree"] += int(base_answer == cand_answer)

        if base_ok != cand_ok:
            stats["flips"].append(
                {
                    "id": qid,
                    "direction": "baseline_only" if base_ok else "candidate_only",
                    "reference": reference,
                    "baseline_answer": base_answer,
                    "candidate_answer": cand_answer,
                }
            )

    n = stats["common"]
    stats["baseline_accuracy"] = stats["baseline_correct"] / n
    stats["candidate_accuracy"] = stats["candidate_correct"] / n
    stats["grade_agreement_rate"] = stats["grade_agree"] / n
    stats["answer_agreement_rate"] = stats["answer_agree"] / n
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="参考预测（HF 历史口径）")
    parser.add_argument("--candidate", type=Path, required=True, help="待校准预测（vLLM）")
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = compare_rows(read_jsonl(args.baseline), read_jsonl(args.candidate), args.dataset)
    summary = {
        key: value
        for key, value in stats.items()
        if key != "flips"
    }
    summary["flip_count"] = len(stats["flips"])
    text = json.dumps({"summary": summary, "flips": stats["flips"]}, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
