"""推理时验证结果表：把采样候选聚合出 pass@1 / maj@k / oracle pass@k。

纯离线分析，不加载模型。候选由 run_sampling.py 在 GPU 上采样产生，
所有判分仍然只用公开答案（answers.is_correct），不自造 ground truth。

产出：
- 逐 k 的 maj@k、oracle pass@k、单条采样平均正确率（含 Wilson 区间）；
- 与 greedy baseline（E0）按 id 对齐后的增益分解：
  「E0 做错但 maj@k 救回」vs「E0 做对却被投票带偏」；
- 格式过滤 / 长度过滤两种验证策略的对照；
- 成本：候选数、估计生成 token、每救回一题的成本。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402
from math_posttrain.inference_time import (  # noqa: E402
    majority_vote,
    oracle_pass_at_k,
    vote_is_correct,
)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间，样本量 200 时比正态近似稳。"""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def filter_candidates(
    candidates: list[str], dataset: str, max_chars: int | None = None
) -> list[str]:
    """验证器前置过滤：丢掉抽取不出答案的候选，可选再丢掉超长候选。

    超长阈值来自 E0 失败分析（输出越长越容易跑飞），是固定超参，
    不在评测集上扫最优点，只报给定阈值下的结果。
    """
    kept = []
    for text in candidates:
        if extract_final_answer(text, dataset) is None:
            continue
        if max_chars is not None and len(text) > max_chars:
            continue
        kept.append(text)
    return kept


def per_question_curve(
    candidates: list[str], reference: str, dataset: str, max_k: int
) -> dict[int, dict[str, float | bool | None]]:
    """单个题目在 k'=1..max_k 上的各项指标（用前缀，保证单调可比）。"""
    curve: dict[int, dict[str, float | bool | None]] = {}
    for k in range(1, max_k + 1):
        prefix = candidates[:k]
        correct = sum(1 for text in prefix if is_correct(text, reference, dataset))
        curve[k] = {
            "sampled_correct": correct,
            "sampled_pass@1": correct / k,
            "maj": vote_is_correct(prefix, reference, dataset),
            "oracle": oracle_pass_at_k(prefix, reference, dataset),
            "chosen": majority_vote(prefix, dataset),
        }
    return curve


def summarize(
    rows: list[dict],
    dataset: str,
    max_k: int,
    baseline: dict[str, bool] | None = None,
    max_chars: int | None = None,
    avg_tokens: float | None = None,
) -> dict:
    total_questions = len(rows)
    curve_rows: dict[int, dict[str, int]] = {
        k: {"sampled_correct": 0, "sampled_slots": 0, "maj": 0, "oracle": 0}
        for k in range(1, max_k + 1)
    }
    filtered_curve: dict[int, dict[str, int]] = {
        k: {"maj": 0, "empty": 0} for k in range(1, max_k + 1)
    }
    valid_counts: list[int] = []
    candidate_total = 0
    candidate_invalid = 0
    lengths: list[int] = []

    per_question: list[dict] = []
    for row in rows:
        candidates = row["candidates"][:max_k]
        reference = row["reference"]
        curve = per_question_curve(candidates, reference, dataset, len(candidates))
        for k, stats in curve.items():
            curve_rows[k]["sampled_correct"] += stats["sampled_correct"]
            curve_rows[k]["sampled_slots"] += k
            curve_rows[k]["maj"] += int(bool(stats["maj"]))
            curve_rows[k]["oracle"] += int(bool(stats["oracle"]))
            kept = filter_candidates(candidates[:k], dataset, max_chars)
            if not kept:
                filtered_curve[k]["empty"] += 1
            elif vote_is_correct(kept, reference, dataset):
                filtered_curve[k]["maj"] += 1
        valid_counts.append(sum(1 for text in candidates if extract_final_answer(text, dataset)))
        candidate_total += len(candidates)
        candidate_invalid += sum(
            1 for text in candidates if extract_final_answer(text, dataset) is None
        )
        lengths.extend(len(text) for text in candidates)
        per_question.append({
            "id": row["id"],
            "greedy_correct": baseline.get(row["id"]) if baseline else None,
            "maj_at_max_k": bool(curve[len(candidates)]["maj"]),
            "oracle_at_max_k": bool(curve[len(candidates)]["oracle"]),
            "valid_candidates": valid_counts[-1],
            "candidate_lengths": [len(text) for text in candidates],
        })

    curve = []
    for k in range(1, max_k + 1):
        maj_hits = curve_rows[k]["maj"]
        oracle_hits = curve_rows[k]["oracle"]
        low, high = wilson_interval(maj_hits, total_questions)
        entry = {
            "k": k,
            "sampled_pass@1": round(curve_rows[k]["sampled_correct"] / max(1, curve_rows[k]["sampled_slots"]), 4),
            "maj@k": round(maj_hits / total_questions, 4),
            "maj@k_ci95": [round(low, 4), round(high, 4)],
            "maj@k_hits": maj_hits,
            "oracle_pass@k": round(oracle_hits / total_questions, 4),
            "filtered_maj@k": round(filtered_curve[k]["maj"] / total_questions, 4),
            "filtered_empty": filtered_curve[k]["empty"],
            "verification_headroom": round((oracle_hits - maj_hits) / total_questions, 4),
        }
        if avg_tokens is not None:
            tokens = total_questions * k * avg_tokens
            entry["estimated_tokens"] = int(tokens)
            entry["tokens_per_question"] = int(k * avg_tokens)
        curve.append(entry)

    report: dict = {
        "questions": total_questions,
        "max_k": max_k,
        "dataset": dataset,
        "length_filter_max_chars": max_chars,
        "avg_candidates_per_question": round(candidate_total / max(1, total_questions), 2),
        "avg_valid_candidates": round(sum(valid_counts) / max(1, total_questions), 2),
        "candidate_extraction_failure_rate": round(candidate_invalid / max(1, candidate_total), 4),
        "avg_candidate_chars": round(sum(lengths) / max(1, len(lengths)), 1),
        "curve": curve,
    }

    if baseline:
        paired = [row for row in per_question if row["greedy_correct"] is not None]
        greedy_hits = sum(1 for row in paired if row["greedy_correct"])
        rescued = sum(
            1 for row in paired if not row["greedy_correct"] and row["maj_at_max_k"]
        )
        hurt = sum(1 for row in paired if row["greedy_correct"] and not row["maj_at_max_k"])
        oracle_on_greedy_wrong = sum(
            1 for row in paired if not row["greedy_correct"] and row["oracle_at_max_k"]
        )
        greedy_wrong = len(paired) - greedy_hits
        report["baseline_comparison"] = {
            "paired_questions": len(paired),
            "greedy_pass@1": round(greedy_hits / len(paired), 4),
            "greedy_hits": greedy_hits,
            "maj_at_max_k": round(sum(1 for row in paired if row["maj_at_max_k"]) / len(paired), 4),
            "rescued_by_vote": rescued,
            "hurt_by_vote": hurt,
            "net_gain_questions": rescued - hurt,
            "greedy_wrong_recall_of_oracle": round(oracle_on_greedy_wrong / max(1, greedy_wrong), 4),
            "oracle_reachable_on_greedy_wrong": oracle_on_greedy_wrong,
        }
    report["per_question"] = per_question
    return report


def to_markdown(report: dict) -> str:
    lines = [
        "# 推理时验证结果表",
        "",
        f"- 题目数：{report['questions']}，最大 k：{report['max_k']}，数据集：{report['dataset']}",
        f"- 平均有效候选：{report['avg_valid_candidates']} / 题",
        f"- 候选级抽取失败率：{report['candidate_extraction_failure_rate']:.2%}",
        f"- 候选平均字符数：{report['avg_candidate_chars']}",
        "",
        "| k | 单条采样 pass@1 | maj@k | maj@k 95% CI | oracle pass@k | 格式过滤后 maj@k | 验证余量 | 估计 tokens/题 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in report["curve"]:
        low, high = entry["maj@k_ci95"]
        tokens = entry.get("tokens_per_question")
        lines.append(
            f"| {entry['k']} | {entry['sampled_pass@1']:.4f} | {entry['maj@k']:.4f} | "
            f"[{low:.3f}, {high:.3f}] | {entry['oracle_pass@k']:.4f} | "
            f"{entry['filtered_maj@k']:.4f} | {entry['verification_headroom']:+.4f} | "
            f"{tokens if tokens is not None else '-'} |"
        )
    if "baseline_comparison" in report:
        b = report["baseline_comparison"]
        lines += [
            "",
            "## 与 greedy baseline 的对齐比较",
            "",
            f"- 配对题数：{b['paired_questions']}；greedy pass@1 = {b['greedy_pass@1']:.4f}（{b['greedy_hits']} 题）",
            f"- maj@k = {b['maj_at_max_k']:.4f}；救回 {b['rescued_by_vote']} 题，带偏 {b['hurt_by_vote']} 题，净增 {b['net_gain_questions']:+d} 题",
            f"- greedy 做错的 {b['oracle_reachable_on_greedy_wrong']} 题里，候选池至少有一条正确（召回 {b['greedy_wrong_recall_of_oracle']:.2%}），这是验证信号的天花板",
        ]
    return "\n".join(lines) + "\n"


def load_baseline(path: Path, dataset: str) -> dict[str, bool]:
    result: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["id"]] = is_correct(row["prediction"], row["reference"], dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-file", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="gsm8k")
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--length-filter", type=int, default=1000)
    parser.add_argument("--avg-tokens", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/analysis/sampling-report.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.sampling_file.open(encoding="utf-8") if line.strip()]
    max_k = args.max_k or min(len(row["candidates"]) for row in rows)
    baseline = load_baseline(args.baseline, args.dataset) if args.baseline else None

    report = summarize(
        rows,
        dataset=args.dataset,
        max_k=max_k,
        baseline=baseline,
        max_chars=args.length_filter,
        avg_tokens=args.avg_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(to_markdown(report), encoding="utf-8")

    summary = {key: value for key, value in report.items() if key != "per_question"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[written] {args.output}\n[written] {markdown_path}")


if __name__ == "__main__":
    main()
