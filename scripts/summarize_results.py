#!/usr/bin/env python3
"""从评测结果 jsonl 重算主表与配对检验（不让数字靠手抄）。

用法：
    python scripts/summarize_results.py --output-dir results
    python scripts/summarize_results.py --output-dir results --reference SFT

设计要点：
- 所有指标都从原始 `prediction` 字段重算，判分统一走 `math_posttrain.answers.is_correct`，
  与训练/评测期完全同一口径；
- 结果文件缺失时如实标记为 MISSING，不跳过、不补数（避免"表里有数、盘里无据"）；
- 配对检验只对共同题目做，并校验各 run 的题目集合一致。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402

DATASET = "gsm8k"

# 实验登记表：key -> (阶段, 显示名, 结果文件, 训练数据说明)
RUNS: list[dict] = [
    {
        "key": "base",
        "stage": "E0",
        "label": "Qwen3-4B-Base",
        "path": "data/server-backup/baseline/gsm8k-dev-probe200-final.jsonl",
        "train_data": "—",
        "note": "零样本基线，无 SFT",
    },
    {
        "key": "sft",
        "stage": "E1",
        "label": "SFT (Base+LoRA, 400 步)",
        "path": "experiments/eval/sft-v3-ckpt400-dev200.jsonl",
        "train_data": "官方 CoT 6973 条",
        "note": "停止符改用 <|endoftext|> 的干净重训版",
    },
    {
        "key": "rft",
        "stage": "E2",
        "label": "RFT (起点 SFT)",
        "path": "experiments/eval/rft-dev200.jsonl",
        "train_data": "自采正确解 412 条",
        "note": "train 120 题 × k=4 → 480 候选 → 过滤后 412 条",
    },
    {
        "key": "dpo_from_rft",
        "stage": "E3a",
        "label": "DPO (起点 RFT, 链式)",
        "path": "experiments/eval/dpo-dev200.jsonl",
        "train_data": "26 对偏好数据",
        "note": "旧链路；本地无结果文件时会标 MISSING（分数见 docs/main-results.md）",
    },
    {
        "key": "dpo_from_sft",
        "stage": "E3b",
        "label": "DPO (起点 SFT, 平行)",
        "path": "experiments/eval/dpo-from-sft-dev200.jsonl",
        "train_data": "26 对偏好数据",
        "note": "lr 5e-5（修复后）、3 epochs",
    },
    {
        "key": "grpo",
        "stage": "E4",
        "label": "GRPO (起点 SFT, 全量 120 题)",
        "path": "experiments/eval/grpo-dev200.jsonl",
        "train_data": "在线采样 + 答案规则奖励",
        "note": "30 步；frac_reward_zero_std≈0.67",
    },
    {
        "key": "grpo_hard26",
        "stage": "E4b",
        "label": "GRPO (起点 SFT, 仅 26 道难题)",
        "path": "experiments/eval/grpo-hard26-dev200.jsonl",
        "train_data": "难度筛选 26 题 + 在线采样",
        "note": "16 步；zero_std 0.33、reward 0.5→0.75 但未泛化",
    },
]


def load_rows(path: Path) -> dict[str, dict]:
    """读 jsonl，按 id 建索引。"""
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows[str(record["id"])] = record
    return rows


def score(rows: dict[str, dict]) -> dict[str, bool]:
    """判分：统一走 is_correct。"""
    return {
        key: is_correct(
            record.get("prediction", ""),
            record.get("reference") or record.get("answer") or "",
            DATASET,
        )
        for key, record in rows.items()
    }


def mcnemar(only_a: int, only_b: int) -> float:
    """McNemar 精确检验（双侧）。"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) * 0.5**n for i in range(0, min(only_a, only_b) + 1))
    return min(1.0, tail * 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="从评测 jsonl 重算主表")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--reference", default="sft", help="配对检验的参照 run key")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 逐个 run 算指标 ---
    summaries: list[dict] = []
    scores: dict[str, dict[str, bool]] = {}
    ids: dict[str, set[str]] = {}

    for run in RUNS:
        path = REPO_ROOT / run["path"]
        entry = {
            "key": run["key"],
            "stage": run["stage"],
            "label": run["label"],
            "path": run["path"],
            "train_data": run["train_data"],
            "note": run["note"],
        }
        if not path.exists():
            entry.update({"status": "MISSING", "questions": None, "correct": None,
                          "accuracy": None, "extract_fail": None, "avg_chars": None})
            summaries.append(entry)
            print(f"[missing] {run['key']:<12} {run['path']}", file=sys.stderr)
            continue

        rows = load_rows(path)
        graded = score(rows)
        total = len(graded)
        correct = sum(graded.values())
        extract_fail = sum(
            1 for record in rows.values()
            if extract_final_answer(record.get("prediction", ""), DATASET) is None
        )
        lengths = [len(record.get("prediction", "")) for record in rows.values()]

        entry.update({
            "status": "OK",
            "questions": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else None,
            "extract_fail": extract_fail,
            "extract_fail_rate": round(extract_fail / total, 4) if total else None,
            "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else None,
        })
        summaries.append(entry)
        scores[run["key"]] = graded
        ids[run["key"]] = set(graded)

    # --- 校验题目集合一致 ---
    sets = [v for v in ids.values()]
    common_ids = set.intersection(*sets) if sets else set()
    for key, value in ids.items():
        if value != common_ids:
            missing = sorted(common_ids - value)[:3]
            extra = sorted(value - common_ids)[:3]
            print(f"[warn] {key} 题目集合与其他 run 不一致：缺 {missing}，多 {extra}", file=sys.stderr)

    # --- 配对检验（vs reference） ---
    pairwise: list[dict] = []
    ref_key = args.reference
    if ref_key in scores:
        ref_scores = scores[ref_key]
        for run in RUNS:
            key = run["key"]
            if key == ref_key or key not in scores:
                continue
            shared = sorted(set(ref_scores) & set(scores[key]))
            only_a = sum(1 for i in shared if scores[key][i] and not ref_scores[i])
            only_b = sum(1 for i in shared if ref_scores[i] and not scores[key][i])
            both = sum(1 for i in shared if scores[key][i] and ref_scores[i])
            neither = len(shared) - only_a - only_b - both
            pairwise.append({
                "run": key,
                "reference": ref_key,
                "shared_questions": len(shared),
                "both_correct": both,
                "only_run": only_a,
                "only_reference": only_b,
                "both_wrong": neither,
                "delta_pt": round((only_a - only_b) / len(shared) * 100, 2) if shared else None,
                "mcnemar_p": round(mcnemar(only_a, only_b), 4),
            })

    # --- 写 summary.json ---
    payload = {
        "dataset": DATASET,
        "split": "dev-200",
        "decoding": "greedy, batch_size=1",
        "reference_run": ref_key,
        "runs": summaries,
        "pairwise_vs_reference": pairwise,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 写 summary.csv ---
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "key", "label", "status", "questions", "correct",
                         "accuracy", "extract_fail", "avg_chars", "train_data", "path"])
        for row in summaries:
            writer.writerow([row["stage"], row["key"], row["label"], row["status"],
                             row["questions"], row["correct"], row["accuracy"],
                             row["extract_fail"], row["avg_chars"],
                             row["train_data"], row["path"]])

    # --- 写 pairwise.csv ---
    with (args.output_dir / "pairwise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "vs", "shared", "both_correct", "only_run",
                         "only_ref", "both_wrong", "delta_pt", "mcnemar_p"])
        for row in pairwise:
            writer.writerow([row["run"], row["reference"], row["shared_questions"],
                             row["both_correct"], row["only_run"], row["only_reference"],
                             row["both_wrong"], row["delta_pt"], row["mcnemar_p"]])

    # --- 控制台表格 ---
    print(f"\n{'阶段':<5} {'模型':<32} {'题数':>5} {'正确':>5} {'准确率':>8} {'抽取失败':>7}")
    print("-" * 72)
    for row in summaries:
        if row["status"] != "OK":
            print(f"{row['stage']:<5} {row['label']:<32} {'—':>5} {'—':>5} {'MISSING':>8} {'—':>7}")
            continue
        print(f"{row['stage']:<5} {row['label']:<32} {row['questions']:>5} "
              f"{row['correct']:>5} {row['accuracy']:>8.3f} {row['extract_fail']:>7}")

    if pairwise:
        print(f"\n配对检验（vs {ref_key}）")
        print("-" * 72)
        print(f"{'run':<14} {'净差':>8} {'仅它对':>7} {'仅参照':>7} {'p 值':>7}")
        for row in pairwise:
            print(f"{row['run']:<14} {row['delta_pt']:>+7.1f}pt {row['only_run']:>7} "
                  f"{row['only_reference']:>7} {row['mcnemar_p']:>7.3f}")

    print(f"\n已写出：{args.output_dir}/summary.json, summary.csv, pairwise.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
