#!/usr/bin/env python3
"""数据纪律审计：泄漏、id 唯一性、集合一致性。

用法：
    python scripts/audit_data_integrity.py
    python scripts/audit_data_integrity.py --json results/data-audit.json

检查项：
1. 每个数据文件的行数与唯一 id 数（暴露"条数虚高"：同题多解会让 412 条其实只有 114 题）；
2. dev 不得出现在任何训练数据里（test 同理，但本项目 test 未参与训练）；
3. 训练数据（采样/RFT/DPO/难题集）的 id 必须全部来自 train；
4. 所有评测结果必须评的是同一批题，且 ⊆ dev、∩ train = 0。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 数据文件登记（相对仓库根）
DATA_FILES: dict[str, str] = {
    "train(清洗后)": "data/processed/sft/gsm8k-train.jsonl",
    "dev(清洗后)": "data/processed/sft/gsm8k-dev.raw.jsonl",
    "train(原始)": "data/processed/sft/gsm8k-train.raw.jsonl",
    "采样 120 题": "experiments/sampling/gsm8k-train120-k4-t0.7.jsonl",
    "采样 300 题": "experiments/sampling/gsm8k-train300-k4-t0.7.jsonl",
    "RFT 数据": "data/processed/rft/gsm8k-train-rft.jsonl",
    "RFT 300q": "data/processed/rft/gsm8k-train-rft-300q.jsonl",
    "DPO 数据": "data/processed/dpo/gsm8k-train-dpo.jsonl",
    "DPO 300q": "data/processed/dpo/gsm8k-train-dpo-300q.jsonl",
    "hard26": "data/processed/sft/gsm8k-train-hard26.raw.jsonl",
}

# 训练用文件（必须全部来自 train）
TRAINING_FILES = ["采样 120 题", "采样 300 题", "RFT 数据", "RFT 300q", "DPO 数据", "DPO 300q", "hard26"]

# 评测结果（必须评同一批题、⊆ dev）
EVAL_FILES: dict[str, str] = {
    "base": "data/server-backup/baseline/gsm8k-dev-probe200-final.jsonl",
    "sft": "experiments/eval/sft-v3-ckpt400-dev200.jsonl",
    "rft": "experiments/eval/rft-dev200.jsonl",
    "dpo_rft": "experiments/eval/dpo-dev200.jsonl",
    "dpo_sft": "experiments/eval/dpo-from-sft-dev200.jsonl",
    "grpo": "experiments/eval/grpo-dev200.jsonl",
    "grpo_hard26": "experiments/eval/grpo-hard26-dev200.jsonl",
    "rft_300q": "experiments/eval/rft300q-dev200.jsonl",
    "dpo_300q": "experiments/eval/dpo300q-dev200.jsonl",
}


def load_ids(rel_path: str) -> list[str] | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    out: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(str(json.loads(line)["id"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="数据纪律审计")
    parser.add_argument("--json", type=Path, default=None, help="把结构化结果写到该路径")
    args = parser.parse_args()

    report: dict = {"files": {}, "leakage": {}, "provenance": {}, "eval_sets": {}}
    ids_by_name: dict[str, list[str]] = {}
    missing: list[str] = []

    print("=== 1. 行数与唯一 id ===")
    for name, rel in DATA_FILES.items():
        values = load_ids(rel)
        if values is None:
            missing.append(name)
            report["files"][name] = {"status": "MISSING", "path": rel}
            print(f"  {name:<14} MISSING ({rel})")
            continue
        ids_by_name[name] = values
        repeats = {k: v for k, v in Counter(values).items() if v > 1}
        avg = len(values) / len(set(values)) if values else 0
        report["files"][name] = {
            "status": "OK",
            "path": rel,
            "rows": len(values),
            "unique_ids": len(set(values)),
            "duplicated_ids": len(repeats),
            "avg_rows_per_id": round(avg, 2),
        }
        extra = f"  重复 {len(repeats)} 个 id，平均 {avg:.1f} 条/题" if repeats else ""
        print(f"  {name:<14} {len(values):>6} 行 / {len(set(values)):>6} 唯一 id{extra}")

    sets = {k: set(v) for k, v in ids_by_name.items()}
    dev = sets.get("dev(清洗后)", set())
    train = sets.get("train(清洗后)", set())

    print("\n=== 2. 泄漏检查：dev 不得进入任何训练数据 ===")
    ok_leak = True
    for name in ["train(清洗后)"] + TRAINING_FILES:
        other = sets.get(name)
        if other is None:
            continue
        overlap = len(dev & other)
        ok_leak &= overlap == 0
        report["leakage"][name] = overlap
        print(f"  dev ∩ {name:<12} = {overlap}  (应为 0)")

    print("\n=== 3. 来源检查：训练数据必须全部来自 train ===")
    ok_prov = True
    for name in TRAINING_FILES:
        values = sets.get(name)
        if values is None:
            continue
        outside = len(values - train)
        ok_prov &= outside == 0
        report["provenance"][name] = outside
        print(f"  {name:<12} 不在 train 中的 id = {outside}  (应为 0)")

    print("\n=== 4. 评测一致性：同一批题、⊆ dev、∩ train = 0 ===")
    eval_sets: dict[str, set[str]] = {}
    for name, rel in EVAL_FILES.items():
        values = load_ids(rel)
        if values is None:
            missing.append(f"eval:{name}")
            report["eval_sets"][name] = {"status": "MISSING"}
            print(f"  {name:<12} MISSING")
            continue
        eval_sets[name] = set(values)
        print(f"  {name:<12} {len(eval_sets[name]):>4} 题")

    all_same = bool(eval_sets) and all(v == next(iter(eval_sets.values())) for v in eval_sets.values())
    evaluated = next(iter(eval_sets.values())) if eval_sets else set()
    subset_dev = evaluated <= dev if dev else False
    intersect_train = len(evaluated & train)
    report["eval_sets"]["summary"] = {
        "all_same_question_set": all_same,
        "subset_of_dev": subset_dev,
        "overlap_with_train": intersect_train,
        "questions": len(evaluated),
    }
    print(f"  全部一致: {all_same}    评测题 ⊆ dev: {subset_dev}    评测题 ∩ train: {intersect_train} (应为 0)")

    passed = ok_leak and ok_prov and all_same and subset_dev and intersect_train == 0 and not missing
    report["passed"] = passed
    print(f"\n=== 结论：{'PASS' if passed else 'FAIL'}"
          + (f"（缺失：{missing}）" if missing else "") + " ===")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
