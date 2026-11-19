"""test 集失败分析：为项目叙事提供可核查的具体案例。

输出三类材料：
1. 各模型的抽取失败率与"可解析题中的正确率"——区分「不会答」和「不会收尾」；
2. A2（不验证）与 E2b（验证）的分叉题样本——看错误解污染长什么样；
3. DPO 与 SFT 的分叉题样本——看偏好数据改变了什么。

用法：
    python scripts/analyze_test_failures.py --eval-dir experiments/eval/vllm-test --output docs/failure-analysis-test.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_final_answer, is_correct  # noqa: E402

MODELS = [
    ("base", "E0 Base"),
    ("sft", "E1 SFT"),
    ("e2a-rft", "E2a RFT-114q"),
    ("e2b-rft300q", "E2b RFT-286q（验证）"),
    ("e2c-rftbase", "E2c RFT-from-Base"),
    ("a2-nofilter", "A2 RFT（不验证）"),
    ("e3a-dpo-from-rft", "E3a DPO(RFT 起)"),
    ("e3b-dpo-from-sft", "E3b DPO(26 对)"),
    ("e3c-dpo-79pairs", "E3c DPO(79 对)"),
    ("e4-grpo", "E4 GRPO"),
    ("e4b-grpo-hard26", "E4b GRPO(难题 26)"),
]


def load(path: Path) -> dict[str, dict]:
    return {str(json.loads(line)["id"]): json.loads(line) for line in path.open()}


def stats(rows: dict[str, dict]) -> dict:
    total = len(rows)
    graded = {qid: is_correct(r.get("prediction", ""), r.get("reference") or "", "gsm8k") for qid, r in rows.items()}
    fail = {
        qid: extract_final_answer(r.get("prediction", ""), "gsm8k") is None
        for qid, r in rows.items()
    }
    parseable = [qid for qid in rows if not fail[qid]]
    correct_parseable = sum(1 for qid in parseable if graded[qid])
    return {
        "n": total,
        "correct": sum(graded.values()),
        "accuracy": sum(graded.values()) / total,
        "extract_fail": sum(fail.values()),
        "extract_fail_rate": sum(fail.values()) / total,
        "accuracy_among_parseable": correct_parseable / len(parseable) if parseable else 0.0,
        "mean_len": mean(len(r.get("prediction", "")) for r in rows.values()),
    }


def diffs(a: dict[str, dict], b: dict[str, dict], dataset: str = "gsm8k") -> tuple[list[str], list[str]]:
    """返回 (只有 a 对, 只有 b 对) 的题目 id。"""
    a_only, b_only = [], []
    for qid in sorted(set(a) & set(b)):
        a_ok = is_correct(a[qid].get("prediction", ""), a[qid].get("reference") or "", dataset)
        b_ok = is_correct(b[qid].get("prediction", ""), b[qid].get("reference") or "", dataset)
        if a_ok and not b_ok:
            a_only.append(qid)
        elif b_ok and not a_ok:
            b_only.append(qid)
    return a_only, b_only


def snippet(text: str, limit: int = 320) -> str:
    text = text.replace("\n", " ⏎ ")
    return text[:limit] + ("…" if len(text) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("experiments/eval/vllm-test"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/gsm8k/test.jsonl"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("docs/failure-analysis-test.md"))
    args = parser.parse_args()

    runs = {key: load(args.eval_dir / f"{key}.jsonl") for key, _ in MODELS}
    questions = {}
    if args.questions.exists():
        questions = {str(json.loads(line)["id"]): json.loads(line)["question"] for line in args.questions.open()}

    lines = ["# test 集失败分析（1319 题，vLLM bf16 口径）", ""]
    lines.append("## 1. 抽取失败 vs 算错：两种失败要分开看")
    lines.append("")
    lines.append("| 模型 | 准确率 | 抽取失败 | 失败率 | 可解析题中的正确率 | 平均长度 |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in MODELS:
        s = stats(runs[key])
        lines.append(
            f"| {label} | {s['accuracy']:.4f} ({s['correct']}/{s['n']}) | {s['extract_fail']} | "
            f"{s['extract_fail_rate']:.1%} | {s['accuracy_among_parseable']:.4f} | {s['mean_len']:.0f} |"
        )
    lines.append("")

    for title, a_key, a_label, b_key, b_label in [
        ("2. A2（不验证）与 E2b（验证）的分叉题", "e2b-rft300q", "E2b 验证", "a2-nofilter", "A2 不验证"),
        ("3. DPO(26 对) 与 SFT 的分叉题", "sft", "SFT", "e3b-dpo-from-sft", "E3b DPO"),
    ]:
        a_only, b_only = diffs(runs[a_key], runs[b_key])
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- {a_label} 独对：{len(a_only)} 题；{b_label} 独对：{len(b_only)} 题")
        lines.append("")
        for tag, ids, winner_key, loser_key, winner, loser in [
            (f"{a_label} 独对（{b_label} 错）", a_only, a_key, b_key, a_label, b_label),
            (f"{b_label} 独对（{a_label} 错）", b_only, b_key, a_key, b_label, a_label),
        ]:
            lines.append(f"### {tag} —— 样例 {min(args.samples, len(ids))}/{len(ids)}")
            lines.append("")
            for qid in ids[: args.samples]:
                ref = runs[a_key][qid].get("reference") or ""
                q = questions.get(qid, "(题目文本见 data/raw/gsm8k/test.jsonl)")
                w_pred = runs[winner_key][qid].get("prediction", "")
                l_pred = runs[loser_key][qid].get("prediction", "")
                lines.append(f"**{qid}** · 参考答案 `{ref}`")
                lines.append("")
                lines.append(f"> {q}")
                lines.append("")
                lines.append(f"- {winner}：`{snippet(w_pred)}`")
                lines.append(f"- {loser}：`{snippet(l_pred)}`")
                lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {args.output} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
