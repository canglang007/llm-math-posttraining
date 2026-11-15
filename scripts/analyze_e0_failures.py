"""离线失败分析：从 E0 预测里产出口径敏感度与失败模式统计。

不加载模型，Mac 上直接跑。正式指标一律用 answers.py 的严格抽取；
这里的「宽松抽取」只用于估计判分口径的上界，不写进正式结果表。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_gsm8k_answer, is_correct  # noqa: E402
from math_posttrain.data import write_jsonl  # noqa: E402

LOOSE = re.compile(r"####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def loose_extract(text: str) -> str | None:
    matches = LOOSE.findall(text)
    if not matches:
        return None
    value = matches[-1].replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else str(number)


def classify(text: str) -> str:
    if "####" not in text:
        return "完全没有 #### 标记"
    if "<final numeric answer>" in text:
        return "复述 prompt 指令后跑飞"
    tail = re.findall(r"####\s*([^\n]{0,40})", text)
    last = tail[-1].strip() if tail else ""
    if re.search(r"\d", last) and re.search(r"[A-Za-z$]", last):
        return "答案带单位或货币符号"
    if re.search(r"(.)\1{20,}", text):
        return "退化成重复长串"
    if len(text) > 1500:
        return "超长续写"
    return "其他"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/analysis/e0-failure-report.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()]
    buckets: dict[str, list[dict]] = {"correct": [], "wrong": [], "extract_fail": []}
    for row in rows:
        prediction = row["prediction"]
        extracted = extract_gsm8k_answer(prediction)
        if extracted is None:
            buckets["extract_fail"].append(row)
        elif is_correct(prediction, row["reference"], "gsm8k"):
            buckets["correct"].append(row)
        else:
            buckets["wrong"].append(row)

    failures = buckets["extract_fail"]
    patterns = Counter(classify(row["prediction"]) for row in failures)
    rescued = [
        row for row in failures
        if loose_extract(row["prediction"]) == str(row["reference"]).strip()
    ]

    # 按输出长度分四层看正确率，验证「越长越容易跑飞」。
    lengths = sorted(((len(row["prediction"]), row) for row in rows), key=lambda pair: pair[0])
    quartile = max(1, len(lengths) // 4)
    by_length = []
    for index in range(4):
        chunk = [row for _, row in lengths[index * quartile: (index + 1) * quartile]]
        if not chunk:
            continue
        hit = sum(1 for row in chunk if is_correct(row["prediction"], row["reference"], "gsm8k"))
        by_length.append({
            "quartile": index + 1,
            "count": len(chunk),
            "max_chars": max(len(row["prediction"]) for row in chunk),
            "accuracy": round(hit / len(chunk), 4),
        })

    report = {
        "total": len(rows),
        "correct": len(buckets["correct"]),
        "wrong_but_extracted": len(buckets["wrong"]),
        "extraction_failures": len(failures),
        "accuracy": round(len(buckets["correct"]) / len(rows), 4),
        "failure_patterns": dict(patterns.most_common()),
        "loose_extraction_rescued": len(rescued),
        "loose_extraction_upper_bound_accuracy": round(
            (len(buckets["correct"]) + len(rescued)) / len(rows), 4
        ),
        "by_output_length_quartile": by_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cases = [
        {**row, "pattern": classify(row["prediction"])}
        for row in failures
    ]
    write_jsonl(args.output.with_suffix(".cases.jsonl"), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
