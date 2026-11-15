"""离线回放：量化「编造下一题」污染的真实规模，以及答案行锚定修复的效果。

不加载模型、不需要 GPU——直接对已落盘的采样候选重放截断逻辑，
回答三个问题：

1. 有多少候选跑飞到了第二道题（出现 ≥2 个 ``#### 数字``）？
2. 旧截断（只认对话标记）漏掉多少条？
3. 修正后判分变化多少——是救回，还是误伤？

用法：
    python scripts/replay_truncation_fix.py \
        --candidates experiments/sampling/gsm8k-dev-probe200-k8-t0.7.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import extract_gsm8k_answer, is_correct  # noqa: E402
from math_posttrain.prompts import (  # noqa: E402
    ANSWER_LINE_PATTERN,
    generation_stop_strings,
    truncate_generation,
)


def legacy_truncate(text: str) -> str:
    """修复前的截断：只按对话标记切，不看答案行。"""
    cut = len(text)
    for marker in generation_stop_strings():
        index = text.find(marker)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut].rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    polluted = 0
    changed_text = 0
    answer_changed = 0
    fixed_cases: list[dict] = []
    regressions: list[dict] = []

    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reference = str(record["reference"])
            for index, raw in enumerate(record["candidates"]):
                total += 1
                hits = len(ANSWER_LINE_PATTERN.findall(raw))
                if hits >= 2:
                    polluted += 1
                old_text = legacy_truncate(raw)
                new_text = truncate_generation(raw)
                if old_text != new_text:
                    changed_text += 1
                old_answer = extract_gsm8k_answer(old_text)
                new_answer = extract_gsm8k_answer(new_text)
                if old_answer != new_answer:
                    answer_changed += 1
                old_ok = is_correct(old_text, reference, "gsm8k")
                new_ok = is_correct(new_text, reference, "gsm8k")
                if not old_ok and new_ok:
                    fixed_cases.append(
                        {
                            "id": record["id"],
                            "candidate": index,
                            "reference": reference,
                            "old": old_answer,
                            "new": new_answer,
                        }
                    )
                elif old_ok and not new_ok:
                    regressions.append(
                        {
                            "id": record["id"],
                            "candidate": index,
                            "reference": reference,
                            "old": old_answer,
                            "new": new_answer,
                        }
                    )

    report = {
        "candidates": total,
        "polluted_multi_answer_line": polluted,
        "polluted_ratio": round(polluted / total, 4) if total else 0.0,
        "text_changed_by_anchor": changed_text,
        "answer_changed_by_anchor": answer_changed,
        "rescued_correct": len(fixed_cases),
        "regressed_correct": len(regressions),
        "rescued_examples": fixed_cases[:5],
        "regressed_examples": regressions[:5],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
