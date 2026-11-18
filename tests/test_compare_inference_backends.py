"""口径校准脚本的判定逻辑。

守的是「换引擎前先量尺子」：一致率、答案一致率、翻转明细都必须算对，
否则校准报告本身会误导后续决策。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_inference_backends import compare_rows  # noqa: E402


def row(qid: str, prediction: str, reference: str = "18") -> dict:
    return {"id": qid, "prediction": prediction, "reference": reference}


def test_identical_backends_have_full_agreement() -> None:
    rows = [row("q1", "so 18\n#### 18"), row("q2", "#### 7", "7")]
    stats = compare_rows(rows, rows, "gsm8k")
    assert stats["grade_agreement_rate"] == 1.0
    assert stats["answer_agreement_rate"] == 1.0
    assert stats["flips"] == []


def test_flip_direction_is_reported() -> None:
    baseline = [row("q1", "#### 18"), row("q2", "#### 20")]
    candidate = [row("q1", "#### 19"), row("q2", "#### 18")]
    stats = compare_rows(baseline, candidate, "gsm8k")
    assert stats["baseline_correct"] == 1
    assert stats["candidate_correct"] == 1
    assert stats["grade_agreement_rate"] == 0.0
    directions = {flip["id"]: flip["direction"] for flip in stats["flips"]}
    assert directions == {"q1": "baseline_only", "q2": "candidate_only"}


def test_same_grade_but_different_answer_counts_as_disagreement_on_answer() -> None:
    # 两题都判错，但错法不同：判分一致、答案不一致
    baseline = [row("q1", "#### 5")]
    candidate = [row("q1", "#### 6")]
    stats = compare_rows(baseline, candidate, "gsm8k")
    assert stats["grade_agreement_rate"] == 1.0
    assert stats["answer_agreement_rate"] == 0.0
    assert stats["flips"] == []


def test_unmatched_ids_are_counted_not_dropped() -> None:
    baseline = [row("q1", "#### 18"), row("q2", "#### 18")]
    candidate = [row("q1", "#### 18"), row("q3", "#### 18")]
    stats = compare_rows(baseline, candidate, "gsm8k")
    assert stats["common"] == 1
    assert stats["baseline_only"] == 1
    assert stats["candidate_only"] == 1


def test_empty_overlap_raises() -> None:
    with pytest.raises(ValueError):
        compare_rows([row("q1", "#### 18")], [row("q2", "#### 18")], "gsm8k")
