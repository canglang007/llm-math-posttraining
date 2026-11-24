"""对比 demo 的选题逻辑。

守的是「demo 展示的必须是真分叉题」：如果选题函数把同对同错的题也选进来，
页面就会变成虚假对比，核验时无法对齐原始结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_comparison_demo import pick  # noqa: E402


def row(prediction: str, reference: str = "18") -> dict:
    return {"prediction": prediction, "reference": reference}


def test_pick_returns_only_disagreements_and_labels_direction() -> None:
    runs = {
        "a": {"q1": row("#### 18"), "q2": row("#### 5"), "q3": row("#### 18"), "q4": row("#### 9", "7")},
        "b": {"q1": row("#### 5"), "q2": row("#### 18"), "q3": row("#### 18"), "q4": row("#### 8", "7")},
    }
    picked = pick(runs, "a", "b", limit=5)
    directions = dict(picked)
    # q3 同对、q4 同错 → 不应入选
    assert set(directions) == {"q1", "q2"}
    assert directions["q1"] == "a_only"
    assert directions["q2"] == "b_only"


def test_pick_respects_limit_per_direction() -> None:
    runs = {
        "a": {f"q{i}": row("#### 18") for i in range(1, 8)},
        "b": {f"q{i}": row("#### 1") for i in range(1, 8)},
    }
    assert len(pick(runs, "a", "b", limit=3)) == 3


def test_pick_handles_missing_overlap() -> None:
    runs = {"a": {"q1": row("#### 18")}, "b": {"q2": row("#### 18")}}
    assert pick(runs, "a", "b", limit=3) == []
