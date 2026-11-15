import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_demo import build_record  # noqa: E402

ROW = {
    "id": "q1",
    "question": "Tom has 3 apples and gets 4 more. How many?",
    "reference": "7",
    "candidates": ["#### 7", "#### 7", "#### 7", "no answer", "#### 9", "#### 7", "#### 7", "#### 7"],
}


def test_build_record_computes_share_and_route() -> None:
    record = build_record(ROW, "gsm8k", greedy_ok=False)
    assert record["chosen"] == "7"
    assert record["vote_ok"] is True
    assert record["valid"] == 7
    assert record["share"] == round(6 / 7, 3)
    assert record["route"] == "auto"  # 6/7 ≈ 0.857 ≥ 0.75
    assert record["votes"]["7"] == 6
    assert record["greedy_ok"] is False


def test_build_record_marks_human_route_on_split_vote() -> None:
    row = dict(ROW)
    row["candidates"] = ["#### 7", "#### 9", "#### 9", "#### 5", "no answer", "#### 5", "#### 9", "#### 5"]
    record = build_record(row, "gsm8k", None)
    assert record["route"] == "human"  # 3/7 ≈ 0.43 < 0.75
    assert record["vote_ok"] is False  # 票数最多的是 9 / 5 中的最早出现者
    assert record["greedy_ok"] is None
