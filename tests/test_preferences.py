import pytest

from math_posttrain.preferences import build_dpo_records, build_rft_records


def candidates() -> list[dict]:
    return [
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "#### 2", "correct": True},
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "#### 3", "correct": False},
        {"id": "q2", "question": "2+2?", "reference": "4", "solution": "#### 4", "correct": True},
    ]


def test_rft_keeps_only_verified_correct_solutions() -> None:
    records = build_rft_records(candidates(), "gsm8k")
    assert len(records) == 2
    assert all(record["messages"][-1]["content"] in {"#### 2", "#### 4"} for record in records)


def test_dpo_pairs_same_question_correct_and_incorrect() -> None:
    records = build_dpo_records(candidates(), "gsm8k")
    assert len(records) == 1
    record = records[0]
    assert record["id"] == "q1"
    assert record["chosen"] == "#### 2"
    assert record["rejected"] == "#### 3"
    assert [message["role"] for message in record["prompt_messages"]] == ["system", "user"]


def test_mismatched_correct_flag_is_rejected() -> None:
    rows = [{"id": "q", "question": "1+1?", "reference": "2", "solution": "#### 2", "correct": False}]
    with pytest.raises(ValueError):
        build_rft_records(rows, "gsm8k")
