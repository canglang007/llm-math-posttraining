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


def multi_solution_candidates() -> list[dict]:
    # 注意：solution 在 _verified 里会 strip()，所以长度差异要体现在内容上，不能靠前导换行。
    return [
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "#### 2", "correct": True},
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "1+1=2\n#### 2", "correct": True},
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "let me think\n1+1=2\n#### 2", "correct": True},
        {"id": "q1", "question": "1+1?", "reference": "2", "solution": "#### 3", "correct": False},
        {"id": "q2", "question": "2+2?", "reference": "4", "solution": "#### 4", "correct": True},
    ]


def test_rft_default_keeps_all_correct_solutions() -> None:
    records = build_rft_records(multi_solution_candidates(), "gsm8k")
    assert len(records) == 4
    assert len({record["id"] for record in records}) == 2


def test_rft_max_per_question_dedups_by_question() -> None:
    records = build_rft_records(multi_solution_candidates(), "gsm8k", max_per_question=1)
    assert len(records) == 2
    assert {record["id"] for record in records} == {"q1", "q2"}
    assert len(records) == len({record["id"] for record in records})


def test_rft_max_per_question_picks_shortest_first() -> None:
    records = build_rft_records(multi_solution_candidates(), "gsm8k", max_per_question=2)
    q1 = [record for record in records if record["id"] == "q1"]
    assert [record["messages"][-1]["content"] for record in q1] == ["#### 2", "1+1=2\n#### 2"]


def test_rft_max_per_question_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_rft_records(multi_solution_candidates(), "gsm8k", max_per_question=0)


def test_rft_ignore_correctness_keeps_wrong_solutions() -> None:
    """A2 消融：不验证时错误解也会进训练集（模拟「不用验证器」）。"""
    records = build_rft_records(multi_solution_candidates(), "gsm8k", ignore_correctness=True)
    contents = [record["messages"][-1]["content"] for record in records]
    assert "#### 3" in contents  # q1 的错误解
    # q1 的 4 条候选 + q2 的 1 条，全部保留（不验证时不过滤）
    assert len(records) == 5


def test_rft_ignore_correctness_with_max_per_question_matches_verified_size() -> None:
    """同量对照：都按题取 1 条时，验证组与不验证组的题数一致，便于干净归因。"""
    verified = build_rft_records(multi_solution_candidates(), "gsm8k", max_per_question=1)
    unverified = build_rft_records(
        multi_solution_candidates(), "gsm8k", max_per_question=1, ignore_correctness=True
    )
    assert len(verified) == len(unverified) == 2
    assert {r["id"] for r in verified} == {r["id"] for r in unverified}
