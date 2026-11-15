from math_posttrain.inference_time import (
    majority_vote,
    oracle_pass_at_k,
    sampling_cost,
    valid_candidate_count,
    vote_is_correct,
)


def test_majority_vote_picks_most_common_answer() -> None:
    candidates = ["steps...\n#### 18", "steps...\n#### 18", "steps...\n#### 20"]
    assert majority_vote(candidates, "gsm8k") == "18"


def test_majority_vote_ignores_unextractable_candidates() -> None:
    candidates = ["no answer here", "#### 7", "#### 7"]
    assert majority_vote(candidates, "gsm8k") == "7"
    assert valid_candidate_count(candidates, "gsm8k") == 2


def test_majority_vote_tie_is_deterministic() -> None:
    candidates = ["#### 3", "#### 5"]
    assert majority_vote(candidates, "gsm8k") == "3"
    assert majority_vote(list(reversed(candidates)), "gsm8k") == "5"


def test_majority_vote_returns_none_without_any_answer() -> None:
    assert majority_vote(["no marker", ""], "gsm8k") is None
    assert vote_is_correct(["no marker"], "18", "gsm8k") is False


def test_vote_and_oracle_against_public_answer() -> None:
    candidates = ["#### 20", "#### 18", "#### 18"]
    assert vote_is_correct(candidates, "18", "gsm8k") is True
    assert vote_is_correct(candidates, "20", "gsm8k") is False
    assert oracle_pass_at_k(candidates, "20", "gsm8k") is True
    assert oracle_pass_at_k(candidates, "99", "gsm8k") is False


def test_sampling_cost_scales_with_k() -> None:
    assert sampling_cost(200, 1, 200.0)["samples"] == 200
    assert sampling_cost(200, 8, 200.0)["samples"] == 1600
    assert sampling_cost(200, 8, 200.0)["estimated_new_tokens"] == 320000
