import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_sampling import (  # noqa: E402
    filter_candidates,
    per_question_curve,
    summarize,
    to_markdown,
    wilson_interval,
)

ROWS = [
    {
        "id": "q1",
        "reference": "18",
        "candidates": ["#### 18", "#### 18", "#### 20"],
    },
    {
        "id": "q2",
        "reference": "42",
        "candidates": ["no final answer", "#### 7", "#### 42"],
    },
]


def test_per_question_curve_uses_prefixes() -> None:
    curve = per_question_curve(ROWS[0]["candidates"], "18", "gsm8k", 3)
    assert curve[1]["maj"] is True
    assert curve[1]["oracle"] is True
    assert abs(curve[3]["sampled_pass@1"] - 2 / 3) < 1e-9
    # 两条 18 对一条 20，投票仍取 18
    assert curve[3]["maj"] is True


def test_vote_ignores_invalid_and_breaks_tie_by_first_seen() -> None:
    curve = per_question_curve(ROWS[1]["candidates"], "42", "gsm8k", 3)
    assert curve[1]["maj"] is False  # 唯一候选抽不出答案
    assert curve[2]["chosen"] == "7"
    assert curve[3]["chosen"] == "7"  # 平票取最早出现
    assert curve[3]["oracle"] is True


def test_summarize_aggregates_expected_rates() -> None:
    report = summarize(ROWS, dataset="gsm8k", max_k=3, baseline={"q1": True, "q2": False})
    by_k = {entry["k"]: entry for entry in report["curve"]}
    assert by_k[1]["maj@k"] == 0.5
    assert by_k[3]["maj@k"] == 0.5
    assert by_k[3]["oracle_pass@k"] == 1.0
    assert by_k[3]["maj@k_hits"] == 1
    assert report["avg_valid_candidates"] == 2.5
    assert abs(report["candidate_extraction_failure_rate"] - 1 / 6) < 1e-4
    assert report["baseline_comparison"]["greedy_hits"] == 1
    assert report["baseline_comparison"]["rescued_by_vote"] == 0
    assert report["baseline_comparison"]["hurt_by_vote"] == 0
    assert report["baseline_comparison"]["oracle_reachable_on_greedy_wrong"] == 1


def test_length_filter_drops_long_candidates() -> None:
    long_text = "#### 18" + "x" * 1200
    kept = filter_candidates(["#### 18", long_text, "no answer"], "gsm8k", max_chars=1000)
    assert kept == ["#### 18"]
    assert filter_candidates(["no answer"], "gsm8k") == []


def test_wilson_interval_brackets_point_estimate() -> None:
    low, high = wilson_interval(154, 200)
    assert 0.70 < low < 0.77 < high < 0.83
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_markdown_contains_curve_rows() -> None:
    report = summarize(ROWS, dataset="gsm8k", max_k=3)
    text = to_markdown(report)
    assert "| 3 |" in text
    assert "oracle pass@k" in text


def test_difficulty_strata_bins_by_reference_steps() -> None:
    from analyze_sampling import difficulty_strata

    per_question = [
        {"id": f"q{i}", "greedy_correct": i % 2 == 0, "maj_at_max_k": True, "oracle_at_max_k": True}
        for i in range(8)
    ]
    steps = {f"q{i}": i for i in range(8)}
    strata = difficulty_strata(per_question, steps, bins=4)
    assert len(strata) == 4
    assert [entry["n"] for entry in strata] == [2, 2, 2, 2]
    assert strata[0]["steps_min"] == 0 and strata[-1]["steps_max"] == 7
    # 每层 maj 都是 8/8，greedy 交错 1/2
    assert all(entry["maj_at_max_k"] == 1.0 for entry in strata)
    assert all(entry["delta_vs_greedy"] == 0.5 for entry in strata)


def test_load_step_counts_counts_calculator_annotations(tmp_path=None) -> None:
    import json
    import tempfile
    from pathlib import Path

    from analyze_sampling import load_step_counts

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write(json.dumps({"id": "a", "solution": "1+1=$<<1+1=2>>2\n#### 2"}) + "\n")
        handle.write(json.dumps({"id": "b", "solution": "no annotation #### 5"}) + "\n")
        path = Path(handle.name)
    counts = load_step_counts(path)
    assert counts == {"a": 1, "b": 0}


def test_consistency_analysis_emits_operating_points() -> None:
    from analyze_sampling import consistency_analysis

    rows = [
        {"id": "a", "reference": "7", "candidates": ["#### 7"] * 8},
        {"id": "b", "reference": "7", "candidates": ["#### 9"] * 4 + ["#### 7"] * 4},
    ]
    result = consistency_analysis(rows, "gsm8k", 8)
    assert result["buckets"][0]["label"] == "全票一致"
    assert result["buckets"][0]["n"] == 1
    points = {(point["share_min"], point["valid_min"]) for point in result["operating_points"]}
    assert (0.8, 7) in points
    strict = [p for p in result["operating_points"] if p["share_min"] == 1.0 and p["valid_min"] == 8][0]
    assert strict["coverage"] == 0.5
    assert strict["accuracy"] == 1.0
