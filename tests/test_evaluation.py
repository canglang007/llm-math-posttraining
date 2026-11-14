from math_posttrain.evaluation import evaluate


def test_evaluate_counts_correct_and_extraction_failures() -> None:
    rows = [
        {"prediction": "#### 5", "reference": "5", "category": "easy"},
        {"prediction": "#### 6", "reference": "5", "category": "easy"},
        {"prediction": "I do not know", "reference": "7", "category": "hard"},
    ]
    summary = evaluate(rows, "gsm8k")
    assert summary.total == 3
    assert summary.correct == 1
    assert summary.extraction_failures == 1
    assert summary.accuracy == 0.3333
    assert summary.by_category["easy"]["total"] == 2
    assert summary.by_category["easy"]["accuracy"] == 0.5
    assert summary.by_category["hard"]["accuracy"] == 0.0
