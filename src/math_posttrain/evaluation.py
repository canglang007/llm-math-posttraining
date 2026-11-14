from collections.abc import Iterable

from .answers import extract_final_answer, is_correct
from .types import EvalSummary


def evaluate(rows: Iterable[dict], dataset: str) -> EvalSummary:
    total = 0
    correct = 0
    extraction_failures = 0
    by_category: dict[str, dict[str, float]] = {}
    for row in rows:
        total += 1
        prediction = str(row.get("prediction", ""))
        reference = str(row.get("reference", ""))
        category = str(row.get("category") or "all")
        extracted = extract_final_answer(prediction, dataset)
        ok = extracted is not None and is_correct(prediction, reference, dataset)
        if extracted is None:
            extraction_failures += 1
        correct += int(ok)
        bucket = by_category.setdefault(
            category, {"total": 0, "correct": 0, "extraction_failures": 0, "accuracy": 0.0}
        )
        bucket["total"] += 1
        bucket["correct"] += int(ok)
        bucket["extraction_failures"] += int(extracted is None)
    for bucket in by_category.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
    return EvalSummary(
        dataset=dataset,
        total=total,
        correct=correct,
        extraction_failures=extraction_failures,
        accuracy=round(correct / total, 4) if total else 0.0,
        by_category=by_category,
    )
