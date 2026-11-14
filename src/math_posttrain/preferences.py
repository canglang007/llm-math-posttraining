from collections import defaultdict
from collections.abc import Iterable

from .answers import is_correct
from .prompts import build_chat_messages


def _verified(candidate: dict, dataset: str) -> tuple[bool, str]:
    solution = str(candidate.get("solution", "")).strip()
    reference = str(candidate.get("reference", ""))
    if not solution or not reference:
        return False, solution
    correct = is_correct(solution, reference, dataset)
    if "correct" in candidate and bool(candidate["correct"]) != correct:
        raise ValueError(f"candidate {candidate.get('id')}: correct flag disagrees with answer check")
    return correct, solution


def build_rft_records(candidates: Iterable[dict], dataset: str) -> list[dict]:
    records = []
    for candidate in candidates:
        correct, solution = _verified(candidate, dataset)
        if not correct:
            continue
        messages = build_chat_messages(str(candidate["question"]), dataset)
        messages.append({"role": "assistant", "content": solution})
        records.append(
            {
                "id": str(candidate.get("id", "")),
                "source": candidate.get("source", dataset),
                "split": candidate.get("split", "train"),
                "category": candidate.get("category"),
                "level": candidate.get("level"),
                "answer": candidate.get("reference"),
                "messages": messages,
            }
        )
    return records


def build_dpo_records(candidates: Iterable[dict], dataset: str) -> list[dict]:
    groups: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"correct": [], "incorrect": []}
    )
    for candidate in candidates:
        key = (
            str(candidate.get("id", "")),
            str(candidate["question"]),
            str(candidate["reference"]),
        )
        correct, _ = _verified(candidate, dataset)
        groups[key]["correct" if correct else "incorrect"].append(candidate)

    records = []
    for (candidate_id, question, reference), group in groups.items():
        if not group["correct"] or not group["incorrect"]:
            continue
        chosen = min(group["correct"], key=lambda item: (len(str(item["solution"])), str(item["solution"])))
        rejected = min(group["incorrect"], key=lambda item: (len(str(item["solution"])), str(item["solution"])))
        records.append(
            {
                "id": candidate_id,
                "source": chosen.get("source", dataset),
                "split": chosen.get("split", "train"),
                "category": chosen.get("category"),
                "level": chosen.get("level"),
                "reference": reference,
                "prompt_messages": build_chat_messages(question, dataset),
                "chosen": str(chosen["solution"]).strip(),
                "rejected": str(rejected["solution"]).strip(),
            }
        )
    return records
