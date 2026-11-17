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


def _rft_record(candidate: dict, solution: str, dataset: str) -> dict:
    messages = build_chat_messages(str(candidate["question"]), dataset)
    messages.append({"role": "assistant", "content": solution})
    return {
        "id": str(candidate.get("id", "")),
        "source": candidate.get("source", dataset),
        "split": candidate.get("split", "train"),
        "category": candidate.get("category"),
        "level": candidate.get("level"),
        "answer": candidate.get("reference"),
        "messages": messages,
    }


def build_rft_records(
    candidates: Iterable[dict],
    dataset: str,
    max_per_question: int | None = None,
) -> list[dict]:
    """构造 RFT 记录（只保留答案验证正确的解）。

    max_per_question=None：保留全部正确解（默认，兼容旧行为）。
    给定 N：每题最多取 N 条正确解，按「先短后长、再按文本」排序取前 N，确定性可复现。

    为什么需要这个参数：同一道题的 k 条候选可能全部正确，全留下会让「条数」远大于
    「题数」（实测 120 题 × k=4 产出 412 条、实际只有 114 题），
    训练时同题被反复喂 3~4 遍。要按题去重就用 max_per_question=1。
    """
    if max_per_question is None:
        records = []
        for candidate in candidates:
            correct, solution = _verified(candidate, dataset)
            if correct:
                records.append(_rft_record(candidate, solution, dataset))
        return records

    if max_per_question < 1:
        raise ValueError("max_per_question must be >= 1")

    grouped: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for candidate in candidates:
        correct, solution = _verified(candidate, dataset)
        if not correct:
            continue
        grouped[str(candidate.get("id", ""))].append((candidate, solution))

    records = []
    for items in grouped.values():
        chosen = sorted(items, key=lambda item: (len(item[1]), item[1]))[:max_per_question]
        for candidate, solution in chosen:
            records.append(_rft_record(candidate, solution, dataset))
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
