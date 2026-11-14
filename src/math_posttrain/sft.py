from collections.abc import Iterable

from .prompts import build_chat_messages
from .types import Example


def build_sft_record(example: Example, dataset: str) -> dict | None:
    solution = (example.solution or "").strip()
    if not solution:
        return None
    messages = build_chat_messages(example.question, dataset)
    messages.append({"role": "assistant", "content": solution})
    return {
        "id": example.id,
        "source": example.source,
        "split": example.split,
        "category": example.category,
        "level": example.level,
        "answer": example.answer,
        "messages": messages,
    }


def build_sft_records(examples: Iterable[Example], dataset: str) -> list[dict]:
    records = []
    for example in examples:
        record = build_sft_record(example, dataset)
        if record is not None:
            records.append(record)
    return records
