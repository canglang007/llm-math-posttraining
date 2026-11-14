from collections.abc import Iterable
from typing import Any

from .prompts import build_chat_messages
from .types import Example


def encode_sft_example(messages: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> dict[str, list[int]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with an assistant turn")
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
    prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)
    input_ids = list(full["input_ids"])
    prompt_ids = list(prompt["input_ids"])
    if not input_ids or not prompt_ids:
        raise ValueError("empty tokenized example")
    if len(prompt_ids) >= len(input_ids):
        raise ValueError("no assistant tokens after truncation; increase max_seq_len")
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not a prefix of the full conversation; check the chat template")
    labels = list(input_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


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
