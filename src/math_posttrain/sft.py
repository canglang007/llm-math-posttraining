from collections.abc import Iterable
from typing import Any

from .prompts import build_chat_messages
from .types import Example

# Qwen3-Base 预训练用 <|endoftext|>(151643) 作文档结束符，lm_head 那一行是
# 训练过的；而 ChatML 的 <|im_end|>(151645) 在 Base 预训练中从未出现，
# 它的输出投影是随机的。
#
# 实测（scripts/diag_eos_loss.py，答案行之后那一个位置）：
#   Base      : <|endoftext|> rank 1     概率 56.7%   <|im_end|> rank 14577
#   SFT 200步 : <|endoftext|> rank 151598 概率 0     <|im_end|> rank 1949
#
# 也就是说 Base 本来就会停，是我们的训练目标把它压下去了，同时又推不上
# 一个没学过的 token，结果两边都不会。给 <|im_end|> 加权 30 训 40 步，
# rank 只从 1982 动到 1949——因为瓶颈在 lm_head 而不在梯度大小。
#
# 所以训练目标改用模型自己学过的 <|endoftext|> 收尾；推理侧
# resolve_eos_token_ids 本来就把 base eos 包含在内，无需改动就能停。
TRAIN_STOP_TOKEN = "<|endoftext|>"
CHAT_TURN_END = "<|im_end|>"


def encode_sft_example(messages: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> dict[str, list[int]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages must end with an assistant turn")
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    # 只替换最后一处 <|im_end|>（assistant 段的收尾），前面的 system/user 轮次
    # 保持 ChatML 原样，避免破坏结构。模板里没有 <|im_end|> 时（如测试用的
    # 假 tokenizer）原样返回。
    turn_end = full_text.rfind(CHAT_TURN_END)
    if turn_end >= 0:
        full_text = full_text[:turn_end] + TRAIN_STOP_TOKEN + full_text[turn_end + len(CHAT_TURN_END) :]
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
