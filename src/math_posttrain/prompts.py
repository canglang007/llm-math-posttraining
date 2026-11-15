from collections.abc import Sequence

from .types import Example

SYSTEM_PROMPT = "You are a careful math reasoning assistant. Show your steps and give a final answer."

QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

GSM8K_INSTRUCTION = "Solve the problem step by step. On the last line write: #### <final numeric answer>."
MATH_INSTRUCTION = "Solve the problem step by step. Put the final answer in \\boxed{}."


def build_prompt(question: str, dataset: str, few_shot: Sequence[Example] = ()) -> str:
    if dataset not in {"gsm8k", "math"}:
        raise ValueError(f"unknown dataset: {dataset}")
    blocks = [SYSTEM_PROMPT]
    for example in few_shot:
        answer = example.solution or example.answer
        blocks.append(f"Question: {example.question}\nAnswer: {answer}")
    blocks.append(f"Question: {question}")
    blocks.append(GSM8K_INSTRUCTION if dataset == "gsm8k" else MATH_INSTRUCTION)
    return "\n\n".join(blocks)


def build_chat_messages(question: str, dataset: str, few_shot: Sequence[Example] = ()) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in few_shot:
        messages.append({"role": "user", "content": example.question})
        messages.append({"role": "assistant", "content": example.solution or example.answer})
    messages.append({"role": "user", "content": question})
    instruction = GSM8K_INSTRUCTION if dataset == "gsm8k" else MATH_INSTRUCTION
    messages[-1]["content"] = f"{question}\n\n{instruction}"
    return messages


CHAT_EOS_TOKEN = "<|im_end|>"
GENERATION_STOP_STRINGS = ("<|im_start|>", "assistant\n", "\n\nQuestion:")


def resolve_eos_token_ids(tokenizer, base_eos_token_id=None) -> list[int]:
    """合并模型自带 eos 与 ChatML 的 ``<|im_end|>``。

    Qwen3-4B-Base 的 generation_config 只把 ``<|endoftext|>``（151643）当 eos，
    用 chat template 生成时不会在 ``<|im_end|>``（151645）停下，会一直写到
    max_new_tokens 并续写下一轮对话，污染答案抽取。
    """
    ids: list[int] = []
    if isinstance(base_eos_token_id, int):
        ids.append(base_eos_token_id)
    elif isinstance(base_eos_token_id, (list, tuple)):
        ids.extend(int(value) for value in base_eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids(CHAT_EOS_TOKEN)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(im_end, int) and im_end != unk and im_end not in ids:
        ids.append(im_end)
    return ids


def generation_stop_strings() -> list[str]:
    """遇到对话续写或题目复述就停，避免 Base 模型跑飞后覆盖已写出的答案。"""
    return list(GENERATION_STOP_STRINGS)
