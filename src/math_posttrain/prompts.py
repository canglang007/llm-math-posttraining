from collections.abc import Sequence

from .types import Example

SYSTEM_PROMPT = "You are a careful math reasoning assistant. Show your steps and give a final answer."

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
