from collections.abc import Sequence
import re

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


CHAT_EOS_TOKENS = ("<|im_end|>", "<|im_start|>")
POST_GENERATION_STOP_STRINGS = ("<|im_start|>", "assistant\n", "\n\nQuestion:")


def resolve_eos_token_ids(tokenizer, base_eos_token_id=None) -> list[int]:
    """合并模型自带 eos 与 ChatML 特殊 token，让停止判断留在 CUDA 侧。

    Qwen3-4B-Base 的 generation_config 只把 ``<|endoftext|>``（151643）当 eos，
    用 chat template 生成时不会在 ``<|im_end|>``（151645）停下；把
    ``<|im_start|>``（151644）也并入，模型一续写下一轮对话就停。
    全部是 token id 比较，不给解码循环加 Python 字符串开销。
    """
    ids: list[int] = []
    if isinstance(base_eos_token_id, int):
        ids.append(base_eos_token_id)
    elif isinstance(base_eos_token_id, (list, tuple)):
        ids.extend(int(value) for value in base_eos_token_id)
    unk = getattr(tokenizer, "unk_token_id", None)
    for token in CHAT_EOS_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != unk and token_id not in ids:
            ids.append(token_id)
    return ids


def generation_stop_strings() -> list[str]:
    """交给 ``generate`` 的字符串停止词，尽早结束对话续写/题目复述。

    实测 Base 模型不会生成 ``<|im_start|>`` 特殊 token，token 级 eos 拦不住
    续写；必须用字符串停止词，否则跑飞的题会写满 max_new_tokens。
    """
    return list(POST_GENERATION_STOP_STRINGS)


ANSWER_LINE_PATTERN = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?[^\n]*")


def truncate_generation(text: str) -> str:
    """生成后再截断一次，兜住 stop_strings 未覆盖的边界，避免污染 ``####`` 答案。

    两层：

    1. 对话标记（``<|im_start|>`` / ``assistant\\n`` / ``\\n\\nQuestion:``）——
       模型开始扮演下一轮对话时立刻切掉；
    2. **答案行锚定**——模型不写任何标记、直接接着编下一道题时，第二个
       ``####`` 是唯一信号。GSM8K 官方 CoT 每题只在最后写一次 ``#### 数字``，
       所以第一个答案行之后的所有内容都属于跑飞，必须丢掉。

    第 2 层是本 bug 的根治点：``extract_gsm8k_answer`` 取第一个 ``####`` 只能
    保住判分，prediction 文本本身仍是脏的，会一路带进采样候选、maj@k 与
    DPO 的 chosen/rejected 配对。这里在源头把文本切干净。

    只扫描字符串，不在解码循环里执行。
    """
    cut = len(text)
    for marker in generation_stop_strings():
        index = text.find(marker)
        if index >= 0:
            cut = min(cut, index)
    text = text[:cut]
    match = ANSWER_LINE_PATTERN.search(text)
    if match:
        text = text[: match.end()]
    return text.rstrip()
