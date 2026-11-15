from math_posttrain.answers import extract_gsm8k_answer
from math_posttrain.prompts import (
    build_chat_messages,
    build_prompt,
    generation_stop_strings,
    resolve_eos_token_ids,
    truncate_generation,
)
from math_posttrain.types import Example


class _FakeTokenizer:
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 151645, "<|im_start|>": 151644}.get(token, 0)


class _UnknownTokenizer(_FakeTokenizer):
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 0


def test_prompt_contains_dataset_instruction() -> None:
    assert "####" in build_prompt("Q", "gsm8k")
    assert "\\boxed{}" in build_prompt("Q", "math")


def test_few_shot_examples_precede_question() -> None:
    examples = [Example("a", "Q0", "A0"), Example("b", "Q1", "A1")]
    prompt = build_prompt("Q2", "gsm8k", examples)
    assert prompt.index("Q0") < prompt.index("Q1") < prompt.index("Q2")


def test_chat_messages_put_instruction_with_question() -> None:
    messages = build_chat_messages("What is 1+1?", "gsm8k")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "What is 1+1?" in messages[-1]["content"]
    assert "####" in messages[-1]["content"]


def test_resolve_eos_token_ids_merges_chat_eos() -> None:
    tokenizer = _FakeTokenizer()
    assert resolve_eos_token_ids(tokenizer, 151643) == [151643, 151645, 151644]
    assert resolve_eos_token_ids(tokenizer, [151643, 151645]) == [151643, 151645, 151644]
    assert resolve_eos_token_ids(tokenizer, None) == [151645, 151644]


def test_resolve_eos_token_ids_skips_unknown_token() -> None:
    assert resolve_eos_token_ids(_UnknownTokenizer(), 151643) == [151643]


def test_truncate_generation_cuts_chat_continuation() -> None:
    text = "Step 1\n#### 18\n\nassistant\nNext question..."
    assert truncate_generation(text) == "Step 1\n#### 18"
    assert truncate_generation("clean answer #### 7") == "clean answer #### 7"


def test_truncate_generation_uses_earliest_marker() -> None:
    text = "ok\n#### 1\n\nQuestion: X\nassistant\n"
    assert truncate_generation(text) == "ok\n#### 1"


def test_truncate_generation_cuts_unmarked_next_question() -> None:
    """模型不写任何对话标记、直接接着编下一道题时，第二个 ``####`` 是唯一信号。"""
    text = (
        "He presses 15 x 10 = 150 pounds.\n#### 450\n\n"
        "A bakery sells 5 cakes a day for 7 days. How many cakes? #### 35"
    )
    truncated = truncate_generation(text)
    assert truncated == "He presses 15 x 10 = 150 pounds.\n#### 450"
    # 判分与文本必须同时干净：截断后只剩本题答案
    assert extract_gsm8k_answer(truncated) == "450"


def test_truncate_generation_keeps_answer_line_unit() -> None:
    """答案行本身带的单位/货币符号不能被误切，否则会把 6$ 变成 6。"""
    text = "so the total is\n#### 6$\n\nNext: 3 more apples #### 9"
    assert truncate_generation(text) == "so the total is\n#### 6$"


def test_generation_stop_strings_cover_chat_continuation() -> None:
    stops = generation_stop_strings()
    assert "<|im_start|>" in stops
    assert "assistant\n" in stops
    assert "\n\nQuestion:" in stops
