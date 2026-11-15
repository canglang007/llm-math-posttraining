from math_posttrain.prompts import (
    build_chat_messages,
    build_prompt,
    generation_stop_strings,
    resolve_eos_token_ids,
)
from math_posttrain.types import Example


class _FakeTokenizer:
    unk_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 151645}.get(token, 0)


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
    assert resolve_eos_token_ids(tokenizer, 151643) == [151643, 151645]
    assert resolve_eos_token_ids(tokenizer, [151643, 151645]) == [151643, 151645]
    assert resolve_eos_token_ids(tokenizer, None) == [151645]


def test_resolve_eos_token_ids_skips_unknown_token() -> None:
    assert resolve_eos_token_ids(_UnknownTokenizer(), 151643) == [151643]


def test_generation_stop_strings_cover_chat_continuation() -> None:
    stops = generation_stop_strings()
    assert "<|im_start|>" in stops
    assert "assistant\n" in stops
    assert "\n\nQuestion:" in stops
