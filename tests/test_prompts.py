from math_posttrain.prompts import build_chat_messages, build_prompt
from math_posttrain.types import Example


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
