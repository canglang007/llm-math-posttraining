import pytest

from math_posttrain.sft import encode_sft_example


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{message['role']}:{message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [(ord(char) % 90) + 1 for char in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "#### 2"},
    ]


def test_encode_masks_prompt_and_keeps_assistant_tokens() -> None:
    tokenizer = FakeTokenizer()
    encoded = encode_sft_example(messages(), tokenizer, max_seq_len=512)
    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert encoded["labels"][-1] != -100
    assert any(label != -100 for label in encoded["labels"])
    assert encoded["labels"][0] == -100


def test_encode_rejects_non_assistant_last_turn() -> None:
    with pytest.raises(ValueError):
        encode_sft_example(messages()[:-1], FakeTokenizer(), max_seq_len=512)


def test_encode_rejects_truncation_without_assistant_tokens() -> None:
    with pytest.raises(ValueError):
        encode_sft_example(messages(), FakeTokenizer(), max_seq_len=5)
