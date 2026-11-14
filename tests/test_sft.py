from pathlib import Path

from math_posttrain.data import load_examples
from math_posttrain.sft import build_sft_record, build_sft_records
from math_posttrain.types import Example

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_sft_records_from_gsm8k() -> None:
    examples = load_examples(FIXTURES / "gsm8k_sample.jsonl", source="gsm8k", split="train")
    records = build_sft_records(examples, "gsm8k")
    assert len(records) == 2
    messages = records[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert "####" in messages[1]["content"]
    assert messages[2]["content"].startswith("3+2=5")


def test_build_sft_record_skips_empty_solution() -> None:
    example = Example("x", "question", "1", solution="")
    assert build_sft_record(example, "gsm8k") is None
