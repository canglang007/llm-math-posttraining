from pathlib import Path

from math_posttrain.data import load_examples, read_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_gsm8k_fixture() -> None:
    examples = load_examples(FIXTURES / "gsm8k_sample.jsonl", source="gsm8k", split="train")
    assert len(examples) == 2
    assert examples[0].id == "g0"
    assert examples[0].answer == "5"
    assert examples[0].source == "gsm8k"


def test_load_math_fixture_keeps_category() -> None:
    examples = load_examples(FIXTURES / "math_sample.jsonl", source="math", split="test")
    assert examples[1].category == "algebra"
    assert examples[1].answer == "\\frac{1}{2}"


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
