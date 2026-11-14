from math_posttrain.answers import (
    extract_boxed,
    extract_final_answer,
    extract_gsm8k_answer,
    is_correct,
    math_equivalent,
)


def test_gsm8k_extracts_last_answer() -> None:
    assert extract_gsm8k_answer("work #### 3 then #### 5") == "5"


def test_gsm8k_handles_commas_and_trailing_zero() -> None:
    assert extract_gsm8k_answer("#### 1,234") == "1234"
    assert extract_gsm8k_answer("#### 42.0") == "42"


def test_gsm8k_extraction_failure() -> None:
    assert extract_gsm8k_answer("no final marker") is None


def test_boxed_handles_nested_braces() -> None:
    assert extract_boxed("answer \\boxed{\\frac{1}{2} + \\frac{1}{3}}") == "\\frac{1}{2} + \\frac{1}{3}"


def test_math_equivalent_normalizes_fraction_commands() -> None:
    assert math_equivalent("\\frac{1}{2}", "\\dfrac{1}{2}")


def test_is_correct_gsm8k_and_math() -> None:
    assert is_correct("so the answer is #### 5", "5", "gsm8k")
    assert not is_correct("so the answer is #### 6", "5", "gsm8k")
    assert is_correct("answer \\boxed{42}", "42", "math")
    assert not is_correct("answer \\boxed{41}", "42", "math")


def test_extract_final_answer_rejects_unknown_dataset() -> None:
    try:
        extract_final_answer("#### 1", "unknown")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
