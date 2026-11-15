from math_posttrain.answers import (
    extract_boxed,
    extract_final_answer,
    extract_gsm8k_answer,
    is_correct,
    math_equivalent,
)


def test_gsm8k_extracts_first_answer() -> None:
    # 2026-11-15 变更：由「取最后一个 ####」改为「取第一个」。
    # 原因：SFT 后的模型答完不输出 <|im_end|>，会继续复述下一道题并给出它的
    # ####。取最后一个会把这题的判分静默换成下一题答案。官方 CoT 每题只有
    # 一个 ####，所以对 baseline 无影响（实测 200 题里 0 题出现多个 ####）。
    assert extract_gsm8k_answer("work #### 3 then #### 5") == "3"


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
