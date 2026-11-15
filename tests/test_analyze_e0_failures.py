import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_e0_failures import classify, loose_extract  # noqa: E402


def test_loose_extract_handles_currency_and_units() -> None:
    assert loose_extract("steps...\n#### $6") == "6"
    assert loose_extract("steps...\n#### 3weeks") == "3"
    assert loose_extract("no marker here") is None


def test_classify_identifies_prompt_echo_and_units() -> None:
    assert classify("... #### <final numeric answer>. assistant\n") == "复述 prompt 指令后跑飞"
    assert classify("steps...\n#### $6") == "答案带单位或货币符号"
    assert classify("no marker") == "完全没有 #### 标记"
