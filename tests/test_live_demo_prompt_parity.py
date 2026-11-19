"""实时演示页与 Python 评测口径的一致性守卫。

前端把 system prompt、指令串、停止词硬编码在 JS 里（为了单文件可分发），
一旦 Python 侧改了这些常量而前端没跟上，演示出的答案就会和文档里的分数对不上。
这条测试把两边的字符串钉在一起。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from math_posttrain.prompts import GSM8K_INSTRUCTION, SYSTEM_PROMPT, generation_stop_strings  # noqa: E402

LIVE_HTML = ROOT / "demo" / "live.html"


def test_live_demo_exists() -> None:
    assert LIVE_HTML.exists(), "demo/live.html 缺失"


def test_system_prompt_matches_python() -> None:
    html = LIVE_HTML.read_text(encoding="utf-8")
    assert f'const SYSTEM = "{SYSTEM_PROMPT}";' in html


def test_instruction_matches_python() -> None:
    html = LIVE_HTML.read_text(encoding="utf-8")
    assert f'const INSTRUCTION = "{GSM8K_INSTRUCTION}";' in html


def test_stop_strings_match_python() -> None:
    html = LIVE_HTML.read_text(encoding="utf-8")
    # JS 源码里换行写成 \n 转义；Python 侧是真实换行，这里按源码形态比对
    assert "const STOPS = [" in html
    for stop in generation_stop_strings():
        escaped = stop.replace("\n", "\\n")
        assert f'"{escaped}"' in html, f"停止词 {stop!r} 未在前端同步"


def test_prompt_template_shape() -> None:
    html = LIVE_HTML.read_text(encoding="utf-8")
    # ChatML 三段式：system → user → assistant 生成前缀
    assert "`<|im_start|>system\\n${SYSTEM}<|im_end|>\\n`" in html
    assert "`<|im_start|>user\\n${question}\\n\\n${INSTRUCTION}<|im_end|>\\n`" in html
    assert "`<|im_start|>assistant\\n`" in html


def test_greedy_decoding_is_requested() -> None:
    html = LIVE_HTML.read_text(encoding="utf-8")
    assert "temperature: 0" in html
    assert "max_tokens: 512" in html


def test_stop_token_ids_are_sent() -> None:
    """线上请求必须同时给 stop_token_ids。

    踩过的坑：skip_special_tokens=true 会把 <|im_start|> 从解码文本里剥掉，
    只靠 stop 字符串在 <|im_end|> 处停不下来，模型续写下一轮对话后答案抽取错位
    （实测 SFT 从 90% 掉到会错抓 ####）。Qwen3 ChatML 的 id 是固定的：
    <|im_end|>=151645、<|im_start|>=151644。
    """
    html = LIVE_HTML.read_text(encoding="utf-8")
    assert "const STOP_TOKEN_IDS = [151645, 151644];" in html
    assert "stop_token_ids: STOP_TOKEN_IDS" in html
    assert "include_stop_str_in_output: false" in html
