"""评测脚本两条推理后端的口径守卫。

守的是「换引擎不换尺子」：vLLM 后端的采样语义必须与 HF 路径等价
（greedy、同一组停止词、输出不含停止串），prompt 渲染必须走同一个 chat template。
任何一条破了，vLLM 与 HF 的分数就不可比，校准数据也会失真。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_model import (  # noqa: E402
    BACKENDS,
    adapter_rank,
    render_prompts,
    require_vllm,
    vllm_sampling_kwargs,
)

from math_posttrain.prompts import (  # noqa: E402
    GSM8K_INSTRUCTION,
    SYSTEM_PROMPT,
    generation_stop_strings,
)


class RecordingTokenizer:
    """记录 apply_chat_template 收到的 messages，返回可辨识的渲染串。"""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def make_example(question: str) -> SimpleNamespace:
    return SimpleNamespace(question=question)


def test_backends_include_hf_and_vllm() -> None:
    assert BACKENDS == ("hf", "vllm")


def test_vllm_sampling_is_greedy_and_reuses_hf_stop_conditions() -> None:
    kwargs = vllm_sampling_kwargs(max_new_tokens=512, eos_ids=[151643, 151645])
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 512
    assert kwargs["stop"] == generation_stop_strings()
    assert kwargs["stop_token_ids"] == [151643, 151645]
    # HF 侧的 stop_strings 不保留停止串，vLLM 必须一致，否则截断行为不同
    assert kwargs["include_stop_str_in_output"] is False
    assert kwargs["skip_special_tokens"] is True


def test_vllm_sampling_kwargs_does_not_alias_mutable_input() -> None:
    eos_ids = [1, 2]
    kwargs = vllm_sampling_kwargs(max_new_tokens=8, eos_ids=eos_ids)
    eos_ids.append(3)
    assert kwargs["stop_token_ids"] == [1, 2]


def test_render_prompts_uses_chat_messages_with_instruction() -> None:
    tokenizer = RecordingTokenizer()
    examples = [make_example("1+1=?"), make_example("2+2=?")]
    prompts = render_prompts(examples, tokenizer, "gsm8k")

    assert len(prompts) == 2
    assert len(tokenizer.calls) == 2
    first = tokenizer.calls[0]
    assert first[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert first[-1]["role"] == "user"
    assert "1+1=?" in first[-1]["content"]
    assert GSM8K_INSTRUCTION in first[-1]["content"]
    assert "2+2=?" in prompts[1]


def test_adapter_rank_reads_config(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
    assert adapter_rank(str(adapter)) == 16


def test_adapter_rank_defaults_to_eight(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"alpha": 16}))
    assert adapter_rank(str(adapter)) == 8


def test_require_vllm_gives_readable_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules 里放 None 会让 `from vllm import ...` 抛 ImportError，
    # 这样即使开发机装了 vllm 也能稳定测到兜底分支。
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(RuntimeError, match="--backend vllm"):
        require_vllm()
