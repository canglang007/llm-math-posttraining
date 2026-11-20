"""本地服务（deploy/local_server.py）的纯逻辑单测。

不加载模型、不 import torch：只守住「配置解析 / adapter 过滤 / 响应格式 /
greedy 守卫 / 跨域头」这些容易悄悄写错的地方。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import pytest  # noqa: E402

import local_server as ls  # noqa: E402


def test_parse_adapter_spec_basic() -> None:
    got = ls.parse_adapter_spec("sft=/a, a2=/b")
    assert got == {"sft": "/a", "a2": "/b"}


def test_parse_adapter_spec_empty() -> None:
    assert ls.parse_adapter_spec(None) == {}
    assert ls.parse_adapter_spec("") == {}


def test_parse_adapter_spec_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        ls.parse_adapter_spec("sft")
    with pytest.raises(ValueError):
        ls.parse_adapter_spec("=/a")


def test_resolve_adapters_skips_missing(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "adapter_config.json").write_text("{}", encoding="utf-8")
    warnings: list[str] = []
    usable = ls.resolve_adapters(
        {"good": str(good), "bad": str(tmp_path / "nope")},
        tmp_path,
        warn=warnings.append,
    )
    assert usable == {"good": str(good)}
    assert len(warnings) == 1 and "bad" in warnings[0]


def test_resolve_adapters_relative_to_root(tmp_path: Path) -> None:
    rel = tmp_path / "experiments" / "x"
    rel.mkdir(parents=True)
    (rel / "adapter_config.json").write_text("{}", encoding="utf-8")
    usable = ls.resolve_adapters({"x": "experiments/x"}, tmp_path, warn=lambda *_: None)
    assert usable == {"x": str(rel)}


def test_check_sampling_allows_greedy_only() -> None:
    assert ls.check_sampling({}) is None
    assert ls.check_sampling({"temperature": 0}) is None
    assert ls.check_sampling({"temperature": 0.0}) is None
    assert ls.check_sampling({"temperature": 0.7}) is not None
    assert ls.check_sampling({"temperature": 1}) is not None


def test_completion_response_shape() -> None:
    got = ls.build_completion_response("sft", "#### 78", 12, 34)
    assert got["object"] == "text_completion"
    assert got["model"] == "sft"
    assert got["choices"][0]["text"] == "#### 78"
    assert got["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }


def test_models_response_lists_names() -> None:
    got = ls.build_models_response(["base", "sft"])
    assert [m["id"] for m in got["data"]] == ["base", "sft"]
    assert got["object"] == "list"


def test_cors_headers_allow_null_origin() -> None:
    """本地页面从 file:// 打开，Origin 是 null，必须允许任意来源。"""
    headers = ls.cors_headers()
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in headers["Access-Control-Allow-Methods"]


def test_default_adapters_match_artifact_layout() -> None:
    """默认映射必须指向仓库里真实的目录布局（写错路径会让本地模式静默失去模型）。"""
    assert set(ls.DEFAULT_ADAPTERS) == {"sft", "e2b", "a2", "dpo"}
    assert ls.DEFAULT_ADAPTERS["a2"].endswith("qwen3-4b-gsm8k-rft-a2nf-300q")
    for path in ls.DEFAULT_ADAPTERS.values():
        assert path.startswith("experiments/")


# ---------------------------------------------------------------- HTTP 往返（桩引擎，不加载模型）


class _StubEngine:
    """只实现 handler 用到的两个方法，用来验证路由/校验/CORS，不碰 torch。"""

    def models(self) -> list[str]:
        return ["base", "sft"]

    def generate(self, model_name: str, prompt: str, max_new_tokens=None) -> dict:
        return {
            "text": f"#### 78 ({model_name})",
            "prompt_tokens": len(prompt),
            "completion_tokens": 3,
        }


@pytest.fixture()
def server_url():
    import threading
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ls.make_handler(_StubEngine()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    import json as _json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url,
        data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json.loads(exc.read())


def test_models_endpoint(server_url: str) -> None:
    import json as _json
    import urllib.request

    with urllib.request.urlopen(f"{server_url}/v1/models", timeout=5) as resp:
        assert resp.status == 200
        body = _json.loads(resp.read())
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert [m["id"] for m in body["data"]] == ["base", "sft"]


def test_completion_endpoint(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/v1/completions",
        {"model": "sft", "prompt": "<|im_start|>assistant\n", "max_tokens": 8},
    )
    assert status == 200
    assert body["choices"][0]["text"].startswith("#### 78")
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 3


def test_completion_rejects_non_greedy(server_url: str) -> None:
    status, body = _post(
        f"{server_url}/v1/completions",
        {"model": "sft", "prompt": "x", "temperature": 0.7},
    )
    assert status == 400
    assert "greedy" in body["error"]["message"]


def test_completion_unknown_model(server_url: str) -> None:
    status, body = _post(f"{server_url}/v1/completions", {"model": "nope", "prompt": "x"})
    assert status == 404
    assert "unknown model" in body["error"]["message"]


def test_options_preflight(server_url: str) -> None:
    import urllib.request

    req = urllib.request.Request(f"{server_url}/v1/completions", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in resp.headers["Access-Control-Allow-Methods"]


def test_main_uses_default_adapters_when_flag_absent(tmp_path, monkeypatch, capsys) -> None:
    """不给 --adapters 时应走内置映射（曾因空 dict 覆盖默认值导致「无可用 adapter」）。"""
    rel = tmp_path / "experiments" / "rft" / "x"
    rel.mkdir(parents=True)
    (rel / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ls, "DEFAULT_ADAPTERS", {"x": str(rel)})
    monkeypatch.setattr(ls, "PROJECT_ROOT", tmp_path)
    assert ls.main(["--dry-run", "--adapters", ""]) == 0          # 空串 → 只跑 base
    assert "（无，只能跑 base）" in capsys.readouterr().out
    assert ls.main(["--dry-run"]) == 0                            # 缺省 → 用默认映射
    assert "['x']" in capsys.readouterr().out


# ---------------------------------------------------------------- adapter 作用域


class _FakePeft:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.disabled = 0

    def disable_adapter(self):
        self.disabled += 1
        outer = self

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()

    def set_adapter(self, name: str) -> None:
        self.calls.append(name)


def test_adapter_scope_base_without_lora_is_noop() -> None:
    """没挂 LoRA 时跑 base 不能去调 disable_adapter（普通模型没有这个方法）。"""
    plain = object()
    ctx = ls.adapter_scope(plain, "base", has_lora=False)
    assert ctx.__enter__() is None
    ctx.__exit__(None, None, None)


def test_adapter_scope_base_with_lora_disables() -> None:
    fake = _FakePeft()
    ctx = ls.adapter_scope(fake, "base", has_lora=True)
    ctx.__enter__()
    assert fake.disabled == 1
    assert fake.calls == []


def test_adapter_scope_switches_adapter() -> None:
    fake = _FakePeft()
    ctx = ls.adapter_scope(fake, "sft", has_lora=True)
    ctx.__enter__()
    assert fake.calls == ["sft"]
    assert fake.disabled == 0


def test_default_attn_implementation_forces_eager_on_mps() -> None:
    """MPS 上 sdpa 会让输出退化为重复循环，必须默认 eager。"""
    assert ls.default_attn_implementation("mps", None) == "eager"
    assert ls.default_attn_implementation("cpu", None) is None
    assert ls.default_attn_implementation("mps", "sdpa") == "sdpa"   # 显式指定优先
    assert ls.default_attn_implementation("cuda", "eager") == "eager"
