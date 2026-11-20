#!/usr/bin/env python3
"""本地推理服务：transformers + MPS/CPU，暴露 OpenAI 兼容的最小接口子集。

**用途**：服务器不可用（没机器 / 没卡 / 不想开机）时，仍能在本机跑 `demo/live.html`。
接口与 `deploy/serve.sh`（vLLM）一致，所以演示页不需要改代码，只换端点。

**口径**：与 `scripts/evaluate_model.py` 的 HF 路径完全一致——
同一 ChatML 模板、greedy（do_sample=False）、同一组停止词与 eos token、
生成后再过同一个 `truncate_generation`。prompt 与停止词都从
`math_posttrain.prompts` 导入，不在这里复制一份。

**边界（必须明确）**：MPS/CPU 的数值路径与 CUDA/vLLM 不同，本地输出只用于演示与自查，
**不作为实验数字来源**；文档里的分数一律来自 `experiments/eval/` 的离线全量评测。

用法：
    python deploy/local_server.py --dry-run          # 只打印将要加载什么，不加载模型
    python deploy/local_server.py                    # 默认 8000 端口
    python deploy/local_server.py --port 8100 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# 默认 adapter 映射；缺失的会被跳过并提示，不影响其余模型可用
DEFAULT_ADAPTERS = {
    "sft": "experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400",
    "e2b": "experiments/rft/qwen3-4b-gsm8k-rft-300q",
    "a2": "experiments/rft/qwen3-4b-gsm8k-rft-a2nf-300q",
    "dpo": "experiments/dpo/qwen3-4b-gsm8k-dpo-from-sft",
}

CHATML_IM_END_ID = 151645
CHATML_IM_START_ID = 151644


# ---------------------------------------------------------------- 纯函数（可单测）


def parse_adapter_spec(spec: str | None) -> dict[str, str]:
    """解析 ``name=path,name2=path2``；空值返回空字典。"""
    if not spec:
        return {}
    mapping: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"adapter 规格应形如 name=path：{item!r}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"adapter 名或路径为空：{item!r}")
        mapping[name] = path
    return mapping


def resolve_adapters(
    spec: dict[str, str] | None,
    root: Path,
    *,
    warn=print,
) -> dict[str, str]:
    """返回实际可用的 adapter（路径存在且含 adapter_config.json）。"""
    spec = spec if spec is not None else DEFAULT_ADAPTERS
    usable: dict[str, str] = {}
    for name, raw in spec.items():
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if (path / "adapter_config.json").is_file():
            usable[name] = str(path)
        else:
            warn(f"[skip] adapter {name} 不存在：{path}")
    return usable


def cors_headers() -> dict[str, str]:
    """本地演示页从 file:// 打开，Origin 为 null，必须放开跨域。"""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def build_completion_response(
    model: str, text: str, prompt_tokens: int, completion_tokens: int
) -> dict:
    return {
        "id": f"cmpl-local-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop", "logprobs": None}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_models_response(names: list[str]) -> dict:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "local"}
            for name in names
        ],
    }


def check_sampling(body: dict) -> str | None:
    """本地后端只实现 greedy；非 0 temperature 直接拒绝，避免口径悄悄漂移。"""
    temperature = body.get("temperature", 0)
    if temperature not in (0, 0.0, None):
        return f"本地后端只支持 greedy 解码（temperature=0），收到 {temperature!r}"
    return None


# ---------------------------------------------------------------- 引擎


class LocalEngine:
    """把 base + 若干 LoRA 放在一个进程里，按请求切换 adapter。"""

    def __init__(
        self,
        model_path: str,
        adapters: dict[str, str],
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        import torch  # 延迟导入：--dry-run 与单测都不需要 torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from math_posttrain.prompts import QWEN_CHATML_TEMPLATE, generation_stop_strings
        from math_posttrain.prompts import resolve_eos_token_ids
        from math_posttrain.prompts import truncate_generation

        self._torch = torch
        self._stop_strings = list(generation_stop_strings())
        self._truncate = truncate_generation
        self._lock = threading.Lock()
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        if dtype == "auto":
            # MPS 上 bf16 在较新 torch 才完整支持，失败时回退 fp16；CPU 用 fp32
            dtype = "bfloat16" if device == "mps" else "float32"
        torch_dtype = getattr(torch, dtype)
        if device == "mps" and dtype == "bfloat16":
            try:
                probe = torch.zeros(1, dtype=torch_dtype, device=device)
                del probe
            except Exception:  # pragma: no cover - 取决于机型
                print("[warn] 该设备不支持 bfloat16，回退 float16", flush=True)
                dtype, torch_dtype = "float16", torch.float16

        self.dtype = dtype
        print(f"[load] base={model_path} device={device} dtype={dtype}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 训练时强制了 ChatML 模板，推理必须同模板，否则分布不一致
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=None
        ).to(device)
        model.eval()

        if adapters:
            from peft import PeftModel

            first_name, first_path = next(iter(adapters.items()))
            print(f"[load] adapter {first_name} <- {first_path}", flush=True)
            model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
            for name, path in list(adapters.items())[1:]:
                print(f"[load] adapter {name} <- {path}", flush=True)
                model.load_adapter(path, adapter_name=name)
            self._has_lora = True
        else:
            self._has_lora = False

        self.model = model
        self.tokenizer = tokenizer
        base_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        self.eos_ids = resolve_eos_token_ids(tokenizer, base_eos)
        self.adapter_names = set(adapters) | {"base"}

    def models(self) -> list[str]:
        return ["base"] + sorted(n for n in self.adapter_names if n != "base")

    def generate(self, model_name: str, prompt: str, max_new_tokens: int | None = None) -> dict:
        torch = self._torch
        n = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with self._lock:  # MPS/CUDA 单设备，串行执行
            with adapter_scope(self.model, model_name, self._has_lora):
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=n,
                        do_sample=False,
                        eos_token_id=self.eos_ids,
                        pad_token_id=self.tokenizer.pad_token_id,
                        stop_strings=self._stop_strings,
                        tokenizer=self.tokenizer,
                    )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {
            "text": self._truncate(text),
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
        }


class _noop:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def adapter_scope(model, model_name: str, has_lora: bool):
    """选择本次生成要用的 adapter 作用域。

    - 跑 ``base`` 且确实挂了 LoRA：临时禁用 adapter（PeftModel.disable_adapter）
    - 跑 ``base`` 但没有 LoRA：什么都不做（普通模型没有 disable_adapter 方法）
    - 跑具体 adapter：切到它，其余不变
    """
    if model_name == "base":
        return model.disable_adapter() if has_lora else _noop()
    model.set_adapter(model_name)
    return _noop()


# ---------------------------------------------------------------- HTTP


def make_handler(engine: LocalEngine | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "math-posttrain-local/1.0"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for key, value in cors_headers().items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            for key, value in cors_headers().items():
                self.send_header(key, value)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                names = engine.models() if engine else ["base"]
                self._send(200, build_models_response(names))
            elif self.path.rstrip("/") in ("/health", "/healthz"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/completions", "/completions"):
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": {"message": f"bad json: {exc}"}})
                return

            problem = check_sampling(body)
            if problem:
                self._send(400, {"error": {"message": problem}})
                return

            model_name = str(body.get("model", "base"))
            if engine is None:
                self._send(503, {"error": {"message": "服务未加载模型（--dry-run 模式）"}})
                return
            if model_name not in engine.models():
                self._send(
                    404,
                    {"error": {"message": f"unknown model {model_name}; 可用：{engine.models()}"}},
                )
                return
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self._send(400, {"error": {"message": "prompt 必须是非空字符串"}})
                return

            try:
                result = engine.generate(model_name, prompt, body.get("max_tokens"))
            except Exception as exc:  # pragma: no cover - 运行时错误
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                return
            self._send(
                200,
                build_completion_response(
                    model_name,
                    result["text"],
                    result["prompt_tokens"],
                    result["completion_tokens"],
                ),
            )

        def log_message(self, fmt: str, *args) -> None:  # 静音默认访问日志
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地（MPS/CPU）OpenAI 兼容推理服务")
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--adapters", default=None, help="name=path,name2=path2；缺省用内置映射")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="只打印配置，不加载模型")
    args = parser.parse_args(argv)

    # 不传 --adapters → 用内置默认映射；传空串 → 明确表示「只跑 base」
    spec = None if args.adapters is None else parse_adapter_spec(args.adapters)
    adapters = resolve_adapters(spec, PROJECT_ROOT)
    print(f"[cfg] base   : {args.model_path}")
    print(f"[cfg] adapter: {list(adapters) or '（无，只能跑 base）'}")
    print(f"[cfg] listen : http://{args.host}:{args.port}")
    if args.dry_run:
        print("[cfg] dry-run：不加载模型，仅校验路径")
        return 0

    engine = LocalEngine(
        args.model_path,
        adapters,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[ready] 可用模型：{engine.models()}", flush=True)
    print(f"[ready] endpoint: http://{args.host}:{args.port}/v1", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 已退出")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
