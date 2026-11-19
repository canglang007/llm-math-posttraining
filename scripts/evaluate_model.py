"""Generate predictions with a local model/adapter and compute pass@1.

两种推理后端，prompt/停止条件/截断完全共用，保证口径可比：

- ``--backend hf``（默认）：transformers ``generate``，4bit NF4 + LoRA，batch=1。
  历史上所有分数（含 dev-200 十一档）都是这条路径跑出来的，改动它等于换尺子。
- ``--backend vllm``：vLLM 引擎，continuous batching，吞吐高一个数量级。
  与 HF 的差异：计算内核不同，且底座按 ``--vllm-quantization`` 设定加载
  （默认 bitsandbytes，与 HF 的 4bit 对齐）。**换后端必须做口径校准**，
  逐题比对一致率并记录，见 docs/issue-log.md。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import load_examples, write_jsonl  # noqa: E402
from math_posttrain.evaluation import evaluate  # noqa: E402
from math_posttrain.prompts import (  # noqa: E402
    QWEN_CHATML_TEMPLATE,
    build_chat_messages,
    generation_stop_strings,
    resolve_eos_token_ids,
    truncate_generation,
)

BACKENDS = ("hf", "vllm")


def load_prompt_tokenizer(model_path: str, *, force_chat_template: bool):
    """HF / vLLM 两条路径共用同一个 tokenizer 来渲染 prompt。

    模板规则与 ``training.load_eval_model`` 一致：adapter 是在强制
    QWEN_CHATML_TEMPLATE 下训练的，评测必须同模板，否则分布不一致。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if force_chat_template or getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    return tokenizer


def render_prompts(examples, tokenizer, dataset: str) -> list[str]:
    """把题目渲染成最终送入模型的字符串（chat template + generation prompt）。"""
    return [
        tokenizer.apply_chat_template(
            build_chat_messages(example.question, dataset),
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]


def vllm_sampling_kwargs(max_new_tokens: int, eos_ids: list[int]) -> dict:
    """与 HF 路径语义等价的 greedy 采样参数。

    纯函数、不 import vllm，方便单测守住口径：temperature=0（greedy）、
    同一组停止字符串与停止 token、输出里不保留停止串（HF 侧同样会截掉）。
    """
    return {
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "stop": generation_stop_strings(),
        "stop_token_ids": list(eos_ids),
        "include_stop_str_in_output": False,
        "skip_special_tokens": True,
    }


def adapter_rank(adapter_path: str) -> int:
    """读 LoRA adapter 的 r，vLLM 的 max_lora_rank 必须 >= 它。"""
    config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    return int(config.get("r", 8))


def require_vllm():
    """延迟导入 vLLM；未安装时给可读的报错而不是裸 ImportError。"""
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 依赖环境
        raise RuntimeError(
            "--backend vllm 需要先安装 vllm（pip install vllm）；"
            "不装就用默认的 --backend hf。"
        ) from exc
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    return LLM, SamplingParams, LoRARequest


def generate_hf(examples, prompts, tokenizer, args) -> list[str]:
    """历史路径：4bit + LoRA，逐批 generate。保持行为不变。"""
    import torch

    from math_posttrain.training import load_eval_model

    model, _ = load_eval_model(
        args.model_path,
        adapter=args.adapter,
        compute_dtype="bfloat16",
        # 训练时强制了 QWEN_CHATML_TEMPLATE 的 adapter，评测必须同模板，
        # 否则训练/推理分布不一致，数字不可比。
        force_chat_template=bool(args.adapter) or args.force_chat_template,
        # 默认保持 4bit（历史口径）；--no-quantize 只用于量化/引擎的诊断实验。
        quantize=not args.no_quantize,
    )
    base_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = resolve_eos_token_ids(tokenizer, base_eos)
    # decoder-only 模型批量生成必须左填充，否则 padding 会被当成真实上文。
    tokenizer.padding_side = "left"

    texts: list[str] = []
    for start in range(0, len(examples), args.batch_size):
        chunk_prompts = prompts[start : start + args.batch_size]
        inputs = tokenizer(chunk_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=generation_stop_strings(),
                tokenizer=tokenizer,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for output in outputs:
            texts.append(tokenizer.decode(output[prompt_len:], skip_special_tokens=True))
        done = start + len(chunk_prompts)
        if done % 10 == 0 or done == len(examples):
            print(f"[progress] {done}/{len(examples)}", flush=True)
    return texts


def generate_vllm(prompts, tokenizer, args) -> list[str]:
    """vLLM 路径：continuous batching，靠引擎自己调度，所以没有 batch-size 概念。"""
    LLM, SamplingParams, LoRARequest = require_vllm()

    eos_ids = resolve_eos_token_ids(tokenizer, getattr(tokenizer, "eos_token_id", None))
    engine_kwargs = {
        "model": args.model_path,
        "tokenizer": args.model_path,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        # prompt ≤1024 + 生成 ≤512，留足余量给不同版本的默认值差异。
        "max_model_len": 2048,
        "trust_remote_code": True,
    }
    lora_request = None
    if args.adapter:
        engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=adapter_rank(args.adapter),
            max_loras=1,
        )
        lora_request = LoRARequest("adapter", 1, args.adapter)
    if args.vllm_quantization and args.vllm_quantization != "none":
        engine_kwargs["quantization"] = args.vllm_quantization

    llm = LLM(**engine_kwargs)
    params = SamplingParams(**vllm_sampling_kwargs(args.max_new_tokens, eos_ids))

    texts: list[str] = []
    for start in range(0, len(prompts), args.vllm_chunk):
        chunk = prompts[start : start + args.vllm_chunk]
        try:
            outputs = llm.generate(chunk, params, lora_request=lora_request)
        except TypeError:  # 旧版 vLLM 的 generate 不接受 lora_request 关键字
            if lora_request is not None:
                raise RuntimeError(
                    "当前 vLLM 版本不支持 generate(..., lora_request=...)，请升级 vllm。"
                )
            outputs = llm.generate(chunk, params)
        # vLLM 对字符串 prompt 只返回新生成的部分，与 HF 侧 output[prompt_len:] 对齐。
        texts.extend(output.outputs[0].text for output in outputs)
        done = start + len(chunk)
        if done % 10 == 0 or done == len(prompts):
            print(f"[progress] {done}/{len(prompts)}", flush=True)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--force-chat-template", action="store_true", help="强制使用 QWEN_CHATML_TEMPLATE")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="hf",
        help="hf=transformers generate（历史口径，默认）；vllm=高吞吐引擎（换后端须做口径校准）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="仅 hf 后端生效。正式评测保持 1；>1 会因左填充与 4bit 数值路径改变 greedy 结果（见 issue-log）",
    )
    parser.add_argument(
        "--vllm-quantization",
        choices=("bitsandbytes", "none"),
        default="bitsandbytes",
        help="vllm 后端的底座量化方式：bitsandbytes 与 HF 的 4bit 对齐（默认）；none=bf16",
    )
    parser.add_argument(
        "--vllm-chunk",
        type=int,
        default=200,
        help="vllm 后端每次提交给引擎的请求数（只影响显存峰值与进度打印）",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictions/predictions.jsonl"))
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="hf 后端专用诊断开关：bf16 全精度加载（不用 4bit），仅用于量化敏感性实验",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.vllm_chunk < 1:
        parser.error("--vllm-chunk must be >= 1")
    if args.no_quantize and args.backend != "hf":
        parser.error("--no-quantize 只对 --backend hf 有意义")

    examples = load_examples(args.data_file, source=args.dataset, split=args.split)
    if args.limit:
        examples = examples[: args.limit]
    if args.dry_run:
        first = build_chat_messages(examples[0].question, args.dataset)
        print(json.dumps({"count": len(examples), "first_messages": first}, ensure_ascii=False, indent=2))
        return

    force_chat_template = bool(args.adapter) or args.force_chat_template
    tokenizer = load_prompt_tokenizer(args.model_path, force_chat_template=force_chat_template)
    prompts = render_prompts(examples, tokenizer, args.dataset)

    if args.backend == "vllm":
        raw_texts = generate_vllm(prompts, tokenizer, args)
    else:
        raw_texts = generate_hf(examples, prompts, tokenizer, args)

    rows = []
    for example, raw in zip(examples, raw_texts):
        rows.append(
            {
                "id": example.id,
                "prediction": truncate_generation(raw),
                "reference": example.answer,
                "category": example.category,
                "level": example.level,
            }
        )
    write_jsonl(args.output, rows)
    summary = evaluate(rows, args.dataset)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(f"BACKEND={args.backend}", flush=True)


if __name__ == "__main__":
    main()
