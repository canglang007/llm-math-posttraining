"""Generate predictions with a local model/adapter and compute pass@1."""
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
    build_chat_messages,
    generation_stop_strings,
    resolve_eos_token_ids,
    truncate_generation,
)


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
        "--batch-size",
        type=int,
        default=1,
        help="正式评测保持 1；>1 会因左填充与 4bit 数值路径改变 greedy 结果（见 issue-log）",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictions/predictions.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    examples = load_examples(args.data_file, source=args.dataset, split=args.split)
    if args.limit:
        examples = examples[: args.limit]
    if args.dry_run:
        first = build_chat_messages(examples[0].question, args.dataset)
        print(json.dumps({"count": len(examples), "first_messages": first}, ensure_ascii=False, indent=2))
        return

    import torch

    from math_posttrain.training import load_eval_model

    model, tokenizer = load_eval_model(
        args.model_path,
        adapter=args.adapter,
        compute_dtype="bfloat16",
        # 训练时强制了 QWEN_CHATML_TEMPLATE 的 adapter，评测必须同模板，
        # 否则训练/推理分布不一致，数字不可比。
        force_chat_template=bool(args.adapter) or args.force_chat_template,
    )
    base_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = resolve_eos_token_ids(tokenizer, base_eos)
    # decoder-only 模型批量生成必须左填充，否则 padding 会被当成真实上文。
    tokenizer.padding_side = "left"

    rows = []
    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(example.question, args.dataset),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in chunk
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
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
        for example, output in zip(chunk, outputs):
            prediction = truncate_generation(
                tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            )
            rows.append(
                {
                    "id": example.id,
                    "prediction": prediction,
                    "reference": example.answer,
                    "category": example.category,
                    "level": example.level,
                }
            )
        done = start + len(chunk)
        if done % 10 == 0 or done == len(examples):
            print(f"[progress] {done}/{len(examples)}", flush=True)
    write_jsonl(args.output, rows)
    summary = evaluate(rows, args.dataset)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
