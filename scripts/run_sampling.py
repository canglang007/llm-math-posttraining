"""对同一批 dev 题做多次采样，落盘候选，供推理时验证实验使用。

只在 GPU 服务器上运行；Mac 不加载模型。固定 seed 与解码参数，
所有候选由公开答案在 analyze_sampling.py 里验证，不自造 ground truth。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import load_examples, write_jsonl  # noqa: E402
from math_posttrain.inference_time import sampling_cost  # noqa: E402
from math_posttrain.prompts import (  # noqa: E402
    build_chat_messages,
    generation_stop_strings,
    resolve_eos_token_ids,
    truncate_generation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = load_examples(args.data_file, source=args.dataset, split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    import torch

    from math_posttrain.training import load_eval_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_eval_model(args.model_path, compute_dtype="bfloat16")
    base_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = resolve_eos_token_ids(tokenizer, base_eos)

    rows = []
    total_new_tokens = 0
    for start, example in enumerate(examples, start=1):
        messages = build_chat_messages(example.question, args.dataset)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.k,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=generation_stop_strings(),
                tokenizer=tokenizer,
            )
        candidates = []
        for output in outputs:
            text = truncate_generation(
                tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            )
            candidates.append(text)
            total_new_tokens += len(output) - prompt_len
        rows.append({
            "id": example.id,
            "question": example.question,
            "reference": example.answer,
            "category": example.category,
            "level": example.level,
            "candidates": candidates,
        })
        if start % 10 == 0 or start == len(examples):
            print(f"[progress] {start}/{len(examples)}", flush=True)

    write_jsonl(args.output, rows)
    avg = total_new_tokens / max(1, len(rows) * args.k)
    print(json.dumps({
        "saved": str(args.output),
        "questions": len(rows),
        "k": args.k,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "avg_new_tokens_per_candidate": round(avg, 1),
        "budget": sampling_cost(len(rows), args.k, avg),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
