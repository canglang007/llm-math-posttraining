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
from math_posttrain.prompts import build_chat_messages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("data/processed/predictions/predictions.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
        args.model_path, adapter=args.adapter, compute_dtype="bfloat16"
    )
    rows = []
    for example in examples:
        messages = build_chat_messages(example.question, args.dataset)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        prediction = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
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
    write_jsonl(args.output, rows)
    summary = evaluate(rows, args.dataset)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
