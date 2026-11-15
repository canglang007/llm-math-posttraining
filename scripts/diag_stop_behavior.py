"""诊断：SFT adapter 生成时是否输出 ``<|im_end|>`` 停止符。

目的不是评测，而是回答一个具体问题——"模型不停止"的表现到底是哪一种：
(a) 压根没写出 ``#### 答案``，一直在写推理；
(b) 写出了 ``#### 答案``，但继续复述下一题 / 换行续写不收敛；
(c) 写出了正确格式且语义结束，只是不吐 ``<|im_end|>``。

用法（服务器上）：
    python -u scripts/diag_stop_behavior.py \
        --model-path /data/models/Qwen3-4B-Base \
        --adapter experiments/sft/qwen3-4b-gsm8k-pilot-v2/checkpoint-200 \
        --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
        --limit 3 --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import load_examples  # noqa: E402
from math_posttrain.prompts import build_chat_messages  # noqa: E402

IM_END_ID = 151645


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch

    from math_posttrain.training import load_eval_model

    model, tokenizer = load_eval_model(
        args.model_path,
        adapter=args.adapter,
        compute_dtype="bfloat16",
        force_chat_template=True,
    )
    examples = load_examples(args.data_file, source=args.dataset, split="dev")[: args.limit]

    for index, example in enumerate(examples, start=1):
        messages = build_chat_messages(example.question, args.dataset)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output[0][prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        report = {
            "index": index,
            "question": example.question[:80],
            "reference": example.answer,
            "generated_tokens": int(len(new_ids)),
            "hit_max": len(new_ids) >= args.max_new_tokens,
            "has_im_end": IM_END_ID in new_ids.tolist(),
            "preview": text[:1200],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
