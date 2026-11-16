"""对同一批 dev 题做多次采样，落盘候选，供推理时验证实验使用。

只在 GPU 服务器上运行；Mac 不加载模型。固定 seed 与解码参数，
所有候选由公开答案在 analyze_sampling.py 里验证，不自造 ground truth。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    parser.add_argument("--adapter", default=None, help="LoRA adapter 目录（RFT 数据构造时指向当前模型）")
    parser.add_argument("--force-chat-template", action="store_true", help="强制使用 QWEN_CHATML_TEMPLATE")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--no-stop-strings",
        action="store_true",
        help="关闭 generate 的字符串停止词。SFT 模型已学会用 <|endoftext|> 主动停止，"
             "stop_strings 每 token 都要 decode 整个序列，是主要性能开销。仅在确认模型会停时用。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取已有输出中的题目 id 并跳过，结果追加写入。不指定则从头覆盖。",
    )
    parser.add_argument("--progress-every", type=int, default=5, help="每隔多少题打印一次进度")
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

    model, tokenizer = load_eval_model(
        args.model_path,
        adapter=args.adapter,
        compute_dtype="bfloat16",
        # SFT/RFT 训练时强制了 QWEN_CHATML_TEMPLATE，采样必须用同一个模板，
        # 否则又回到「训练/推理模板不一致」那个坑里。
        force_chat_template=bool(args.adapter) or args.force_chat_template,
    )
    base_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = resolve_eos_token_ids(tokenizer, base_eos)

    # 断点续跑：已有产物按 id 跳过；结果逐题追加并立即 flush，
    # 这样长时间采样被中断也不会丢掉已经算完的题。
    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(str(json.loads(line)["id"]))
        print(f"[resume] skip {len(done_ids)} completed questions", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_handle = args.output.open("a" if args.resume else "w", encoding="utf-8")

    n_written = 0
    total_new_tokens = 0
    started = time.monotonic()
    for start, example in enumerate(examples, start=1):
        if str(example.id) in done_ids:
            continue
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
                **(
                    {}
                    if args.no_stop_strings
                    else {
                        "stop_strings": generation_stop_strings(),
                        "tokenizer": tokenizer,
                    }
                ),
            )
        candidates = []
        for output in outputs:
            text = truncate_generation(
                tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            )
            candidates.append(text)
            total_new_tokens += len(output) - prompt_len
        row = {
            "id": example.id,
            "question": example.question,
            "reference": example.answer,
            "category": example.category,
            "level": example.level,
            "candidates": candidates,
        }
        out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_handle.flush()
        n_written += 1
        if start % args.progress_every == 0 or start == len(examples):
            elapsed = time.monotonic() - started
            rate = elapsed / max(1, n_written)
            remaining = (len(examples) - start) * rate
            print(
                f"[progress] {start}/{len(examples)} written={n_written} "
                f"elapsed={elapsed:.0f}s rate={rate:.1f}s/q eta={remaining/60:.1f}min",
                flush=True,
            )

    out_handle.close()
    avg = total_new_tokens / max(1, n_written * args.k)
    print(json.dumps({
        "saved": str(args.output),
        "questions": n_written,
        "k": args.k,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "avg_new_tokens_per_candidate": round(avg, 1),
        "budget": sampling_cost(n_written, args.k, avg),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
