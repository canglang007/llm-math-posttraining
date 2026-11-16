"""诊断：SFT 到底有没有学会在正确位置输出 ``<|im_end|>``。

不训练、不生成，只对训练分布做一次 forward，然后按 token 位置拆 loss。
回答一个具体问题：**停止失败是"欠拟合"还是"loss 权重被稀释"**。

两个指标：
1. ``eos_loss`` —— ``<|im_end|>`` 那一个 token 的 loss，与 assistant 段平均 loss 对比。
   平均 loss 低而 eos loss 高，说明 65:1 的 token 平均把停止信号淹没了。
2. ``eos_top1_hit`` —— 在应当输出 ``<|im_end|>`` 的位置，greedy argmax 是否就是
   151645。generate 用的是 greedy，这个数直接等于"会不会停"。

同一进程内先测 Base 再挂 adapter 测 SFT，两份结果并排，差值即 SFT 的贡献。

用法（服务器上）：
    python -u scripts/diag_eos_loss.py \
        --model-path /data/models/Qwen3-4B-Base \
        --adapter experiments/sft/qwen3-4b-gsm8k-pilot-v2/checkpoint-200 \
        --data-file data/processed/sft/gsm8k-train.jsonl \
        --samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.data import read_jsonl  # noqa: E402
from math_posttrain.sft import encode_sft_example  # noqa: E402

IM_END_ID = 151645
# 对照组：Qwen3-Base 自己的 eos，预训练中大量出现，lm_head 行是训练过的。
# 若它在同一位置 rank 很高、而 <|im_end|> 排到 2000 名，就证明问题不在
# 「模型不知道该停」，而在「<|im_end|> 这个 token 本身没被学过」。
END_OF_TEXT_ID = 151643


def _measure(model, tokenizer, records, max_seq_len: int) -> dict:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    assistant_losses: list[float] = []
    eos_losses: list[float] = []
    eos_hits = 0
    eos_ranks: list[int] = []
    eos_probs: list[float] = []
    alt_im_end: list[tuple[int, float]] = []
    alt_eot: list[tuple[int, float]] = []
    tail_losses: list[list[float]] = []
    top5_samples: list[dict] = []

    for record in records:
        encoded = encode_sft_example(record["messages"], tokenizer, max_seq_len)
        input_ids = torch.tensor([encoded["input_ids"]], device=device)
        labels = torch.tensor([encoded["labels"]], device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0]

        # labels[j] 由位置 j-1 的 logits 预测，故先整体左移一位再对齐。
        # 注意：必须保留完整长度——压缩掉 -100 后位置索引就和 labels 对不上了。
        shift_logits = logits[:-1].float()
        shift_labels = labels[0][1:]
        token_losses = F.cross_entropy(
            shift_logits, shift_labels, reduction="none", ignore_index=-100
        )
        mask = shift_labels != -100
        assistant_losses.append(float(token_losses[mask].mean()))
        tail_losses.append([round(float(value), 4) for value in token_losses[mask][-5:].tolist()])

        # 停止符不再写死 151645：训练目标可能换成 <|endoftext|>(151643)。
        # assistant 段的最后一个非 -100 token 就是停止符本身，直接定位它，
        # 这样换停止符方案时同一个脚本可以前后对比。
        valid_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
        eos_position = int(valid_positions[-1])
        if eos_position == 0:
            continue
        eos_token = int(labels[0][eos_position])
        eos_losses.append(float(token_losses[eos_position - 1]))
        position_logits = shift_logits[eos_position - 1]
        top_ids = position_logits.argsort(descending=True)
        eos_hits += int(int(top_ids[0]) == eos_token)
        eos_ranks.append(int((top_ids == eos_token).nonzero(as_tuple=True)[0][0]) + 1)
        eos_probs.append(float(F.softmax(position_logits, dim=-1)[eos_token]))
        # 对照：同一个位置，两个候选停止符各自的排名。
        for token_id, bucket in ((IM_END_ID, alt_im_end), (END_OF_TEXT_ID, alt_eot)):
            bucket.append(
                (
                    int((top_ids == token_id).nonzero(as_tuple=True)[0][0]) + 1,
                    float(F.softmax(position_logits, dim=-1)[token_id]),
                )
            )
        if len(top5_samples) < 3:
            # 诊断关键：如果 top-5 是语义合理的续写（换行、"Question:" 等），
            # 说明 lm_head 正常，只是偏好问题；若是杂乱 token，说明该 token
            # 的输出投影没被训练过，LoRA 不覆盖 lm_head 根本改不动它。
            top5 = top_ids[:5].tolist()
            top5_samples.append(
                {
                    "tokens": [
                        {"text": tokenizer.decode([tid]), "prob": round(float(F.softmax(position_logits, dim=-1)[tid]), 5)}
                        for tid in top5
                    ],
                    "eos_rank": int((top_ids == IM_END_ID).nonzero(as_tuple=True)[0][0]) + 1,
                }
            )

    return {
        "samples": len(assistant_losses),
        "assistant_mean_loss": round(sum(assistant_losses) / len(assistant_losses), 4),
        "eos_loss": round(sum(eos_losses) / len(eos_losses), 4) if eos_losses else None,
        "eos_top1_hit": eos_hits,
        "eos_top1_rate": round(eos_hits / len(eos_losses), 4) if eos_losses else None,
        "eos_mean_rank": round(sum(eos_ranks) / len(eos_ranks), 2) if eos_ranks else None,
        "eos_mean_prob": round(sum(eos_probs) / len(eos_probs), 4) if eos_probs else None,
        "endoftext_mean_rank": round(sum(r for r, _ in alt_eot) / len(alt_eot), 2) if alt_eot else None,
        "endoftext_mean_prob": round(sum(p for _, p in alt_eot) / len(alt_eot), 4) if alt_eot else None,
        "im_end_mean_rank": round(sum(r for r, _ in alt_im_end) / len(alt_im_end), 2) if alt_im_end else None,
        "im_end_mean_prob": round(sum(p for _, p in alt_im_end) / len(alt_im_end), 4) if alt_im_end else None,
        "eos_top5": top5_samples,
        "tail_loss_examples": tail_losses[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    from peft import PeftModel

    from math_posttrain.training import load_eval_model

    records = read_jsonl(args.data_file)[: args.samples]
    model, tokenizer = load_eval_model(args.model_path, force_chat_template=True)

    report = {"samples": len(records)}
    report["base"] = _measure(model, tokenizer, records, args.max_seq_len)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        report["sft"] = _measure(model, tokenizer, records, args.max_seq_len)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
