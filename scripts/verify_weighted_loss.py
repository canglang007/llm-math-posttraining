"""校验 EosWeightedTrainer 的加权 loss 计算，定位 loss=30.7 的来源。

对一条真实训练样本，把中间量全部打出来：valid token 数、eos 位置数、
权重和、普通平均 loss、加权 loss。用来区分两种情况：

(a) 公式本身有问题（比如分母/索引错位）；
(b) 公式没错，但 eos 权重 30 让梯度一步把模型推飞。
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/data/llm-math-posttraining/src")

import torch
import torch.nn.functional as F
from peft import PeftModel

from math_posttrain.data import read_jsonl
from math_posttrain.sft import encode_sft_example
from math_posttrain.training import load_eval_model

EOS = 151645
WEIGHT = 30.0

records = read_jsonl("/data/llm-math-posttraining/data/processed/sft/gsm8k-train.jsonl")[:3]
model, tokenizer = load_eval_model("/data/models/Qwen3-4B-Base", force_chat_template=True)
model = PeftModel.from_pretrained(
    model, "/data/llm-math-posttraining/experiments/sft/qwen3-4b-gsm8k-pilot-v2/checkpoint-200"
)
model.eval()

for index, record in enumerate(records):
    encoded = encode_sft_example(record["messages"], tokenizer, 1024)
    ids = torch.tensor([encoded["input_ids"]], device=model.device)
    labels = torch.tensor([encoded["labels"]], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    losses = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)
    valid = flat_labels != -100
    weights = torch.ones_like(losses)
    weights[flat_labels == EOS] = WEIGHT

    print(
        {
            "sample": index,
            "seq_len": int(labels.shape[1]),
            "valid_tokens": int(valid.sum()),
            "eos_positions": int((flat_labels == EOS).sum()),
            "plain_mean_loss": round(float(losses[valid].mean()), 4),
            "eos_loss": round(float(losses[flat_labels == EOS].mean()), 4)
            if int((flat_labels == EOS).sum())
            else None,
            "weight_sum": float(weights[valid].sum()),
            "weighted_loss": round(float((losses * weights)[valid].sum() / weights[valid].sum()), 4),
            "max_token_loss": round(float(losses[valid].max()), 4),
        },
        flush=True,
    )
