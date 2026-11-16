"""QLoRA DPO：在 SFT adapter 基础上做偏好优化。

数据来自 build_preference_data.py 的 DPO 产物，每题一组「同题正确解 / 错误解」配对，
正确性全部由公开答案判定，不引入任何人工偏好标注。

关键设计：
- 起点是 SFT adapter（config.model.base_adapter），不是 Base —— 否则等于把 SFT 白做一遍；
- ref_model 传 None，让 TRL 直接复用「关掉 adapter」的同一份模型，12GB 卡上省一份权重；
- 用 messages 格式喂数据，模板沿用 QWEN_CHATML_TEMPLATE，与 SFT / 评测三处一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.config import load_config  # noqa: E402
from math_posttrain.data import read_jsonl  # noqa: E402


def to_preference_dataset(rows: list[dict]) -> list[dict]:
    """转成 TRL 的 conversational preference 格式。

    chosen / rejected 各自包成一条 assistant 消息，prompt 保留原始多轮结构，
    让 DPOTrainer 用 tokenizer 的 chat_template 统一渲染。
    """
    dataset = []
    for row in rows:
        dataset.append(
            {
                "prompt": row["prompt_messages"],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
        )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    preference_file = Path(training["preference_file"])
    if not preference_file.exists():
        raise SystemExit(f"preference file not found: {preference_file}; run scripts/build_preference_data.py first")

    rows = read_jsonl(preference_file)
    if not rows:
        raise SystemExit("preference file is empty")

    adapter = config["model"].get("base_adapter")
    plan = {
        "model": config["model"]["path"],
        "base_adapter": adapter,
        "preference_file": str(preference_file),
        "pairs": len(rows),
        "beta": training.get("beta", 0.1),
        "max_length": config["model"]["max_seq_len"],
        "output_dir": training["output_dir"],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not adapter:
        raise SystemExit("config.model.base_adapter is required: DPO must start from the SFT adapter")
    if not Path(adapter).exists():
        raise SystemExit(f"base adapter not found: {adapter}")

    from trl import DPOConfig, DPOTrainer

    from math_posttrain.training import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    # TRL 0.29 的 DPOTrainer 需要 datasets.Dataset，不接受 list。
    from datasets import Dataset as HFDataset

    dataset = HFDataset.from_list(to_preference_dataset(rows))

    dpo_args = DPOConfig(
        output_dir=training["output_dir"],
        beta=float(training.get("beta", 0.1)),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=int(training["max_steps"]) if training.get("max_steps") else -1,
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training["logging_steps"]),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=training.get("optim", "paged_adamw_8bit"),
        seed=int(training["seed"]),
        max_length=int(config["model"]["max_seq_len"]),
        # TRL 0.29 已移除 max_prompt_length，prompt 截断由 max_length 统一管理。
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training["output_dir"])
    tokenizer.save_pretrained(training["output_dir"])
    print(json.dumps({"saved": training["output_dir"], **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
