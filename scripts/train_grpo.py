"""E4 GRPO：用可验证奖励（答案对错）做在线 RL，替代离线 RFT 的数据量瓶颈。

为什么做 GRPO 而不是继续 RFT：
- RFT 的 412 条离线自蒸馏数据与 SFT 的 0.845 在统计上打平（McNemar p=0.286，
  翻转率 11%）——瓶颈是离线数据量，不是方法。
- GRPO 在线采样、用规则奖励打分，不需要预先攒大量正确解；且不需要 critic
  网络（PPO 需要），16GB 卡可跑。

数据纪律：
- 奖励来自 GSM8K 公开答案（同一个 is_correct），不自造 ground truth；
- 训练题只取 train 前 N 题（与 RFT 采样同批，保证两法可比），dev 不参与；
- test 不参与任何环节。

模型：
- 起点是 SFT adapter（config.model.base_adapter），不是 Base，也不是 RFT
  （RFT 未证明有效，不作为 GRPO 起点，避免污染对比）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.answers import is_correct  # noqa: E402
from math_posttrain.config import load_config  # noqa: E402
from math_posttrain.data import load_examples  # noqa: E402
from math_posttrain.prompts import build_chat_messages, truncate_generation  # noqa: E402


def build_prompt_dataset(config: dict, dataset_name: str, limit: int) -> list[dict]:
    """构造 GRPO 训练集：只给 prompt + 参考答案，答案只用于算奖励，不进模型输入。"""
    path = config["training"]["prompt_file"]
    examples = load_examples(path, source=dataset_name, split="train")
    if limit:
        examples = examples[:limit]
    rows = []
    for example in examples:
        rows.append(
            {
                "prompt": build_chat_messages(example.question, dataset_name),
                "reference": example.answer,
            }
        )
    return rows


def make_reward_fn(dataset_name: str):
    """规则奖励：答案解析后与公开答案比较，对=1.0、错=0.0。

    GRPO 的 completion 可能是 str（plain prompt）或 [{"role","content"}]（conversational），
    两种都要能处理。奖励函数只读字符串，不碰模型内部状态。
    """

    def correctness_reward(completions, reference, **kwargs):
        rewards = []
        for completion, ref in zip(completions, reference):
            if isinstance(completion, str):
                text = completion
            else:
                # conversational 输出取 assistant 段内容
                text = completion[-1]["content"] if completion else ""
            text = truncate_generation(text)
            rewards.append(1.0 if is_correct(text, str(ref), dataset_name) else 0.0)
        return rewards

    return correctness_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    dataset_name = training.get("dataset", "gsm8k")
    limit = int(training.get("num_prompts", 120))

    rows = build_prompt_dataset(config, dataset_name, limit)
    if not rows:
        raise SystemExit("empty GRPO prompt set")

    adapter = config["model"].get("base_adapter")
    plan = {
        "model": config["model"]["path"],
        "base_adapter": adapter,
        "prompts": len(rows),
        "dataset": dataset_name,
        "num_generations": int(training.get("num_generations", 4)),
        "max_completion_length": int(training.get("max_completion_length", 384)),
        "beta": float(training.get("beta", 0.0)),
        "learning_rate": float(training["learning_rate"]),
        "output_dir": training["output_dir"],
    }
    if args.dry_run:
        # 只打印计划，顺便验证奖励函数在样例上能跑通（用假 completion）
        sample = make_reward_fn(dataset_name)(
            completions=[f"steps\n#### {rows[0]['reference']}", "steps\n#### -999999"],
            reference=[rows[0]["reference"], rows[0]["reference"]],
        )
        plan["reward_sanity_check"] = sample
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not adapter:
        raise SystemExit("config.model.base_adapter is required: GRPO must start from the SFT adapter")
    if not Path(adapter).exists():
        raise SystemExit(f"base adapter not found: {adapter}")

    from datasets import Dataset as HFDataset
    from trl import GRPOConfig, GRPOTrainer

    from math_posttrain.training import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    # TRL 0.29 要求 datasets.Dataset；额外列（reference）会自动传给 reward_funcs。
    train_dataset = HFDataset.from_list(rows)

    grpo_args = GRPOConfig(
        output_dir=training["output_dir"],
        num_generations=int(training.get("num_generations", 4)),
        max_completion_length=int(training.get("max_completion_length", 384)),
        temperature=float(training.get("temperature", 0.9)),
        # beta=0：不启用 KL/ref model，省下约一半显存；奖励是规则型（0/1），
        # 不需要参考模型来约束分布。若要更强约束可改为 0.04 并承担显存代价。
        beta=float(training.get("beta", 0.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        max_steps=int(training["max_steps"]) if training.get("max_steps") else -1,
        warmup_ratio=float(training.get("warmup_ratio", 0.05)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training.get("logging_steps", 2)),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 20)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=training.get("optim", "paged_adamw_8bit"),
        seed=int(training["seed"]),
        scale_rewards=training.get("scale_rewards", "group"),
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_fn(dataset_name)],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training["output_dir"])
    tokenizer.save_pretrained(training["output_dir"])
    print(json.dumps({"saved": training["output_dir"], **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
