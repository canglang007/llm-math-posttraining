"""QLoRA SFT for GSM8K/MATH. Does not run during foundation checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_posttrain.config import load_config  # noqa: E402
from math_posttrain.data import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps

    train_path = Path(training["train_file"])
    dev_path = Path(training["dev_file"]) if training.get("dev_file") else None
    if not train_path.exists():
        raise SystemExit(f"train file not found: {train_path}; run scripts/build_sft_data.py first")
    if dev_path is not None and not dev_path.exists():
        raise SystemExit(f"dev file not found: {dev_path}")

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path) if dev_path is not None else []
    if not train_rows:
        raise SystemExit("train file is empty")

    model_path = Path(config["model"]["path"])
    plan = {
        "model": str(model_path),
        "model_exists": model_path.exists(),
        "train_file": str(train_path),
        "train_rows": len(train_rows),
        "dev_file": str(dev_path) if dev_path else None,
        "dev_rows": len(dev_rows),
        "max_seq_len": config["model"]["max_seq_len"],
        "lora_r": config["lora"]["r"],
        "batch_size": training["per_device_train_batch_size"],
        "grad_accum": training["gradient_accumulation_steps"],
        "epochs": training["num_train_epochs"],
        "output_dir": training["output_dir"],
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not model_path.exists():
        raise SystemExit(
            f"model not found: {model_path}; download Qwen/Qwen3-4B-Base first "
            "(hf download Qwen/Qwen3-4B-Base --local-dir ...)"
        )

    from math_posttrain.training import build_trainer, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    trainer = build_trainer(config, model, tokenizer, train_rows, dev_rows)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(training["output_dir"])
    tokenizer.save_pretrained(training["output_dir"])
    print(json.dumps({"saved": training["output_dir"], **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
