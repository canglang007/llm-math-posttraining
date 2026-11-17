from pathlib import Path
from typing import Any

import yaml

REQUIRED = {
    "model": ("path", "load_in_4bit", "compute_dtype", "max_seq_len"),
    "lora": ("r", "alpha", "dropout", "target_modules"),
    "training": (
        "output_dir",
        "train_file",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "logging_steps",
        "save_strategy",
        "seed",
    ),
}

# DPO 不需要 train_file（SFT 格式），而是 preference_file（chosen/rejected 配对）。
DPO_TRAINING_KEYS = (
    "output_dir",
    "preference_file",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "logging_steps",
    "save_strategy",
    "seed",
)


# GRPO 需要 prompt_file（只给 prompt + 参考答案，答案仅用于算规则奖励）。
GRPO_TRAINING_KEYS = (
    "output_dir",
    "prompt_file",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "num_train_epochs",
    "logging_steps",
    "save_strategy",
    "seed",
)

_TASK_TRAINING_KEYS = {
    "dpo": DPO_TRAINING_KEYS,
    "grpo": GRPO_TRAINING_KEYS,
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    for section, keys in REQUIRED.items():
        values = data.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"config missing section: {section}")
        required = keys
        if section == "training":
            required = _TASK_TRAINING_KEYS.get(str(data.get("task") or "sft"), keys)
        for key in required:
            if key not in values:
                raise ValueError(f"config missing {section}.{key}")
    return data
