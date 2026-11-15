from pathlib import Path

import pytest

from math_posttrain.config import load_config

VALID = """
model:
  path: /data/models/Qwen3-4B-Base
  load_in_4bit: true
  compute_dtype: bfloat16
  max_seq_len: 4096
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj]
training:
  output_dir: experiments/sft/test
  train_file: data/processed/sft/gsm8k-train.jsonl
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 0.0002
  num_train_epochs: 2
  logging_steps: 10
  save_strategy: epoch
  seed: 42
"""


def test_load_config_reads_required_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID, encoding="utf-8")
    config = load_config(path)
    assert config["model"]["max_seq_len"] == 4096
    assert config["lora"]["r"] == 16
    assert config["training"]["seed"] == 42


def test_load_config_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID.replace("  seed: 42\n", ""), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_pilot_config_disables_trainer_eval_and_keeps_checkpoints() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "sft_qwen3_4b_12gb_pilot.yaml"
    training = load_config(path)["training"]
    assert "dev_file" not in training
    assert training["save_strategy"] == "steps"
    assert training["save_steps"] == 50
    assert training["max_steps"] == 200
