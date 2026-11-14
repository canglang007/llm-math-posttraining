from typing import Any

from torch.utils.data import Dataset

from .prompts import QWEN_CHATML_TEMPLATE
from .sft import encode_sft_example


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer: Any, max_seq_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return encode_sft_example(self.rows[index]["messages"], self.tokenizer, self.max_seq_len)


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config["model"]
    lora_config = config["lora"]
    dtype = _torch_dtype(model_config["compute_dtype"])
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(model_config["load_in_4bit"]),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_config["path"],
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True))
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_config["r"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            target_modules=list(lora_config["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    return model, tokenizer


def load_eval_model(model_path: str, *, adapter: str | None = None, compute_dtype: str = "bfloat16"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = _torch_dtype(compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = QWEN_CHATML_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=dtype,
        device_map="auto",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def build_trainer(config: dict[str, Any], model: Any, tokenizer: Any, train_rows: list[dict], dev_rows: list[dict]):
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    training = config["training"]
    max_seq_len = int(config["model"]["max_seq_len"])
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
    )
    arguments = TrainingArguments(
        output_dir=training["output_dir"],
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training["logging_steps"]),
        save_strategy=training["save_strategy"],
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=bool(training.get("bf16", True)),
        fp16=False,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=training.get("optim", "paged_adamw_8bit"),
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        dataloader_num_workers=0,
        eval_strategy="epoch" if dev_rows else "no",
        max_steps=int(training.get("max_steps", -1)),
    )
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=SFTDataset(train_rows, tokenizer, max_seq_len),
        eval_dataset=SFTDataset(dev_rows, tokenizer, max_seq_len) if dev_rows else None,
        data_collator=collator,
    )
