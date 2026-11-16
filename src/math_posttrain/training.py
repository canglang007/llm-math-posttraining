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


def load_model_and_tokenizer(config: dict[str, Any], *, adapter_path: str | None = None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
    # Qwen3 官方模板会在 assistant 目标前注入空的 <think>\n\n</think>\n\n，
    # 而推理侧 apply_chat_template 不注入，训练/推理 prompt 分布不一致。
    # force_chat_template 让两边都用 QWEN_CHATML_TEMPLATE，去掉这段空 think 块。
    if bool(model_config.get("force_chat_template", False)) or getattr(tokenizer, "chat_template", None) is None:
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
    # RFT / DPO 的起点是 SFT adapter（而不是 Base），否则等于把 SFT 白做一遍。
    # 配置里写 model.base_adapter，或命令行显式传 adapter_path。
    resume_adapter = adapter_path or model_config.get("base_adapter")
    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
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


def load_eval_model(
    model_path: str,
    *,
    adapter: str | None = None,
    compute_dtype: str = "bfloat16",
    force_chat_template: bool = False,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = _torch_dtype(compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 与训练侧保持一致：SFT 用 QWEN_CHATML_TEMPLATE 训练时，评测也必须用同一个模板。
    if force_chat_template or getattr(tokenizer, "chat_template", None) is None:
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


def _eos_weighted_trainer_class():
    """延迟到调用时才 import transformers：Mac 侧不装训练依赖也能 import 本模块。

    给 ``<|im_end|>`` 加权，修「会答但不会停」。

    实测（``scripts/diag_eos_loss.py``，20 条训练样本，checkpoint-200）：

    ==================  ==============  ===============  ============
    指标                  Base            SFT(200 步)      说明
    ==================  ==============  ===============  ============
    assistant 平均 loss   —               0.338           看着已收敛
    ``<|im_end|>`` loss  15.22           7.83            困惑度 ≈ 2500
    argmax 排名          14582           1982            要进 top1 才会停
    top1 命中             0/20            0/20            generate 用 greedy
    ==================  ==============  ===============  ============

    尾部逐 token loss 是 ``[1e-4, 1e-4, 0.0, 7.83, 0.0]``——答案行之前每个
    token 都拟合到 1e-4，唯独收尾那一位没学会。原因是 CE 按 token 平均，
    assistant 段约 64 token，停止符只占 1/64，梯度被稀释 64 倍，摊进平均
    loss 只剩约 0.12，看起来「已经收敛」，实际最关键的决策完全没学到。

    所以这里把停止符位置的 loss 权重放大到 ``eos_loss_weight``，让它成为
    一个独立的学习目标；而不是靠加轮数、用 1/64 的效率慢慢磨。
    """
    from transformers import Trainer

    class EosWeightedTrainer(Trainer):
        def __init__(self, *args, eos_token_id: int = 151645, eos_weight: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.eos_token_id = int(eos_token_id)
            self.eos_weight = float(eos_weight)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # labels[j] 由位置 j-1 的 logits 预测，整体左移一位对齐。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            losses = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)
            weights = torch.ones_like(losses)
            weights[flat_labels == self.eos_token_id] = self.eos_weight
            valid = flat_labels != -100
            # 分母也用权重求和：否则加权后 loss 量级整体放大，LR 相当于被偷偷改了。
            loss = (losses * weights)[valid].sum() / weights[valid].sum()
            # Trainer 只在 model_accepts_loss_kwargs=False 时才帮忙除 accum，
            # 而 Qwen3 的 forward 有 num_items_in_batch 参数，被判为 True，
            # 于是它假定「模型自己做了缩放」不会再除。我们绕过了那套逻辑，
            # 必须自己补上——否则梯度被放大 accum 倍（实测 loss 30.7 ≈ 1.9×16，
            # grad_norm 4.07 vs 正常 0.3，等效 LR 被放大 16 倍）。
            loss = loss / self.args.gradient_accumulation_steps
            return (loss, outputs) if return_outputs else loss

    return EosWeightedTrainer


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
        save_steps=int(training.get("save_steps", 500)),
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
    trainer_kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": SFTDataset(train_rows, tokenizer, max_seq_len),
        "eval_dataset": SFTDataset(dev_rows, tokenizer, max_seq_len) if dev_rows else None,
        "data_collator": collator,
    }
    # eos_loss_weight > 1 时才换 trainer，默认 1.0 与历史实验完全一致。
    eos_weight = float(training.get("eos_loss_weight", 1.0))
    if eos_weight > 1.0:
        trainer_cls = _eos_weighted_trainer_class()
        return trainer_cls(
            eos_token_id=int(training.get("eos_token_id", 151645)),
            eos_weight=eos_weight,
            **trainer_kwargs,
        )
    return Trainer(**trainer_kwargs)
