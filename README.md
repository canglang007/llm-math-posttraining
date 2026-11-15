# LLM Math Post-Training

一个独立的大模型后训练项目：面向教育场景的私有化数学推理解题助手，用公开 GSM8K/MATH 数据做 SFT、拒绝采样和偏好优化。

这个项目与已有的 Agent 项目分开，不包含 agent、tool calling、user simulator 或 benchmark 复现。目标是展示**数据构造、SFT、RFT/DPO、评测和失败分析**的完整链路。

## 当前状态

- 项目骨架、数据格式、答案抽取、prompt 构造、评测和切分工具已建立；
- 28 个本地单测通过；
- GSM8K 已下载：train 7473 / test 1319；
- MATH 已下载：train 7498 / test 5000（train 有 2 条无 boxed 答案被跳过）；
- SFT 数据：GSM8K 6973 train / 500 dev，MATH 6998 train / 500 dev；
- QLoRA SFT 训练脚本和 12GB/24GB 配置已写好，只做了 dry-run；
- RFT/DPO 数据构造器已实现，并在小样本上验证（RFT 2 条、DPO 1 对）；
- 数据放在 `data/`（gitignored），provenance 在 `data/raw/provenance.json`；
- 尚未训练；GPU 服务器（RTX 4070 12GB）已就绪，工作区在 `/data/llm-math-posttraining`，依赖与 Base 模型正在准备中。

## 原则

- 主训练集、验证集、测试集来自公开数据（GSM8K/MATH，MIT）；
- 合成数据只用于 RFT/DPO，且由公开答案验证；不自造 ground truth 当主证据；
- test 只用于最终评测，不参与训练、过滤或调参；
- Mac 只做开发和单测，不加载模型；训练/推理在 GPU 服务器或 API；
- 先验证本地 foundation，再上服务器。

## 结构

```text
src/math_posttrain/
  types.py        Example / EvalSummary
  answers.py      GSM8K `####`、MATH `\boxed{}` 抽取与等价判断
  prompts.py      0-shot / few-shot prompt 与 chat messages
  data.py         JSONL 读取、Example 构造
  splits.py       dev 切分、切分隔离检查
  evaluation.py   pass@1、抽取失败、分类指标
  sft.py          公开 CoT → Qwen chat SFT records
  preferences.py  RFT / DPO records（答案验证）
  config.py       YAML 配置校验
  training.py     SFTDataset、QLoRA 加载、Trainer

scripts/
  prepare_data.py          下载 GSM8K/MATH（带 provenance）
  eval_predictions.py      对预测 JSONL 做评测
  build_sft_data.py        生成 SFT 数据（可切 dev）
  build_preference_data.py 生成 RFT/DPO 数据
  train_sft.py             QLoRA SFT 训练
  evaluate_model.py        生成预测并算 pass@1

configs/
  sft_qwen3_4b_24gb.yaml   seq 4096 / r16
  sft_qwen3_4b_12gb.yaml   seq 2048 / r8

docs/
  handoff.md                新会话交接（自包含）
  project-charter.md        项目背景、目标、非目标
  experiment-design.md      数据、方法、实验矩阵、指标
  data-provenance.md        数据来源、许可证、切分、污染说明
  issue-log.md              问题记录（只追加）
```

## 快速开始

```bash
cd llm-math-posttraining
python3.13 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

如果本机已有带 pytest 的 Python，也可以直接运行：

```bash
/opt/anaconda3/bin/python3 -m pytest -q
```

准备数据（GSM8K 会直接下载；MATH 需要 `.[eval]` 里的 `datasets`）：

```bash
python scripts/prepare_data.py --dataset gsm8k
python scripts/prepare_data.py --dataset math
```

脚本默认使用 `HF_ENDPOINT=https://hf-mirror.com`；GitHub raw 失败时会自动走 HF。已下载数据在 `data/raw/`（gitignored）。

评测预测文件：

```bash
python scripts/eval_predictions.py --predictions predictions.jsonl --dataset gsm8k
```

生成 SFT / RFT / DPO 数据：

```bash
python scripts/build_sft_data.py --dataset gsm8k --split train
python scripts/build_sft_data.py --dataset math --split train

python scripts/build_preference_data.py \
  --candidates candidates.jsonl --dataset gsm8k
```

`candidates.jsonl` 每行包含 `id`、`question`、`reference`、`solution`，可选 `correct`；`correct` 必须与公开答案检查一致。

## 数据格式

SFT record：

```json
{"id": "...", "source": "gsm8k", "split": "train",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "题目 + 指令"},
              {"role": "assistant", "content": "公开 CoT"}]}
```

RFT record 与 SFT 相同，只保留答案验证正确的 solution。

DPO record：

```json
{"id": "...", "prompt_messages": [system, user],
 "chosen": "正确解答", "rejected": "错误解答", "reference": "..."}
```

## SFT 训练

先 dry-run 校验配置和数据（不需要模型）：

```bash
python scripts/train_sft.py --config configs/sft_qwen3_4b_24gb.yaml --dry-run
```

服务器上先下载 Base 模型：

```bash
hf download Qwen/Qwen3-4B-Base --local-dir /data/models/Qwen3-4B-Base
```

训练：

```bash
# 24GB
python scripts/train_sft.py --config configs/sft_qwen3_4b_24gb.yaml

# 12GB（4070）
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb.yaml
```

评测（dev 选 checkpoint；test 只评一次）：

```bash
# dev pass@1，选 checkpoint
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-24gb/checkpoint-XXX \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
  --dataset gsm8k

# test 只跑一次
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-24gb \
  --data-file data/raw/gsm8k/test.jsonl \
  --dataset gsm8k
```

如果 Base tokenizer 没有 chat template，训练/评测会自动使用 Qwen ChatML fallback。

## 下一步

1. 服务器就绪后用 dev 子集（`data/processed/sft/gsm8k-dev.raw.jsonl`，200 题，greedy）跑 Qwen3-4B-Base 的 baseline probe，确认 headroom 和答案抽取失败率；test 留给最终统一评测；
2. baseline 可解释后，实现 SFT 训练并用 dev 选 checkpoint；
3. 再做 RFT/DPO 和失败分析。
