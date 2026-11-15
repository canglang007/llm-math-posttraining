# LLM Math Post-Training

一个独立的大模型后训练项目：面向教育场景的私有化数学推理解题助手，用公开 GSM8K/MATH 数据做 SFT、拒绝采样和偏好优化。

这个项目与已有的 Agent 项目分开，不包含 agent、tool calling、user simulator 或 benchmark 复现。目标是展示**数据构造、SFT、RFT/DPO、评测和失败分析**的完整链路。

## 主要结果（推理时验证，2026-11-15）

模型冻结、不做任何训练，只加推理时计算：Qwen3-4B-Base 在 GSM8K dev 200 题上每题采样 8 条，
把 `####` 后的答案归一化后投票。

| 方法 | pass@1 | 说明 |
|---|---|---|
| E0 greedy | 0.770 | 基线，154/200 |
| 单条采样（T=0.7） | 0.749 | 单条还不如 greedy |
| **maj@3** | **0.880** | +11 点，3× token，性价比拐点 |
| **maj@8** | **0.920** | +15 点，8× token；救回 31 题、带偏 1 题 |
| oracle pass@8 | 0.965 | 验证信号上界，还剩 4.5 点要靠 verifier |

按投票一致度做选择性预测：一致度 ≥75% 的 168 题（84%）准确率 **97.6%**，
剩下 32 题准确率 45%–71% → 教育场景的「自动批改 / 转人工」阈值。

完整的表、增益分解、失败案例与负结果见 **`docs/first-sampling-results.md`**；
可离线打开的演示页见 `demo/index.html`。

## 当前状态

- 项目骨架、数据格式、答案抽取、prompt 构造、评测和切分工具已建立；
- 50 个单测通过（本机需加 `--basetemp=/tmp/mpt-pytest`，沙箱会拦 pytest 的临时目录创建）；
- GSM8K：train 7473 / test 1319；MATH：train 7498 / test 5000（2 条无 boxed 答案已跳过）；
- SFT 数据：GSM8K 6973 train / 500 dev，MATH 6998 train / 500 dev；
- E0 baseline（dev 200 题，`--batch-size 1`）：pass@1 0.77、抽取失败 27（13.5%）；
- 推理时验证：dev 200 × k=8 采样 52 分钟出第一张结果表（见上）；`scripts/run_sampling.py` 采样、`scripts/analyze_sampling.py` 出表与曲线、`scripts/build_demo.py` 生成 demo；
- E1 pilot SFT（200 步）已完成但**三层诊断确认退化**（0.46 epoch 欠拟合），不能与 E0 对比；训练已从主线降为加分项（原因见 `docs/pivot-plan.md`）；
- QLoRA SFT 训练脚本与 12GB/24GB 配置、RFT/DPO 数据构造器均已实现并在小样本验证；
- GPU 服务器（RTX 4060 Ti 16GB）工作区 `/data/llm-math-posttraining`，`/data` 为持久卷，恢复清单见 handoff 第 9 节；
- 调试与决策过程见 `docs/interview-debug-journal.md`（面试用）。

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
  inference_time.py 多数投票、oracle pass@k、成本估算

scripts/
  prepare_data.py          下载 GSM8K/MATH（带 provenance）
  eval_predictions.py      对预测 JSONL 做评测
  build_sft_data.py        生成 SFT 数据（可切 dev）
  build_preference_data.py 生成 RFT/DPO 数据
  train_sft.py             QLoRA SFT 训练
  evaluate_model.py        生成预测并算 pass@1
  run_sampling.py          推理时验证：每题采样 k 条候选（GPU）
  analyze_sampling.py      maj@k / oracle / 选择性预测，出表与曲线
  build_demo.py            生成离线 demo 页（demo/index.html）

configs/
  sft_qwen3_4b_24gb.yaml   seq 4096 / r16
  sft_qwen3_4b_12gb.yaml   seq 2048 / r8

docs/
  handoff.md                新会话交接（自包含）
  project-charter.md        项目背景、目标、非目标
  experiment-design.md      数据、方法、实验矩阵、指标
  data-provenance.md        数据来源、许可证、切分、污染说明
  issue-log.md              问题记录（只追加）
  interview-debug-journal.md 面试用调试与决策实录（A/B 数据、反转、讲法）
  first-sampling-results.md 推理时验证第一张结果表（主表、分解、负结果）
  pivot-plan.md             为什么把训练移出关键路径
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

主线（推理时验证）已闭环出表，接下来按收益排序：

1. **温度对照**：T=1.0 / k=8 与 T=0.7 对比，看多样性是否抬高 oracle、是否拖低 maj；
2. **验证器增强**：maj@8 与 oracle pass@8 之间还剩 4.5 点。免费规则（长度、出现顺序、长度加权、相对长度过滤）已实测全部无效，
   下一步需要真正的 verifier / PRM 重排，或继续加大 k 看曲线是否饱和；
3. **选择性预测二维化**：一致度 + 长度/格式完整度，给教育场景更细的「自动批改 / 转人工」阈值；
4. **SFT（加分项）**：换 `QWEN_CHATML_TEMPLATE` 去掉 Qwen3 注入的空 `<think>` 块，重跑 pilot 或全量 2 epochs；
5. **test 最终评测**：主线方法在 test 上各评一次（1319 题，只评一次）。
