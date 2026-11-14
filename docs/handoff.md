# Handoff：LLM Math Post-Training

更新时间：2026-11-15 07:14（Asia/Shanghai，UTC+08:00）

本文自包含，新会话不需要任何旧项目上下文。按本文和第 11 节的阅读顺序即可继续。

## 1. 项目目标

做一个独立的大模型后训练项目：面向教育场景的私有化数学推理解题助手。

- 输入：数学题；
- 输出：分步解答 + 最终答案；
- 核心：公开 GSM8K/MATH 数据 → 数据引擎 → SFT → 拒绝采样（RFT）→ DPO → 公开 test 评测；
- 主张：答案验证的过滤/配比/偏好优化能提升 Qwen3-4B-Base 的数学推理 pass@1，并单独量化数据引擎的贡献。

项目路径：

```text
/Users/canglang007/Desktop/work/me/job/project/llm-math-posttraining
```

## 2. 核心原则

1. **公开数据是主证据**：主训练/验证/测试来自 GSM8K/MATH（MIT），有官方切分和许可证。
2. **合成数据只做增广/消融**：RFT/DPO 的候选必须由公开答案判对错；不自造 ground truth 当主证据。
3. **test 隔离**：test 不参与训练、过滤、选模或调参；dev 从 train 切出。
4. **不做 Agent/工具/benchmark 复现**：本项目只做 LLM 后训练，不引入 tool calling、user simulator、agent loop。
5. **先本地 foundation，再上 GPU**：代码、数据格式、答案抽取、评测、单测先跑通；Mac 不加载模型。
6. **先测 baseline，再训练**：Qwen3-4B-Base 的 headroom 未知，先跑小规模 baseline。

## 3. 当前设计

A：私有化数学推理解题助手（教育场景，成本/隐私/可控）。

B：后训练数据引擎：

- 清洗、去重、难度分层（步数/长度/模型 pass rate）、配比；
- 同题采样多条解答，用公开答案做 rejection sampling；
- 同题正确/错误解答构造 DPO 偏好对；
- 数据量、过滤比例、配比做消融。

实验矩阵：

| ID | 方法 | 数据 |
|---|---|---|
| E0 | Base | 无 |
| E1 | Public-SFT | 官方 train CoT |
| E2 | RFT | E1 采样，答案正确才保留 |
| E3 | RFT + DPO | 同题正确/错误配对 |
| E4（可选） | Teacher distill | 强模型生成，答案过滤 |
| A1 | 数据消融 | 去重/难度/配比 |
| A2 | 过滤消融 | 全部采样 vs 只保留正确 |

## 4. 当前进度

已提交：

```text
458d159 初始化数学推理后训练项目基础链路
```

已完成：

- 项目骨架、数据格式、答案抽取、prompt、pass@1 评测、dev 切分；
- SFT / RFT / DPO 数据构造；
- QLoRA SFT 训练脚本 + 12GB/24GB 配置；
- 本机评测脚本（Base/adapter → predictions → pass@1）；
- 28 个本地单测通过；
- GSM8K：raw train 7473 / test 1319；
- MATH：raw train 7498 / test 5000（train 2 条无 `\boxed{}` 答案已跳过）；
- SFT 数据：
  - GSM8K：6973 train / 500 dev，dev 原始文件 `data/processed/sft/gsm8k-dev.raw.jsonl`；
  - MATH：6998 train / 500 dev，dev 原始文件 `data/processed/sft/math-dev.raw.jsonl`；
- RFT/DPO 构造器在小样本验证（RFT 2 条、DPO 1 对）。

未提交（截至本 handoff）：

- SFT 训练层、configs、evaluate_model、本 handoff；
- `git status` 可查看；提交时使用中文消息，不要提交 `data/`、模型或实验产物。

## 5. 当前未完成 / 阻塞

- Qwen3-4B-Base 未下载到本地；训练配置里的路径是 `/data/models/Qwen3-4B-Base`。
- GPU 服务器目前关闭；没有运行过训练。
- Base baseline 尚未测，headroom 未知。
- RFT/DPO 还没有真实候选轨迹（需要模型采样）。

## 6. 下一步顺序

1. **review + 提交**：看 `git status`，确认 SFT 层和 handoff 后提交。
2. **GPU 就绪后先测 baseline**：
   - 下载 `Qwen/Qwen3-4B-Base` 到 `/data/models/Qwen3-4B-Base`；
   - 用 `evaluate_model.py` 在 GSM8K test 200 题上跑 Base（无 adapter）；
   - 看 pass@1、答案抽取失败率、格式错误；决定用 GSM8K 还是 MATH/更难子集。
3. **SFT**：
   - 先 `train_sft.py --config ... --dry-run`；
   - 用 12GB 或 24GB 配置训练；
   - 每个 epoch checkpoint 用 `evaluate_model.py` 在 dev.raw 上算 pass@1，选最好；
   - test 只评一次。
4. **RFT/DPO**：从 SFT 采样候选，用公开答案验证；生成 RFT/DPO 数据；做 A1/A2 消融。
5. **失败分析 + demo + 报告**。

## 7. 关键命令

本地测试：

```bash
cd /Users/canglang007/Desktop/work/me/job/project/llm-math-posttraining
/opt/anaconda3/bin/python3 -m pytest -q     # 28 passed
```

重建数据（换机器时；`data/` 被 gitignore）：

```bash
python scripts/prepare_data.py --dataset gsm8k
python scripts/prepare_data.py --dataset math
python scripts/build_sft_data.py --dataset gsm8k --split train --dev-size 500
python scripts/build_sft_data.py --dataset math --split train --dev-size 500
```

SFT dry-run：

```bash
python scripts/train_sft.py --config configs/sft_qwen3_4b_24gb.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb.yaml --dry-run
```

服务器训练/评测：

```bash
hf download Qwen/Qwen3-4B-Base --local-dir /data/models/Qwen3-4B-Base

# baseline
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --data-file data/raw/gsm8k/test.jsonl \
  --dataset gsm8k --limit 200

# SFT
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb.yaml

# dev 选 checkpoint
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-12gb/checkpoint-XXX \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
  --dataset gsm8k

# test 只跑一次
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-12gb \
  --data-file data/raw/gsm8k/test.jsonl \
  --dataset gsm8k
```

## 8. 关键文件

- `src/math_posttrain/answers.py`：GSM8K `####` / MATH `\boxed{}` 抽取与等价判断
- `src/math_posttrain/prompts.py`：prompt/chat messages；Qwen ChatML fallback
- `src/math_posttrain/evaluation.py`：pass@1、抽取失败、分类指标
- `src/math_posttrain/sft.py`：SFT record、assistant-only mask
- `src/math_posttrain/preferences.py`：RFT/DPO，答案验证
- `src/math_posttrain/training.py`：QLoRA、SFTDataset、Trainer
- `scripts/prepare_data.py`、`build_sft_data.py`、`build_preference_data.py`
- `scripts/train_sft.py`、`evaluate_model.py`、`eval_predictions.py`
- `configs/sft_qwen3_4b_12gb.yaml`、`configs/sft_qwen3_4b_24gb.yaml`
- `docs/experiment-design.md`、`docs/data-provenance.md`、`docs/issue-log.md`

## 9. 环境与已知问题

- 新机器安装：`python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev,eval,train]'`。
- Mac 用 `/opt/anaconda3/bin/python3` 跑测试（Homebrew python3.14 没有 pytest，且 pyproject 要求 <3.14）。
- 数据下载：`prepare_data.py` 默认 `HF_ENDPOINT=https://hf-mirror.com`；会收紧 `NO_PROXY` 规避 `[::1]` 的 httpx 解析错误。
- MATH HF 字段是 `problem/solution/level/type`，没有 `answer`；脚本从 solution 的 `\boxed{}` 抽答案。
- Base tokenizer 可能没有 chat template；训练/评测会自动使用 Qwen ChatML fallback。
- SFT 只对 assistant 片段算 loss；prompt 不是 full 前缀时会直接报错，不会静默训错。
- 12GB 配置 seq 2048/r8；24GB 配置 seq 4096/r16；12GB OOM 时先降 seq/r。
- GSM8K/MATH 可能有预训练污染，报告必须写明；可加 GSM-Symbolic/扰动测试。
- Base headroom 未知，先测 baseline，不要直接训练。

## 10. 不要做

- 不要引入 agent、tool calling、user simulator、agent benchmark 复现。
- 不要用自造标签当 ground truth；RFT/DPO 必须由公开答案验证。
- 不要用 test 训练、过滤、选 checkpoint 或调参。
- 不要在 Mac 上加载模型/训练；Mac 只做开发和单测。
- 不要把 `data/`、模型权重、实验输出提交到 git。
- 不要在没测 baseline 前直接跑 SFT。

## 11. 新会话阅读顺序

1. 本文件；
2. `README.md`；
3. `docs/experiment-design.md`；
4. `docs/data-provenance.md`；
5. `docs/issue-log.md`；
6. `git status` 看未提交改动。

然后按第 6 节顺序继续。
