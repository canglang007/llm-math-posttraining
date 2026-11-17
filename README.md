# LLM Math Post-Training

一个独立的大模型后训练项目：面向教育场景的私有化数学推理解题助手，用公开 GSM8K/MATH 数据做 SFT、
拒绝采样微调（RFT）、DPO 与 GRPO 偏好优化。

这个项目与已有的 Agent 项目分开，不包含 agent、tool calling、user simulator 或 benchmark 复现。目标是展示**数据构造、SFT、RFT/DPO/GRPO、评测、统计检验和失败分析**的完整链路。

## 主结果一：训练链路（GSM8K dev-200，greedy，同口径）

六个模型全部在**同一批 200 题、同一判分函数、batch=1 greedy** 下评测，
表格由 `scripts/summarize_results.py` 从原始预测 jsonl 重算（`results/`），不是手抄。

| 阶段 | 模型 | 准确率 | 抽取失败 | 训练数据 | vs SFT | McNemar p |
|---|---|---|---|---|---|---|
| E0 | Qwen3-4B-Base | 0.770 (154/200) | 27 (13.5%) | — | −7.5pt | **0.036** |
| E1 | **SFT**（Base+LoRA，400 步） | **0.845 (169/200)** | **1 (0.5%)** | 官方 CoT 6973 条 | 基准 | — |
| E2a | RFT（起点 SFT，114 题） | 0.815 (163/200) | 0 | 自采正确解 412 条 | −3.0pt | 0.286 |
| E2b | RFT（起点 SFT，286 题） | 0.840 (168/200) | 0 | 自采正确解 286 条（每题 1 条） | −0.5pt | 1.000 |
| E3a | DPO（起点 RFT，链式） | 0.845 (169/200) | 0 | 26 对偏好数据 | 0 | — |
| E3b | DPO（起点 SFT，26 对） | 0.860 (172/200) | 0 | 26 对偏好数据 | +1.5pt | 0.581 |
| E3c | **DPO（起点 SFT，79 对）** | **0.865 (173/200)** | 0 | 79 对偏好数据 | **+2.0pt** | 0.424 |
| E4 | GRPO（起点 SFT，全量 120 题） | 0.850 (170/200) | 0 | 在线采样 + 答案规则奖励 | +0.5pt | 1.000 |
| E4b | GRPO（起点 SFT，仅 26 道难题） | 0.835 (167/200) | 0 | 难度筛选后在线采样 | −1.0pt | 0.688 |

三条后训练分支（RFT / DPO / GRPO）**全部从同一个 SFT checkpoint-400 出发**，
差异才能归因到方法本身，而不是上游起点不同。

**四个结论**：

1. **只有 SFT 带来统计显著的提升**（+7.5pt，p=0.036；抽取失败 13.5%→0.5%）。
   三种后训练方法相对 SFT 全部落在噪声内（p>0.28），而 dev-200 的分辨率上限就是 ±2.8pt
   ——200 题本身分不出 1.5pt 的效应。
2. **链式不如平行**：SFT→DPO 0.860，SFT→RFT→DPO 0.845。上游退化会被下游继承。
3. **瓶颈在数据规模与难度匹配，不在算法**：120 道采样题里 73% 模型 4 条候选全对，
   GRPO 组内优势恒 0（`frac_reward_zero_std≈0.67`）；只保留 26 道有区分度的题后
   训练信号确实变强（zero_std 0.33、reward 0.5→0.75），但 dev 反而 0.835——26 题只够过拟合。
4. **数据量决定「学得稳不稳」，信息来源决定「能否超过教师」**：RFT 从 114 题扩到 286 题，
   +2.5pt 但仅追平 SFT（p=1.000）——自我蒸馏的信息上限就是教师；DPO 的偏好标签来自公开答案验证器，
   是外部信号，两个数据量下都超过 SFT（+1.5 / +2.0pt），为目前最好结果 0.865。

完整表、配对四格、负结果分析见 **`docs/main-results.md`**；
每个产物的位置与复现命令见 **`docs/experiment-artifacts.md`**。

## 主结果二：推理时验证（2026-11-15）

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

## 当前状态（2026-11-17）

**主线已闭环**：Base → SFT → 三条后训练分支（RFT / DPO / GRPO）全部训完并同口径评测，
出六档主表（见上），代码、配置、结论全部入库。

- 63 个单测通过（本机需加 `--basetemp=/tmp/mpt-pytest`，沙箱会拦 pytest 的临时目录创建）；
- GSM8K：train 7473 / test 1319；MATH：train 7498 / test 5000（2 条无 boxed 答案已跳过）；
- SFT 数据：GSM8K 6973 train / 500 dev，MATH 6998 train / 500 dev；
- 训练与评测脚本：`train_sft.py`（SFT/RFT 共用）、`train_dpo.py`、`train_grpo.py`、`evaluate_model.py`；
- 数据引擎：`run_sampling.py`（逐题落盘 + `--resume` 断点续跑）、`expand_sampling_candidates.py`、
  `build_preference_data.py`、`build_hard_subset.py`（按难度筛题）；
- 汇总：`summarize_results.py` 从原始 jsonl 重算主表与 McNemar 配对检验，产物在 `results/`。

**两个关键 bug 的修复（都写进了 issue-log）**：

- **停止符**：Qwen3-Base 预训练里从未出现 `<|im_end|>`，ChatML SFT 一边压掉它旧会用的
  `<|endoftext|>`、一边教不会新的，导致生成跑满 512 token、抽取失败 13.5%。
  修复是把训练目标末尾的 `<|im_end|>` 换成 `<|endoftext|>`，推理侧零改动 → 抽取失败降到 0.5%。
- **DPO 学习率**：LoRA 上沿用全参数 DPO 的 `5e-6`，2 步训练参数完全没动
  （loss 恒为 ln(2)、`rewards/margins` 精确为 0）。改 `5e-5` + 3 epochs 后偏好学习才真正生效。

**服务器**：GPU 实例已于 2026-11-17 释放，`/data` 持久卷已保存（6 个 adapter、数据、评测结果、日志）。
本地保留代码 + 评测 jsonl + 汇总；**RFT/DPO 训练数据与训练日志只在持久卷**，
缺口清单与恢复命令见 `docs/experiment-artifacts.md`。

调试与决策过程见 `docs/interview-debug-journal.md` 与 `docs/issue-log.md`（面试用）。

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
  prepare_data.py                下载 GSM8K/MATH（带 provenance）
  eval_predictions.py            对预测 JSONL 做评测
  build_sft_data.py              生成 SFT 数据（可切 dev）
  train_sft.py                   QLoRA SFT 训练（RFT 复用同一脚本）
  train_dpo.py                   DPO 训练（TRL）
  train_grpo.py                  GRPO 训练（在线采样 + 答案规则奖励）
  evaluate_model.py              生成预测并算 pass@1
  run_sampling.py                每题采样 k 条候选（逐题落盘、--resume、进度带 ETA）
  expand_sampling_candidates.py  采样结果展开 + 按公开答案打标
  build_preference_data.py       构造 RFT / DPO 数据
  build_hard_subset.py           按「正确候选数」筛出有区分度的题
  summarize_results.py           从评测 jsonl 重算主表 + McNemar 配对检验 → results/
  audit_data_integrity.py        数据纪律审计：泄漏、唯一 id、评测集合一致性
  analyze_sampling.py            maj@k / oracle / 选择性预测，出表与曲线
  analyze_e0_failures.py         E0 失败分类
  diag_eos_loss.py               一次前向诊断停止符 loss / argmax 排名（停不下来的元凶）
  diag_stop_behavior.py          逐题观察生成是否正常停止
  verify_weighted_loss.py        校验加权 loss 实现
  replay_truncation_fix.py       回放截断修复
  watch_training.py              训练日志进度监控
  build_demo.py                  生成离线 demo 页（demo/index.html）

configs/
  sft_qwen3_4b_12gb_v3.yaml     主线 SFT（seq 2048 / r8 / 400 步，停止符修好后的版本）
  sft_qwen3_4b_24gb.yaml        seq 4096 / r16
  sft_qwen3_4b_12gb.yaml        最早版（已被 v3 取代）
  rft_qwen3_4b_12gb.yaml        拒绝采样微调（起点 sft-v3）
  dpo_qwen3_4b_12gb.yaml        DPO（起点 sft-v3，lr 5e-5）
  grpo_qwen3_4b_16gb.yaml       GRPO 全量 120 题
  grpo_hard26_16gb.yaml         GRPO 难度筛选 26 题（E4b 对照）
  sft_qwen3_4b_12gb_{pilot,pilot_v2,eosfix,stopfix}.yaml  历史失败实验（保留作证据）

results/
  summary.json / summary.csv / pairwise.csv   主表的机器可读版本（脚本生成，勿手改）
  data-audit.json                数据纪律审计结果（泄漏 / 唯一 id / 评测一致性）

docs/
  main-results.md                六档主表、配对检验、三条结论
  experiment-artifacts.md        产物在哪、怎么复现、缺什么
  handoff.md                     新会话交接（自包含）
  project-charter.md             项目背景、目标、非目标
  experiment-design.md           数据、方法、实验矩阵、指标
  data-provenance.md             数据来源、许可证、切分、污染说明
  eos-stop-diagnosis.md          停止符根因诊断与 E1 结果
  issue-log.md                   问题记录（只追加）
  interview-debug-journal.md     面试用调试与决策实录（A/B 数据、反转、讲法）
  first-sampling-results.md      推理时验证第一张结果表（主表、分解、负结果）
  e0-failure-analysis.md         E0 失败案例分析
  mainline-plan.md / pivot-plan.md  主线规划与转向记录
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

## 下一步（按收益排序）

主线已闭环，剩下的都是"把结论做硬"和"补全证据链"：

1. **补齐数据完整性**：拉回 E3a（DPO 链式）的逐题结果、RFT/DPO 训练数据、训练日志到本地，
   清单见 `docs/experiment-artifacts.md` 第四节；
2. **扩 dev 到 500 题重评六档**：分辨率从 ±2.8pt 提到 ±1.8pt，才能判定 DPO 的 +1.5pt 是真效应还是抖动
   （成本约 7 小时，6 个模型 × 500 题）；
3. **扩大采样规模重跑 RFT**：现有 RFT 只用了 412 条（SFT 数据量的 6%），
   扩到 300~500 题 × k=4 才能验证"瓶颈在数据量"这个判断；
4. **test 最终评测**：主线方法在 test 上各评一次（1319 题，只评一次，不参与任何调参）；
5. **MATH 扩展**（加分项）：MATH 数据已备好（6998 train / 500 dev），
   用来检验结论是否能从 GSM8K 迁移到更难的分布。

推理时验证那一路（maj@k、选择性预测）已经出表，若要继续，
方向在 `docs/first-sampling-results.md` 的"下一步"里（温度对照、verifier 重排）。
