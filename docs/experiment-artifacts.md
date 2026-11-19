# 实验资产台账（产物在哪、能不能复现、缺什么）

整理时间：2026-11-17 11:40 初版 / 16:47 更新（补齐缺口 + 数据纪律审计）（Asia/Shanghai UTC+08:00）

服务器实例已于 2026-11-17 11:30 释放（保存数据并释放），16:35 在新实例上验证持久卷完好，
并已把本地缺口全部补齐（详见第五、六节）。
本文件回答三个问题：**每个产物在哪、怎么复现、现在缺什么**。

> 入库范围（2026-11-19 起）：**逐题预测 jsonl 入库**（`experiments/eval/`，7.9 MB，可直接核对所有数字）；
> **模型 adapter / checkpoint 不入库**（体积），按本文件的配置与脚本可重训。下表路径均为服务器侧真实路径。

---

## 一、模型 adapter（8 个，全部在服务器持久卷）

| 阶段 | 路径（服务器 `/data/llm-math-posttraining/`） | 起点 | 训练配置 |
|---|---|---|---|
| E1 SFT | `experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400` | Qwen3-4B-Base | `configs/sft_qwen3_4b_12gb_v3.yaml` |
| E2 RFT | `experiments/rft/qwen3-4b-gsm8k-rft` | E1 | `configs/rft_qwen3_4b_12gb.yaml` |
| E3a DPO(链式) | `experiments/dpo/qwen3-4b-gsm8k-dpo` | E2 | `configs/dpo_qwen3_4b_12gb.yaml`（旧版，起点 RFT） |
| E3b DPO(平行) | `experiments/dpo/qwen3-4b-gsm8k-dpo-from-sft` | E1 | `configs/dpo_qwen3_4b_12gb.yaml`（现版，起点 SFT） |
| E4 GRPO | `experiments/grpo/qwen3-4b-gsm8k-grpo` | E1 | `configs/grpo_qwen3_4b_16gb.yaml` |
| E4b GRPO-hard26 | `experiments/grpo/qwen3-4b-gsm8k-grpo-hard26` | E1 | `configs/grpo_hard26_16gb.yaml` |
| E2b RFT-300q | `experiments/rft/qwen3-4b-gsm8k-rft-300q` | E1 | `configs/rft_qwen3_4b_12gb_300q.yaml` |
| E3c DPO-300q | `experiments/dpo/qwen3-4b-gsm8k-dpo-300q` | E1 | `configs/dpo_qwen3_4b_12gb_300q.yaml` |

- 首次释放前已逐个校验 `adapter_model.safetensors`（6/6 OK）；E2b/E3c 为 2026-11-17 扩样新增，产物齐全。
- 本地**不保存**权重（`.gitignore` 已忽略 `*.safetensors`），靠上述路径 + 配置复现。
- 唯一的例外：`data/server-backup/sft/`（411M）是早期 pilot 阶段的旧 adapter，**不是**主线的 sft-v3，仅作历史备份。

## 二、数据文件

| 数据 | 服务器 `/data/llm-math-posttraining/` | 本地仓库 | 说明 |
|---|---|---|---|
| GSM8K 训练（清洗后） | `data/processed/sft/gsm8k-train.jsonl` | ✅ 6.2M | 6973 条，SFT 用 |
| GSM8K dev（清洗后） | `data/processed/sft/gsm8k-dev.raw.jsonl` | ✅ 308K | 500 条，只用于最终评测 |
| 采样候选（120 题 × k=4） | `experiments/sampling/gsm8k-train120-k4-t0.7.jsonl` | ✅ 166K | 480 条候选，RFT/GRPO 数据的源头 |
| 展开后的候选 | `data/processed/rft/gsm8k-train120-candidates.jsonl` | ✅ 270K | 带 is_correct 标记 |
| **RFT 训练集** | `data/processed/rft/gsm8k-train-rft.jsonl` | ✅ 350K | 412 条正确解 |
| **DPO 训练集** | `data/processed/dpo/gsm8k-train-dpo.jsonl` | ✅ 31K | 26 对偏好数据 |
| 难题子集（hard26） | `data/processed/sft/gsm8k-train-hard26.raw.jsonl` | ✅ 7.8K | 0<正确数<4 的 26 题 |
| 采样候选（300 题 × k=4） | `experiments/sampling/gsm8k-train300-k4-t0.7.jsonl` | ✅ 310K | 1200 条候选（含复用前 120 题） |
| 展开后的候选（300 题） | `data/processed/rft/gsm8k-train300-candidates.jsonl` | ✅ 712K | 1144 条，带 is_correct |
| **RFT 训练集（扩样）** | `data/processed/rft/gsm8k-train-rft-300q.jsonl` | ✅ 240K | **286 条 = 286 题**（每题 1 条） |
| **DPO 训练集（扩样）** | `data/processed/dpo/gsm8k-train-dpo-300q.jsonl` | ✅ 97K | **79 对**偏好数据 |

> `data/processed/rft/sample.jsonl`、`data/processed/dpo/sample.jsonl` 是格式样例，**不是**训练数据。

## 三、评测结果（dev-200，greedy，batch=1，同一批题）

| 阶段 | 本地文件 | 状态 |
|---|---|---|
| E0 Base | `data/server-backup/baseline/gsm8k-dev-probe200-final.jsonl` | ✅ |
| E1 SFT | `experiments/eval/sft-v3-ckpt400-dev200.jsonl` | ✅ |
| E2 RFT | `experiments/eval/rft-dev200.jsonl` | ✅ |
| E3a DPO(链式) | `experiments/eval/dpo-dev200.jsonl` | ✅（2026-11-17 16:36 补齐） |
| E3b DPO(平行) | `experiments/eval/dpo-from-sft-dev200.jsonl` | ✅ |
| E4 GRPO | `experiments/eval/grpo-dev200.jsonl` | ✅ |
| E4b GRPO-hard26 | `experiments/eval/grpo-hard26-dev200.jsonl` | ✅ |
| E2b RFT-300q | `experiments/eval/rft300q-dev200.jsonl` | ✅ |
| E3c DPO-300q | `experiments/eval/dpo300q-dev200.jsonl` | ✅ |
| E1 SFT (dev-500) | `experiments/eval/sft-v3-ckpt400-dev500.jsonl` | ✅ |
| E2b RFT-300q (dev-500) | `experiments/eval/rft-300q-dev500.jsonl` | ✅ |
| E3c DPO-300q (dev-500) | `experiments/eval/dpo-300q-dev500.jsonl` | ✅ |

**九档全部有本地逐题文件**，`summarize_results.py` 可完整重算，无 MISSING。

汇总产物（已入库）：
- `results/summary.json` — 机器可读，含全部指标与配对检验
- `results/summary.csv` — 贴表格用
- `results/pairwise.csv` — 逐 run 的 McNemar 配对四格
- `results/data-audit.json` — 数据纪律审计的结构化结果（泄漏、唯一 id、评测一致性）
- `results/dev500/` — dev-500 变体的 summary/pairwise（`--variant dev500` 生成）

**重算方式**（数字不靠手抄）：

```bash
python scripts/summarize_results.py --output-dir results
```

## 四、训练日志（本地备份）

主线 24 份日志已备份到 `data/server-backup/logs/main-run/`：

| 阶段 | 日志 |
|---|---|
| SFT | `train-v3.log`（400 步）、`eval-v3-200.log` |
| 采样 | `rft120.log`（120 题 k=4）、`rft-sampling.log`、`smoke-sampling.log`、`speedtest*.log` |
| RFT | `rft-train.log`（52 步） |
| DPO | `dpo-train.log`（从 RFT）、`dpo-from-sft.log`（从 SFT）、`dpo-eval.log`、`dpo-from-sft-eval.log` |
| GRPO | `grpo-train.log`（30 步）、`grpo-hard26.log`（16 步）、`grpo-smoke.log`、各评测日志 |
| 收官评测 | `final-evals.log`（RFT+DPO 串行那次） |
| 扩样（2026-11-17） | `rft300-sampling.log`（300 题采样）、`rft300-train.log`（36 步）、`rft300-eval.log`、`dpo300-train.log`（30 步）、`dpo300-eval.log` |

（服务器 `/data/*.log` 上还留着早期 pilot/测速日志，未逐份备份，结论已进 issue-log。）
扩样实验的 5 份日志（`rft300-*`、`dpo300-*`）与 dev-500 的 3 份评测日志（`dev500-*`）也已备份。

## 五、数据纪律审计（2026-11-17 16:45，PASS）

```bash
python scripts/audit_data_integrity.py --json results/data-audit.json
```

| 检查项 | 结果 |
|---|---|
| dev ∩ train / 采样 / RFT / DPO / hard26 | 0 / 0 / 0 / 0 / 0（无泄漏） |
| 采样 / RFT / DPO / hard26 的 id 是否全在 train 里 | 全部 0 个越界 |
| 7 份评测是否评同一批题 | 是，均为 200 题 |
| 评测题 ⊆ dev | 是 |
| 评测题 ∩ train | 0 |

**审计暴露的一个重要事实**：RFT 训练集 412 行里只有 **114 个唯一 id**（平均 3.6 条/题）——
同一道题的多条正确解全部保留，导致「条数」虚高 3.6 倍。
有效样本量只相当于 SFT 训练集的 1.6%（114 / 6973）。
详见 `docs/main-results.md`「数据规模全景」。

## 六、已知缺口

1. ~~E3a 逐题结果~~ — **已补齐**（2026-11-17 16:36），重算得 169/200，与文档一致；配对 11 对 / 11 错、p=1.000，净差 0。
2. ~~RFT/DPO 训练数据只在服务器~~ — **已补齐**到本地 `data/processed/{rft,dpo}/`。
3. ~~训练日志未备份~~ — 主线 19 份已备份（见第四节）。
4. **dev 只评了 200 题**（实际有 500 题）。若要把分辨率从 ±2.8pt 提到 ±1.8pt，需扩到 500 题重跑 7 个模型。
5. **模型权重不在本地**（只在国内服务器持久卷），靠配置 + 数据可复现。

## 五、复现一条完整的实验链（命令骨架）

```bash
# 0) 环境
export PATH=/data/miniconda/bin:$PATH            # tmux 已装进持久卷
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4       # 容器 CPU 配额有限

# 1) SFT（从 Base）
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb_v3.yaml

# 2) 采样（train 120 题 × k=4）→ 构造 RFT/DPO 数据
python scripts/run_sampling.py --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400 \
  --data-file data/processed/sft/gsm8k-train.raw.jsonl --dataset gsm8k \
  --split train --limit 120 --k 4 --no-stop-strings --resume \
  --output experiments/sampling/gsm8k-train120-k4-t0.7.jsonl
python scripts/expand_sampling_candidates.py --sampling-file <上面产物> --split train --output <候选>
python scripts/build_preference_data.py --candidates <候选> \
  --rft-output data/processed/rft/gsm8k-train-rft.jsonl \
  --dpo-output data/processed/dpo/gsm8k-train-dpo.jsonl

# 3) 三条平行分支（都从 sft-v3/checkpoint-400 出发）
python scripts/train_sft.py --config configs/rft_qwen3_4b_12gb.yaml     # RFT
python scripts/train_dpo.py --config configs/dpo_qwen3_4b_12gb.yaml     # DPO
python scripts/train_grpo.py --config configs/grpo_qwen3_4b_16gb.yaml   # GRPO

# 4) 统一评测（dev-200，greedy，batch=1）
python scripts/evaluate_model.py --model-path /data/models/Qwen3-4B-Base \
  --adapter <某个 adapter> --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
  --dataset gsm8k --limit 200 --batch-size 1 --output experiments/eval/<name>.jsonl

# 5) 汇总（可复现出表）
python scripts/summarize_results.py --output-dir results
```
