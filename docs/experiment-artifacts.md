# 实验资产台账（产物在哪、能不能复现、缺什么）

整理时间：2026-11-17 11:40（Asia/Shanghai UTC+08:00）

服务器实例已于 2026-11-17 11:30 释放，数据盘 `/data` 已选择「保存数据并释放」。
本文件回答三个问题：**每个产物在哪、怎么复现、现在缺什么**。

---

## 一、模型 adapter（6 个，全部在服务器持久卷）

| 阶段 | 路径（服务器 `/data/llm-math-posttraining/`） | 起点 | 训练配置 |
|---|---|---|---|
| E1 SFT | `experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400` | Qwen3-4B-Base | `configs/sft_qwen3_4b_12gb_v3.yaml` |
| E2 RFT | `experiments/rft/qwen3-4b-gsm8k-rft` | E1 | `configs/rft_qwen3_4b_12gb.yaml` |
| E3a DPO(链式) | `experiments/dpo/qwen3-4b-gsm8k-dpo` | E2 | `configs/dpo_qwen3_4b_12gb.yaml`（旧版，起点 RFT） |
| E3b DPO(平行) | `experiments/dpo/qwen3-4b-gsm8k-dpo-from-sft` | E1 | `configs/dpo_qwen3_4b_12gb.yaml`（现版，起点 SFT） |
| E4 GRPO | `experiments/grpo/qwen3-4b-gsm8k-grpo` | E1 | `configs/grpo_qwen3_4b_16gb.yaml` |
| E4b GRPO-hard26 | `experiments/grpo/qwen3-4b-gsm8k-grpo-hard26` | E1 | `configs/grpo_hard26_16gb.yaml` |

- 磁盘释放前已逐个校验 `adapter_model.safetensors` 存在（6/6 OK）。
- 本地**不保存**权重（`.gitignore` 已忽略 `*.safetensors`），靠上述路径 + 配置复现。
- 唯一的例外：`data/server-backup/sft/`（411M）是早期 pilot 阶段的旧 adapter，**不是**主线的 sft-v3，仅作历史备份。

## 二、数据文件

| 数据 | 服务器 `/data/llm-math-posttraining/` | 本地仓库 | 说明 |
|---|---|---|---|
| GSM8K 训练（清洗后） | `data/processed/sft/gsm8k-train.jsonl` | ✅ 6.2M | 6973 条，SFT 用 |
| GSM8K dev（清洗后） | `data/processed/sft/gsm8k-dev.raw.jsonl` | ✅ 308K | 500 条，只用于最终评测 |
| 采样候选（120 题 × k=4） | `experiments/sampling/gsm8k-train120-k4-t0.7.jsonl` | ❌ | 480 条候选，RFT/GRPO 数据的源头 |
| 展开后的候选 | `data/processed/rft/gsm8k-train120-candidates.jsonl` | ❌ | 带 is_correct 标记 |
| **RFT 训练集** | `data/processed/rft/gsm8k-train-rft.jsonl` | ❌（本地只有 `sample.jsonl` 占位） | 412 条正确解 |
| **DPO 训练集** | `data/processed/dpo/gsm8k-train-dpo.jsonl` | ❌（本地只有 `sample.jsonl` 占位） | 26 对偏好数据 |
| 难题子集（hard26） | `data/processed/sft/gsm8k-train-hard26.raw.jsonl` | ❌ | 0<正确数<4 的 26 题 |

> 注意：`data/processed/rft/sample.jsonl`、`data/processed/dpo/sample.jsonl` 是格式样例，**不是**训练数据。

## 三、评测结果（dev-200，greedy，batch=1，同一批题）

| 阶段 | 本地文件 | 状态 |
|---|---|---|
| E0 Base | `data/server-backup/baseline/gsm8k-dev-probe200-final.jsonl` | ✅ |
| E1 SFT | `experiments/eval/sft-v3-ckpt400-dev200.jsonl` | ✅ |
| E2 RFT | `experiments/eval/rft-dev200.jsonl` | ✅ |
| E3a DPO(链式) | — | ❌ **本地缺**（服务器上有 `experiments/eval/dpo-dev200.jsonl`） |
| E3b DPO(平行) | `experiments/eval/dpo-from-sft-dev200.jsonl` | ✅ |
| E4 GRPO | `experiments/eval/grpo-dev200.jsonl` | ✅ |
| E4b GRPO-hard26 | `experiments/eval/grpo-hard26-dev200.jsonl` | ✅ |

汇总产物（本次整理新增，已入库）：
- `results/summary.json` — 机器可读，含全部指标与配对检验
- `results/summary.csv` — 贴表格用
- `results/pairwise.csv` — 逐 run 的 McNemar 配对四格

**重算方式**（数字不靠手抄）：

```bash
python scripts/summarize_results.py --output-dir results
```

## 四、已知缺口（下次开机待补）

1. **E3a DPO(链式) 的逐题结果没拉回本地**。分数 0.845（169/200）来自评测日志，本地无 jsonl，`summarize_results.py` 会把它标成 `MISSING`。
   补法：开机后 `scp /data/llm-math-posttraining/experiments/eval/dpo-dev200.jsonl experiments/eval/`。
2. **RFT / DPO / hard26 训练数据只在服务器**。若要离线复现或做数据分析，需按第二节拉回（合计约几 MB）。
3. **训练日志只在服务器 `/data/*.log`**（`train-v3.log`、`rft-train.log`、`dpo-train.log`、`dpo-from-sft.log`、`grpo-train.log`、`grpo-hard26.log`、各评测日志）。关键数字已摘进 `docs/issue-log.md` 与 `docs/main-results.md`，原始日志未备份到本地。
4. **dev 只评了 200 题**（实际有 500 题）。若要把分辨率从 ±2.8pt 提到 ±1.8pt，需扩到 500 题重跑 6 个模型。

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
