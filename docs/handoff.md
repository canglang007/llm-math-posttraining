# Handoff：LLM Math Post-Training

更新时间：2026-11-15 20:20（Asia/Shanghai，UTC+08:00）

本文自包含，新会话不需要任何旧项目上下文。按本文和第 11 节的阅读顺序即可继续。

## 1. 项目目标

做一个独立的大模型后训练项目：面向教育场景的私有化数学推理解题助手。

- 输入：数学题；
- 输出：分步解答 + 最终答案；
- 核心：公开 GSM8K/MATH 数据 → 数据引擎 → SFT → 拒绝采样（RFT）→ DPO → 公开 test 评测；
- 主张：答案验证的过滤/配比/偏好优化能提升 Qwen3-4B-Base 的数学推理 pass@1，并单独量化数据引擎的贡献。

**项目动机（2026-11-15 用户确认）**：为找实习面试准备的作品，教育场景是叙事载体。第一优先是「主链路闭环 + 可复现数字 + 能单独量化数据引擎贡献的消融 + 失败分析 + 可演示 demo」，实验规模以讲得清为准，不追求刷榜或堆方法。优先级：E0→E1→E2 主线 > A1/A2 消融 > E3 DPO / E4 蒸馏 / MATH 全量扩展；时间或算力不足时砍加分项，但不能停在「只有脚本没有结果」。

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

已提交（2026-11-15 09:28 核对，工作区干净）：基础链路、SFT/评测层、评测隔离与生成停止修复、性能实测、E0 baseline 结果和面试调试实录均已入库；完整历史用 `git log --oneline` 查看。

已完成：

- 项目骨架、数据格式、答案抽取、prompt、pass@1 评测、dev 切分；
- SFT / RFT / DPO 数据构造；
- QLoRA SFT 训练脚本 + 12GB/24GB 配置；
- 本机评测脚本（Base/adapter → predictions → pass@1）；
- 33 个单测通过（本地与服务器一致）；
- GSM8K：raw train 7473 / test 1319；
- MATH：raw train 7498 / test 5000（train 2 条无 `\boxed{}` 答案已跳过）；
- SFT 数据：
  - GSM8K：6973 train / 500 dev，dev 原始文件 `data/processed/sft/gsm8k-dev.raw.jsonl`；
  - MATH：6998 train / 500 dev，dev 原始文件 `data/processed/sft/math-dev.raw.jsonl`；
- RFT/DPO 构造器在小样本验证（RFT 2 条、DPO 1 对）；
- E0 baseline（dev 200 题，batch=1 正式口径，2026-11-15 09:26）：pass@1 **0.77**（154/200）、抽取成功但错 19、抽取失败 27（13.5%）、空输出 0；预测 `experiments/baseline/gsm8k-dev-probe200-final.jsonl`，日志 `/data/probe-v3.log`；
- 面试用调试与决策实录 `docs/interview-debug-journal.md`（时间线、A/B 数字、两次优化反转、1 分钟讲法、Q&A，已补第 7 个决策「长训练可恢复点」）；
- E1 pilot SFT（200 步 / seq 1024 / seed 42，2026-11-15 15:33 完成）：`train_loss ≈ 0.490`，adapter 66MB 与 `checkpoint-50/100/150/200` 在 `experiments/sft/qwen3-4b-gsm8k-pilot200/`；配置 `configs/sft_qwen3_4b_12gb_pilot.yaml`，日志 `/data/sft-pilot-v2.log`；
- E0 同批前 50 题基准：0.76（38/50）、抽取失败 9；E1 同批 50 题评测 15:34 启动，因服务器断开未完成。
- **推理时验证主线已有第一张正式结果表（2026-11-15 20:08 采样完成，20:15 出表）**：模型冻结不训练，Qwen3-4B-Base 在 dev 200 题上 k=8、T=0.7 采样 52 分钟，
  greedy 0.770 → **maj@8 0.920**（184/200），救回 31 题、带偏 1 题；maj@3 已到 0.880，oracle pass@8 = 0.965（验证器天花板 +4.5 点）；
  按投票一致度做选择性预测：84% 题目一致度 ≥75%、准确率 97.6%，剩余 16% 准确率仅 45%–71%，可转人工。
  详见 `docs/first-sampling-results.md`；候选 `experiments/sampling/gsm8k-dev-probe200-k8-t0.7.jsonl`，报告与图 `experiments/analysis/sampling-k8-t0.7.{json,md,png}`。
- 新增离线分析链路（Mac 可跑，不加载模型）：`scripts/analyze_sampling.py` + `tests/test_analyze_sampling.py`（含 maj@k / oracle / Wilson 区间 / E0 分层 / 一致度选择性预测 / 长度过滤对照 / 曲线 PNG）。
- 全项目单测 48 项通过（注意本机需加 `--basetemp=/tmp/mpt-pytest`，否则沙箱会把 pytest 临时目录 mkdir 拦成 EEXIST）。

状态修正：原 handoff 记为「未提交」的 SFT 训练层、configs、`evaluate_model.py`、handoff 文档，均已随 `2b7484d` 提交。后续提交仍用中文消息，不要提交 `data/`、模型或实验产物。

## 5. 当前未完成 / 阻塞

- **服务器在线（2026-11-15 19:07 换卡重开）**：RTX 4060 Ti **16GB**（与之前同型号）、驱动随镜像、96 核 / 125G 内存；`/data` 持久卷 89G（余 58G）内容完好，conda 环境 `/data/miniconda/envs/torch`（torch 2.10.0+cu128）可用，代码已用 `git bundle` 增量同步到 `3a21aef` 之后。system 盘上的免密公钥与 tmux 会随实例重建丢失，恢复清单见第 9 节。凭据不写入仓库，由用户当次提供。
- `/data/models/Qwen3-4B-Base` 与后训练版 `Qwen3-4B` 都在；**`experiments/sft/` 在本次重建后为空**，E1 pilot 的 adapter/checkpoint 只剩本地备份 `data/server-backup/sft/`（413M），需要时可回传。
- E0 baseline 已测（dev 0.77），headroom 有限，27 个失败多为输出格式未对齐；**E1 pilot 训练完成但模型退化不可用**（200 步欠拟合，输出以乱码开头、跑满 512 tokens 不生成 `<|im_end|>`），同批 50 题评测已停止，不能用于 E0/E1 对比。
- 推理时验证主线**不依赖任何训练产物**，已在第一张表上闭环；T=1.0 的温度对照采样于 2026-11-15 20:10 启动（同为 200×k=8，约 52 分钟），用于检验「提高多样性能否抬高 oracle、是否拖低 maj」。
- 全量 E1（2 epochs，约 3h20m）尚未跑；RFT/DPO 还没有真实候选轨迹（需要模型采样）。

## 6. 下一步顺序

**主线已切到推理时验证**（见 `docs/pivot-plan.md`），训练降为加分项；下面的顺序按此排列。

1. **推理时验证第一张表（已完成，2026-11-15 20:15）**：greedy 0.770 → maj@8 0.920（+30 题）、oracle pass@8 0.965、一致度选择性预测 84% / 97.6%；见 `docs/first-sampling-results.md`。
2. **温度对照（进行中）**：T=1.0、k=8、同 200 题，与 T=0.7 对比 oracle（多样性是否升高）与 maj（投票是否被带偏）；跑完用同一条 `analyze_sampling.py` 命令出第二张表，两行合成温度消融。
3. **验证器增强（下一步的主攻方向）**：maj@8 与 oracle pass@8 之间还有 4.5 个点，且 8 道题「池里有正确答案但投票投错」。
   - 可做的低成本方案：候选重排（按 CoT 长度/格式完整度）、答案一致性以外的信号（`####` 前的推理长度、是否出现 `<<>>` 计算链）、平票规则显式化；
   - 目标是补上这 4.5 点，并给出「验证器 vs 投票」的消融行。
4. **选择性预测（已有结果，可扩展）**：当前用投票一致度分 4 桶；可加「一致度 + 长度」的二维分桶，给教育场景的「自动批改 / 转人工」阈值。
5. **SFT / E1（降为加分项）**：pilot 200 步欠拟合不可用；若要补训练硬证据，优先换 `QWEN_CHATML_TEMPLATE`（去掉注入的 `<think>` 空块）重跑 pilot（约 46 分钟）或直接全量 2 epochs（约 3h20m）。成功就加一行 E1，失败不动主表。
6. **RFT/DPO**：候选已可由采样产出，若做 RFT 可直接复用 `experiments/sampling/` 的候选并用公开答案验证。
7. **失败分析 + demo + 报告**：以 `docs/first-sampling-results.md` 为骨架扩展。

## 7. 关键命令

本地测试：

```bash
cd /Users/canglang007/Desktop/work/me/job/project/llm-math-posttraining
/opt/anaconda3/bin/python3 -m pytest -q --basetemp=/tmp/mpt-pytest     # 48 passed
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

# baseline probe（dev，不碰 test）
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
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
- 服务器凭据不写入仓库；工作区 `/data/llm-math-posttraining`，长任务用 `tmux`（会话 `bootstrap`），日志 `/data/bootstrap.log`。pip 用清华源（`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`），HF 用 `HF_ENDPOINT=https://hf-mirror.com`；pypi.org 直连很慢，会卡在 build dependencies。
- 12GB 卡对应 `configs/sft_qwen3_4b_12gb.yaml`（seq 2048/r8）；24GB 配置在这台服务器上不适用。
- conda 环境 `torch` 预置 numpy 1.26.4，与 datasets 4.x 要求的 pyarrow>=21 冲突（会导致 `import transformers` 失败）；已统一升级到 numpy 2.5.4 + pyarrow 26.0.1，详见 issue-log。服务器自检除 `pytest -q` 外，还要做一次训练依赖全量 import。
- `evaluate_model.py` 生成时已合并 `<|im_end|>`/`<|im_start|>` 到 eos，并加字符串停止词与生成后截断；Base 的 generation_config 只认 `<|endoftext|>`（151643），不修会跑满 max_new_tokens 并续写下一轮对话，污染 `####` 答案抽取，详见 issue-log。
- 正式评测固定 `--batch-size 1`：`>1` 会因左填充与 4bit 数值路径改变 greedy 结果（实测 5 题 accuracy 0.6→0.4），且 GPU 利用率低是 batch=1 自回归 decode 的正常形态，不是故障。`evaluate_model.py` 每 10 题打印 `[progress] n/total`，便于在 tmux 里看进度。
- **服务器重启恢复清单**（2026-11-15 13:43 实测）：host/port/密码会变，`/data` 持久盘内容全部保留，但 `/root/.ssh/authorized_keys` 和 apt 装的 tmux 在系统盘会丢。开机后：① 用新凭据把本机 `~/.ssh/id_ed25519.pub` 写入服务器 `~/.ssh/authorized_keys`（目录 700、文件 600）；② `apt-get update -qq && apt-get install -y -qq tmux`；③ 自检 `pytest -q`、训练依赖全量 import、`train_sft.py --dry-run`。凭据不写入仓库。镜像恢复后**第一优先级**：确认 `experiments/sft/qwen3-4b-gsm8k-pilot200/` 存在，把 adapter 与 checkpoint 回传本地 `data/server-backup/sft/`；本地有完整配置/数据/单测，最坏可用 seed 42 重训 46 分钟复现。

## 10. 不要做

- 不要引入 agent、tool calling、user simulator、agent benchmark 复现。
- 不要用自造标签当 ground truth；RFT/DPO 必须由公开答案验证。
- 不要用 test 训练、过滤、选 checkpoint 或调参。
- 不要在 Mac 上加载模型/训练；Mac 只做开发和单测。
- 不要把 `data/`、模型权重、实验输出提交到 git。
- 不要在没测 baseline 前直接跑 SFT。
- 不要用 test 做 baseline probe、prompt 调试或数据集选择；这类探针一律用 dev。

## 11. 新会话阅读顺序

1. 本文件；
2. `README.md`；
3. `docs/experiment-design.md`；
4. `docs/data-provenance.md`；
5. `docs/issue-log.md`；
6. `git status` 看未提交改动。

然后按第 6 节顺序继续。
