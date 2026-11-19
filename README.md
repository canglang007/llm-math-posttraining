# LLM Math Post-Training：一个可量化的后训练数据引擎

用一个 4B 模型（Qwen3-4B-Base）和公开数据（GSM8K），把**"后训练的数据怎么造、怎么验、值多少钱"**
做成可复现、可检验的实验：11 个模型权重（Base / SFT / 3 个 RFT / 3 个 DPO / 2 个 GRPO）、
1319 题 test 终评、每一步都做配对显著性检验，
并且如实报告**9 个方法对比里有 8 个没有显著增益**。

> 一句话主张：**数据引擎的价值有两层——SFT 用官方 CoT 建立格式地基，自采样数据必须过答案验证才能用。**
> 其余所有方法（RFT / DPO / GRPO）在这个规模上都无法稳定超过 SFT。

## 摘要

| 结论 | 数字 | 显著性 |
|---|---|---|
| SFT 是唯一的大效应 | Base 0.7604 → SFT 0.8112，**+5.08pt** | p=0.0002 |
| 采样数据不经验证会伤模型 | 验证版 0.8120 vs 不验证版 0.7945，**+1.74pt** | p=0.022 |
| 其余方法均无显著增益 | 最好的 DPO(26 对) +0.91pt，其余 −1.0~+0.76pt | p≥0.23 |

评测与线上服务统一使用 vLLM（0.34 + bf16 + greedy）。为与历史结果对照，评测前先做口径校准：
vLLM bf16 与历史 HF 4bit 的逐题一致率 84.5%，同精度下为 96.7%——差异主要来自 4bit 量化噪声，
因此两种口径的数字不可混用。

## 主结果：test 1319 题（vLLM 0.34 + bf16 + greedy）

test 集全程未参与训练、过滤、选模或调参，仅用于最终评测一次。

| 阶段 | 模型 / 起点 | test 准确率 | 抽取失败 | 相对 SFT | McNemar p |
|---|---|---|---|---|---|
| E0 | Qwen3-4B-Base | 0.7604 (1003) | 175 (13.3%) | −5.08pt | **0.0002** |
| E1 | **SFT**（Base + LoRA，400 步） | 0.8112 (1070) | 3 (0.2%) | 基准 | — |
| E2a | RFT（起点 SFT，114 题） | 0.8089 (1067) | 1 | −0.23pt | 0.848 |
| E2b | RFT（起点 SFT，286 题，**答案验证**） | 0.8120 (1071) | 0 | +0.08pt | 1.000 |
| E2c | RFT（起点 Base，287 题） | 0.8074 (1065) | 53 (4.0%) | −0.38pt | 0.807 |
| A2 | RFT（起点 SFT，286 题，**不验证**） | **0.7945 (1048)** | 0 | **−1.67pt** | 0.076 |
| E3a | DPO（起点 RFT，链式） | 0.8014 (1057) | 1 | −0.99pt | 0.319 |
| E3b | **DPO（起点 SFT，26 对）** | **0.8203 (1082)** | 2 | +0.91pt | 0.230 |
| E3c | DPO（起点 SFT，79 对） | 0.8188 (1080) | 3 | +0.76pt | 0.363 |
| E4 | GRPO（全量 120 题） | 0.8089 (1067) | 2 | −0.23pt | 0.766 |
| E4b | GRPO（仅 26 道难题） | 0.8112 (1070) | 1 | 0.00pt | 1.000 |

所有后训练分支（RFT / DPO / GRPO）**从同一个 SFT checkpoint-400 出发**，超参一致，
差异才能归因到方法与数据，而不是起点。完整配对四格见 `docs/main-results.md`「口径三」。

## 三条发现及其证据链

### 1. SFT 的价值是「格式地基」，不是「算术能力」

Base 在**能输出可解析答案**的题上正确率 0.8767，反而比 SFT 的 0.8131 高；
但 Base 有 13.3% 的题压根没输出 `####` 结尾。SFT 把可解析率从 86.7% 拉到 99.8%，
test 准确率随之 +5.08pt（p=0.0002）。
→ 后面所有方法的提升，都建立在"先会规范收尾"这个地基上。

### 2. 自采样数据必须过答案验证，否则模型会学错

A2 消融与 E2b **只有一处差别**：同一批候选、同题量（286 题）、同起点、同超参，
A2 直接把候选全用上（其中 37 条、12.9% 是错误解）。结果：

| | E2b（验证） | A2（不验证） | 差值 |
|---|---|---|---|
| test | 0.8120 | 0.7945 | **+1.74pt（p=0.022）** |
| dev-200 | 0.870 | 0.875 | −0.5pt（p=1.000，判不出） |

失败样本证明机制：A2 的典型错误与训练里那些错误解**形态同源**——推理结构正确、局部算术滑落
（`900-2*60` 算成 840；总时长漏掉 10 分钟）。模型不会"自己纠正"数据里的错，而是把它当正确模式学进去。
样例见 `docs/failure-analysis-test.md`。

### 3. RFT / DPO / GRPO 相对 SFT 全部不显著——这是结论，不是失败

9 个对比里最好的 DPO(26 对) 也只有 +0.91pt（p=0.230）。原因可定位：

- **信息来源决定天花板**：RFT 的监督信号来自模型自己（自我蒸馏），上限即教师；
  DPO 的偏好标签来自公开答案验证（外部信号），两个数据量下都略高于 SFT。
- **分布决定有没有东西可学**：120 道采样题里 73% 模型 4 条候选全对，GRPO 的组内优势恒为 0
  （`frac_reward_zero_std≈0.67`）；只留 26 道有区分度的题后信号变强（zero_std 0.33），
  但 26 题只够过拟合（dev 0.835）。
- **链式不如平行**：SFT→DPO 0.8203，SFT→RFT→DPO 0.8014。上游退化会被下游继承。

## 工程与实验方法

| 工程实践 | 具体做法 | 产物 |
|---|---|---|
| **消融设计** | A2 对照组与处理组只差一个变量（是否答案验证），其余全部对齐 | `configs/rft_qwen3_4b_12gb_a2nf.yaml` |
| **口径校准** | 用 vLLM 评测前先与历史 HF 结果逐题对照：同精度 96.7%、跨精度 84.5%，据此判定量化噪声为主因并统一 bf16 口径 | `scripts/compare_inference_backends.py` |
| **双后端评测** | HF（4bit，历史口径）与 vLLM（bf16，现行口径）并存，prompt/停止词/截断完全共用 | `scripts/evaluate_model.py --backend` |
| **统计检验** | 每个对比做 McNemar 配对检验，报 p 值而不是只报均值差 | `docs/main-results.md` |
| **失败分解** | 把"分数"拆成抽取失败率 × 可解析题正确率，区分"不会答"与"不会收尾" | `docs/failure-analysis-test.md` |
| **数据纪律** | test 只评一次；train/dev/test 严格隔离；自采样数据一律由公开答案验证 | `scripts/audit_data_integrity.py` |
| **可复现** | 91 个单测；配置化训练；所有表格由脚本从预测 jsonl 重算，不手抄 | `pytest -q` |

## 局限与结论边界

- **规模小**：单模型（4B）、单数据集（GSM8K）。所有"不显著"的结论只对"这个规模 + dev-200/test"成立，
  不能外推成"DPO 没用"。
- **dev-200 判不出 1pt 级效应**：A2 在 dev 上 p=1.000、在 test 上 p=0.022，是这件事最好的例证。
- **两个口径不可混用**：历史表为 HF 4bit（`docs/main-results.md`「口径一/二」），
  权威表为 vLLM bf16（「口径三」），混用会引入约 15% 的逐题翻转。
- **未做**：多 seed 重复、MATH 跨分布、更大模型。这些是"把结论做宽"，不是"把结论做对"。

## 仓库结构

```
configs/         训练配置（SFT / RFT / DPO / GRPO / A2 对照组，24GB 与 12GB 两套）
scripts/         数据构造、训练、采样、评测、校准、审计、失败分析
src/             数据/答案判定/prompt/训练/评测的库代码
tests/           91 个单测（数据纪律、答案判定、口径守卫、比对逻辑）
experiments/eval/     全部逐题预测 jsonl（**入库**，可直接核对本文每个数字）
experiments/{sft,rft,dpo,grpo}/  模型 adapter 与 checkpoint（**不入库**，体积大；脚本可复现）
deploy/          服务化：serve.sh（vLLM 多 LoRA 一键启动）+ 部署说明（按需启停，非常开）
docs/            main-results（三套口径）、failure-analysis-test、issue-log、设计与数据溯源
demo/comparison.html  可交互逐题对比（11 个模型的分数 + 同题五模型回答），由 scripts/build_comparison_demo.py 生成
demo/live.html        实时对比页：连本地 vLLM 服务，同题并发问 5 个模型（口径与评测一致）
```

## 快速开始

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q          # 91 passed

python scripts/prepare_data.py --dataset gsm8k
python scripts/build_sft_data.py --dataset gsm8k --split train
```

训练（先 dry-run 校验配置与数据，不需要模型）：

```bash
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb.yaml --dry-run
python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb.yaml
```

评测（两条后端，prompt 与判分完全共用）：

```bash
# 历史口径（HF 4bit，batch=1）
python scripts/evaluate_model.py --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400 --force-chat-template \
  --data-file data/raw/gsm8k/test.jsonl --dataset gsm8k --split test \
  --output experiments/eval/test-hf.jsonl

# 现行口径（vLLM bf16，1319 题约 3 分钟）
python scripts/evaluate_model.py --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400 --force-chat-template \
  --data-file data/raw/gsm8k/test.jsonl --dataset gsm8k --split test \
  --backend vllm --vllm-quantization none --output experiments/eval/vllm-test/sft.jsonl
```

换后端前先校准口径：

```bash
python scripts/compare_inference_backends.py \
  --baseline experiments/eval/sft-v3-ckpt400-dev200.jsonl \
  --candidate experiments/eval/sft-v3-ckpt400-dev200-vllm-bf16.jsonl --dataset gsm8k
```

## 服务化（按需启停，非常开）

```bash
bash deploy/serve.sh          # 起 vLLM 服务：base + sft/e2b/a2/dpo 四个 LoRA，OpenAI 兼容接口
curl http://localhost:8000/v1/models   # 验证
```

用途：需要时起一个服务，跑通一条请求即可确认可用；用完即停，不需要常驻。
解码与 prompt 与评测完全一致（greedy、同一 ChatML 模板、同一组 stop 串），
所以线上答案和文档里的分数可以互相对照。详见 `deploy/README.md`。

## 数据与产物在哪

- **逐题预测**：`experiments/eval/`（HF 口径 dev）、`experiments/eval/vllm-dev200/`（bf16 dev）、
  `experiments/eval/vllm-test/`（test 11 个模型）——每条含 `id / question / reference / prediction`，
  可用 `scripts/eval_predictions.py` 重新打分，本文所有准确率与 McNemar p 值都能从这些文件重算。
- **汇总表**：`results/`（summary / pairwise / 数据审计）。
- **模型权重**：不入库。训练配置与脚本在 `configs/`、`scripts/`，按「快速开始」可重训。
- **原始数据**：`data/` 不入库，由 `scripts/prepare_data.py` 从公开 GSM8K 生成。

## 文档导航

| 文档 | 内容 |
|---|---|
| `docs/main-results.md` | 三套口径的完整表格与配对检验（口径一 HF-4bit dev、口径二 bf16 dev、口径三 vLLM test） |
| `docs/failure-analysis-test.md` | test 失败分析：抽取失败、错解污染、DPO 改变的是推理路径 |
| `docs/issue-log.md` | 问题与决策记录（只追加，含流程教训与口径变更） |
| `docs/experiment-design.md` | 数据、方法、实验矩阵、指标定义 |
| `docs/data-provenance.md` | 数据来源、许可证、切分与污染说明 |

## 后续工作

1. **多 seed 重复**：为 A2 vs E2b 的 1.74pt（p=0.022）提供稳健性证据（3 个 seed × 2 组）；
2. **MATH 跨分布**：检验"验证器贡献"是否在更难分布上放大（数据已备：6998 train / 500 dev）；
3. **DPO 扩到 300+ 对**：当前 79 对已到平台期，需要更多外部信号验证"信息来源决定天花板"。
