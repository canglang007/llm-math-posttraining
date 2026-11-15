# 方向调整方案：从「后训练」到「推理时验证」

更新时间：2026-11-15 18:15（Asia/Shanghai，UTC+08:00）

状态：**待确认**。本文只提出方案与已完成的基础模块，不改动
`docs/project-charter.md`、`docs/experiment-design.md` 和 `AGENTS.md`；确认后一并更新。

## 为什么调

- 项目脉络是 `tracebench → agent-tool-learning → 本项目的 SFT 主线`，两年内两次转向，
  但每次卡住的都是同一件事：**12GB 单机把 QLoRA 训练跑出可用结果**。
- 2026-11-15 实测：全量 SFT 需 3h20m；200 步 pilot 欠拟合、输出退化；Trainer 的
  epoch 末 eval 曾 OOM；实例两次因余额/释放中断。「再试一次训练」的不确定性最高。
- 用户诉求：确定性交付，早点收尾去复习，不要反复重试。
- 结论：**保留项目与全部已有资产，把「训练」从关键路径移出**，而不是第三次换题。

## 新主张

> 在固定模型与公开答案的前提下，推理时的采样 + 答案验证/投票，
> 能以可量化的成本换来多少 pass@1 提升？数据引擎（采样数、过滤、难度配比）贡献多少？

不再声称「SFT/RFT 提升模型能力」。模型权重冻结（Qwen3-4B-Base 4bit），
所有增益来自推理时计算与验证信号，因而**不会出现训坏、欠拟合、训练 OOM**。

## 可复用资产（零重做）

- GSM8K 公开 train/dev/test、官方切分与 provenance；
- `answers.py` 答案抽取与等价判断、`evaluation.py` 的 pass@1 / 抽取失败率；
- E0 baseline：dev 200 题 greedy **0.77**（154/200）、抽取失败 27，
  `experiments/baseline/gsm8k-dev-probe200-final.jsonl`；
- 27 个失败案例的分类、`prompts.py` 的生成停止修复、`--batch-size 1` 固定口径；
- 面试调试实录、issue-log、全部日志与本地权重备份（`data/server-backup/`）。

## 实验矩阵（全部在 dev；test 只最终评一次）

| ID | 方法 | 变量 |
|---|---|---|
| E0 | greedy（已有） | 基线 0.77 |
| T1 | temperature 采样 k=1/3/5/8 | 采样数 |
| T2 | majority vote（抽取答案归一化后投票） | 聚合策略 |
| T3 | verifier 过滤：只保留格式合法、抽取成功的候选 | 过滤规则 |
| T4 | oracle pass@k（上界，说明验证信号天花板） | 参考线 |
| A1 | 难度分层配比（按 E0 对错 / 输出长度分层） | 数据引擎贡献 |
| A2 | 过滤 vs 不过滤、投票 vs 单条 | 消融 |

指标：pass@1 / maj@k / oracle pass@k、抽取失败率、平均有效候选数、
生成 tokens 与耗时（成本）、按 E0 对错分层的增益。

## 交付与时间盒（2–3 天）

1. **Day 1（Mac，不加载模型）**：`src/math_posttrain/inference_time.py`（投票 / 聚合 /
   成本，纯逻辑）+ 单测（已完成）；`scripts/run_sampling.py`（服务器 GPU 采样）与
   `scripts/analyze_sampling.py`（出表）。
2. **Day 2（服务器一次开机，1–2 小时）**：对同一 dev 200 题跑 k=8 采样
   （固定 seed / temperature / top_p），落盘 JSONL；只做推理，跑完即可关机。
3. **Day 3（Mac）**：结果表 + 失败分析 + 更新面试讲法与 demo，收尾。

兜底：若不再开服务器，可退化为纯 Mac 的「答案判分可信度」研究，
用 27 个失败案例 + 格式扰动测试出表，同样不需要 GPU。

## 训练证据怎么处理

- 移出主线：主结果不依赖任何训练，训练跑不出来也不影响交付。
- 若后续仍想补「训练」硬证据，作为加分项单独跑全量 SFT（修 think 块 + 2 epochs），
  成功就加一行 E1，失败也不动主表。
- `agent-tool-learning` 里同样缺的 SFT 另案处理，不与本项目捆绑。

## 边界与需要同步修改的约定

- 主证据仍是公开 GSM8K；采样生成的候选**只用于推理时聚合**、由公开答案验证，
  不作为训练数据、不自造 ground truth。确认后需把 AGENTS.md 中
  「合成数据只用于 RFT/DPO」的措辞改为「合成候选只用于推理时聚合并由公开答案验证」。
- dev 用于调参与选策略；test 只在最终统一评测时对每个方法各评一次。
- 记录 seed、temperature、top_p、k、模型、硬件、耗时与失败案例。
- 长任务仍用 tmux + 日志；产物落在 `/data` 并回传本地。
