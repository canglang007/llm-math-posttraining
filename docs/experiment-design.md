# 实验设计

更新时间：2026-11-15 07:46（Asia/Shanghai，UTC+08:00）

## 数据

| 数据集 | 许可证 | 切分 | 判分 |
|---|---|---|---|
| GSM8K | MIT | train 7.5k / test 1.3k | `####` 数值；`exact_match`（flexible-extract） |
| MATH | MIT | train 7.5k / test 5k；常用 MATH500 | `\boxed{}`；sympy 等价 |

- 主训练、验证、测试均来自公开数据；合成数据只用于 RFT/DPO，由公开答案验证。
- 从 train 中切出小规模 dev，用于 baseline probe 和选 checkpoint；test 只用于最终统一评测，每个方法各评一次，不参与任何选择。
- 污染风险：GSM8K/MATH 可能被预训练见过；报告必须说明，可加 GSM-Symbolic/扰动测试。

## 实验矩阵

| ID | 方法 | 数据 | 目的 |
|---|---|---|---|
| E0 | Base | 无 | 基线 |
| E1 | Public-SFT | 官方 train CoT | 公开数据 SFT |
| E2 | RFT | E1 采样，答案正确才保留 | 拒绝采样增益 |
| E3 | RFT + DPO | 同题正确/错误解答配对 | 偏好优化增益 |
| E4（可选） | Teacher distill | 强模型生成，答案过滤 | 蒸馏对照 |
| A1 | 数据消融 | 去重/难度分层/配比 | 数据引擎贡献 |
| A2 | 过滤消融 | 全部采样 vs 只保留正确 | 验证执行反馈 |

## 数据引擎

- 清洗、去重、难度分层（步数、题面长度、模型 pass rate）。
- 配比：按难度和题型控制混合。
- Rejection Sampling：每题采样 K 条，答案正确才保留，优先步骤完整、简洁的轨迹。
- 偏好对：同题正确/错误配对；chosen/rejected 必须由公开答案验证。
- 记录生成模型、采样参数、过滤规则和数量。

## 评测

- GSM8K test pass@1；MATH500/MATH test pass@1；
- 答案抽取失败率、格式错误率；
- 失败类型：算术、关系理解、多步中断、单位/格式；
- 数据效率：达到同一 pass@1 所需样本量；
- 成本：显存、耗时、tokens。

固定 prompt、few-shot 和解码参数；主结果 greedy；test 不参与任何选择。

## 训练

- 起点：`Qwen/Qwen3-4B-Base`；
- QLoRA：4-bit NF4，r=16，alpha=32，seq 2048–4096；12GB 用 seq 2048/r=8，24GB 用 seq 4096/r=16；
- Qwen chat template，只对 assistant 片段算 loss；
- dev 选 checkpoint；RFT 最多 1–2 轮，避免自我强化错误。

## 风险

- 4B-Base 的 GSM8K baseline 未知，必须先测 headroom；
- 答案抽取/格式错误会污染指标；
- RFT 冷启动：先 SFT 再 RFT；
- 数据污染：报告限制，加扰动测试；
- 算力：12GB 需要降 seq/r。

## 阶段

1. 本地 foundation：数据格式、答案抽取、prompt、评测、切分、单测；
2. 数据下载与审计；
3. Base baseline pilot；
4. SFT；
5. RFT/DPO；
6. 失败分析、demo、报告。
