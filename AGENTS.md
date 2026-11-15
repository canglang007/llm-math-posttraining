# AGENTS.md

## 项目定位

独立的大模型后训练项目：数学推理解题助手 + 数据引擎。与已有 Agent 项目分开；不实现 agent、tool calling、user simulator、benchmark 平台。

## 工作规则

- 先完成数据/评测审计和设计，再写训练代码。
- 主训练/评测数据必须来自公开数据集；合成数据只用于 RFT/DPO 或推理时聚合（采样候选），且一律由公开答案验证，不自造 ground truth。
- test 只用于最终评测；不参与训练、过滤、选模或调参。
- 先做本地 foundation 和单测；验证通过后才上 GPU 服务器。
- Mac 只做开发、数据检查和单测；不加载模型，不做推理/训练。
- 记录数据来源、许可证、切分、模型、seed、硬件和失败案例。
- 文档更新时间精确到分钟，注明 Asia/Shanghai（UTC+08:00）。
- Git commit 消息使用中文。

## 文档

- `docs/project-charter.md`：背景与目标
- `docs/experiment-design.md`：数据、方法、实验矩阵、指标
- `docs/data-provenance.md`：数据来源和有效性边界
- `docs/issue-log.md`：问题记录，只追加不删
