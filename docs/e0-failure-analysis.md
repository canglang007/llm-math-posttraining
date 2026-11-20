# E0 失败分析：格式与长度主导的可恢复空间

更新时间：2026-11-15 18:20（Asia/Shanghai，UTC+08:00）

口径：E0 = Qwen3-4B-Base + 4bit NF4，dev 前 200 题，greedy，batch=1，max_new_tokens=512。

复现（原始预测文件未入库，先重新生成再分析）：

```bash
# 1) 用当时的 HF 4bit 口径重跑 dev-200
python scripts/evaluate_model.py --model-path <Qwen/Qwen3-4B-Base> \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl --dataset gsm8k --split dev --limit 200 \
  --output experiments/eval/e0-dev200.jsonl

# 2) 生成失败模式报告
python scripts/analyze_e0_failures.py \
  --predictions experiments/eval/e0-dev200.jsonl \
  --output experiments/analysis/e0-failure-report.json
```

## 总览

| 指标 | 数值 |
|---|---|
| 总题数 | 200 |
| 判对 | 154 |
| 抽取成功但答案错 | 19 |
| 答案抽取失败 | 27（13.5%） |
| pass@1 | **0.77** |

## 27 个抽取失败的模式

| 模式 | 数量 | 说明 |
|---|---|---|
| 复述 prompt 指令后跑飞 | 16 | 输出里重复 `#### <final numeric answer>` 后开始编造下一道题 |
| 答案带单位或货币符号 | 9 | `#### $6`、`#### 3weeks` |
| 完全没有 `####` 标记 | 2 | 生成中断，没写到答案行 |

**25/27 与输出格式相关，不是推理能力问题。**

## 判分口径敏感度（仅作上界，不改正式指标）

允许 `#### $6` / `#### 3weeks` 这类「数字 + 单位」被抽取时，可救回 **7 题**，
pass@1 上界 0.77 → **0.805**。正式指标仍使用严格抽取的 0.77；
这 3.5 个点说明「格式规范化 / 过滤」本身就是一个可量化的增益来源，
但不能靠放宽判分来刷分。

## 输出长度与正确率强相关

| 输出长度四分位 | 最大字符数 | 准确率 |
|---|---|---|
| Q1 | 367 | **0.94** |
| Q2 | 548 | 0.82 |
| Q3 | 754 | 0.74 |
| Q4 | 2186 | **0.58** |

输出越长越容易跑飞，Q4 比 Q1 低 36 个点。这为「早停 / 格式过滤 / 候选重排」
提供了直接依据，也解释了为什么 SFT pilot 退化后每题跑满 512 tokens 时评测会慢一个数量级。

## 对推理时验证方向的意义

- 16 个「复述指令跑飞」和 9 个「带单位」的候选，正是多数投票与 verifier 过滤能处理的对象；
- `oracle pass@k` 与 `maj@k` 的上界可以用这 27 个案例单独量化；
- 采样时应记录每条候选的长度与是否成功抽取，作为 A1/A2 消融的分层变量。
