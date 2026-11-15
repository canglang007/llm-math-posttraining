# 主线：Base → SFT → RFT → DPO

更新时间：2026-11-15 21:11（Asia/Shanghai，UTC+08:00）
状态：**方案与代码已就绪，等服务器开机执行**

## 为什么是这个主线

之前的问题不是做得不对，是重心放错了：在「投票」这个不需要多少难度的方向反复做加法和补表，
覆盖到的岗位技能只有「评测」一项。后训练岗位真正考核的四件事——数据构造、SFT、拒绝采样、
偏好优化——一项都没有产出结果。

所以主线重排为一条**后训练链路**，投票降为数据构造工具：

```
E0 Base ──→ E1 SFT ──→ E2 RFT ──→ E3 DPO
 0.770      官方 CoT     模型自采样     正确/错误
            QLoRA        答案过滤       自动配对
```

四档共用同一套评测口径（dev 200、batch=1、同一 `is_correct`），最终画成一条 pass@1 曲线。
test 全程不碰，等四档都定稿后各评一次。

## 阶段表

| 阶段 | 数据来源 | 方法 | 成功判据 | GPU 成本 |
|---|---|---|---|---|
| E0 Base | — | 直接评测 | 已完成：pass@1 **0.770** | — |
| E1 SFT | GSM8K train 6973 条官方 CoT | QLoRA r8，400 步 | pass@1 明显 > 0.770 | ~95 min |
| E2 RFT | SFT 模型在 **train** 子集采样，只留答案正确的 | QLoRA 续训（起点是 SFT adapter） | pass@1 > E1 | 采样 ~2h + 训练 ~90 min |
| E3 DPO | 同题「正确解 vs 错误解」配对 | DPO beta=0.1（起点是 RFT adapter） | pass@1 ≥ E2 或 chosen 概率上升 | ~60 min |
| 最终评测 | dev 200 | batch=1 | 四条曲线同一张图 | 每档 ~15 min |

**数据纪律（红线）**：RFT / DPO 的训练数据只能来自 **train** 切分。
现有的 1600 条候选是 dev 的，**不能用于训练**，只能用来验证管线。
dev 产物会被脚本强制改名为 `dev-*` 前缀。

## 消融（差异化部分）

- **A2 验证反馈的贡献**：同一批 RFT 候选，一组「只保留正确」，一组「全部保留」。
  两组的差 = 答案过滤本身的贡献，这是全项目最能体现「数据引擎」价值的一行。
  这一行不需要额外采样，只是同一批候选的两种用法。
- **A1 数据配比/去重**：暂不做。预算不够支撑一组有统计意义的配比实验，硬做只会得到噪声。

## 开机后的执行顺序

```bash
# 0. 确认环境与代码
cd /data/llm-math-posttraining && git log --oneline -1

# 1. E1 SFT（400 步，约 95 分钟）
nohup python scripts/train_sft.py --config configs/sft_qwen3_4b_12gb_pilot_v2.yaml > /data/sft-v2.log 2>&1 &

# 2. 训练完立刻评测，确认 SFT 是否真的学会了（这是整条链的闸门）
python scripts/evaluate_model.py --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-pilot-v2 \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl --dataset gsm8k \
  --limit 200 --batch-size 1 --output experiments/eval/sft-v2-dev200.jsonl

# 3. 用 SFT 模型在 train 子集采样（1000 题 × 4 条，约 2 小时）
python scripts/run_sampling.py --model-path /data/models/Qwen3-4B-Base \
  --adapter experiments/sft/qwen3-4b-gsm8k-pilot-v2 \
  --data-file data/processed/sft/gsm8k-train.raw.jsonl --dataset gsm8k \
  --limit 1000 --k 4 --temperature 0.8 --top-p 0.95 --max-new-tokens 384 --seed 42 \
  --output experiments/sampling/gsm8k-train-sft-k4.jsonl

# 4. 展开 + 过滤 + 构造 RFT / DPO 数据（纯 CPU）
python scripts/expand_sampling_candidates.py --sampling-file experiments/sampling/gsm8k-train-sft-k4.jsonl \
  --dataset gsm8k --split train --output data/processed/rft/gsm8k-train-candidates.jsonl
python scripts/build_preference_data.py --candidates data/processed/rft/gsm8k-train-candidates.jsonl \
  --dataset gsm8k --rft-output data/processed/rft/gsm8k-train-rft.jsonl \
  --dpo-output data/processed/dpo/gsm8k-train-dpo.jsonl

# 5. E2 RFT（约 90 分钟）
python scripts/train_sft.py --config configs/rft_qwen3_4b_12gb.yaml

# 6. E3 DPO（先 dry-run 确认 TRL API 对得上，再跑）
python scripts/train_dpo.py --config configs/dpo_qwen3_4b_12gb.yaml --dry-run
python scripts/train_dpo.py --config configs/dpo_qwen3_4b_12gb.yaml
```

## 已知风险与预案

| 风险 | 判据 | 预案 |
|---|---|---|
| SFT 仍不收敛 | 400 步后 pass@1 ≤ 0.770，或输出仍不停止 | 看 loss 曲线：仍在降就加步数；震荡就降 lr 到 1e-4 |
| RFT 数据量不足 | 过滤后 < 1000 条 | 扩大采样题数到 2000，或 k 提到 6 |
| DPO 显存不足 | OOM | 降 max_length 到 768、max_prompt_length 到 384 |
| TRL 版本 API 不匹配 | `--dry-run` 后报参数错误 | 按服务器实际 trl 版本改 DPOTrainer 调用签名 |
| DPO 把模型训坏 | pass@1 掉超过 2 个点 | 降 beta 到 0.05 或减 epoch 到 0.5，DPO 行标注失败 |

## 每阶段对应的简历技能词

| 阶段 | 技能词 |
|---|---|
| E0 评测体系 | 评测设计、答案抽取、可复现口径 |
| E1 SFT | QLoRA / LoRA、4bit 量化、gradient checkpointing、训练调试 |
| E2 RFT | 拒绝采样微调、采样策略、数据过滤、消融设计 |
| E3 DPO | 偏好数据构造、DPO、beta 调节 |
| 全链路 | 后训练 pipeline、pass@1 归因、成本-收益取舍 |

## 与旧方案的接续

`docs/pivot-plan.md` 里的推理时验证结果（maj@8 0.920）不删除，但**降级为附录**：
它的作用是（a）在 E1 失败时提供过一个可用基线；（b）验证了「答案过滤 + 聚合」这条路有效，
为 RFT 的数据过滤提供先验。它不再作为主线，也不再增加任何新的分析维度。
