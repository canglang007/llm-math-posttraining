# 数据来源与有效性边界

更新时间：2026-11-20 16:35（Asia/Shanghai，UTC+08:00）

## GSM8K

- 来源：OpenAI，论文 *Training Verifiers to Solve Math Word Problems*（arXiv:2110.14168）
- 仓库：https://github.com/openai/grade-school-math
- 数据卡：https://huggingface.co/datasets/openai/gsm8k
- 许可证：MIT
- 切分：train 7.5k / test 1.3k
- 答案格式：解答末尾 `#### <number>`
- 评测：抽取最终数值，`exact_match`（flexible-extract）；lm-eval 常用 8-shot 口径

## MATH

- 来源：Hendrycks et al.，*Measuring Mathematical Problem Solving With the MATH Dataset*（arXiv:2103.03874）
- 仓库：https://github.com/hendrycks/math
- 数据卡：https://huggingface.co/datasets/EleutherAI/hendrycks_math
- 许可证：MIT
- 切分：train 7.5k / test 5k；常用 MATH500 子集
- 答案格式：`\boxed{...}`
- 评测：抽取 boxed 答案，sympy 等价判断；官方 eval 脚本和 MATH500 口径
- **本项目最终未使用 MATH**：数据已下载备查，但实验主线只做 GSM8K（原因见 `docs/main-results.md` 的「局限与结论边界」）

## 实际下载记录

- GSM8K：train 7473 / test 1319；通过 `HF_ENDPOINT=https://hf-mirror.com` 的 `openai/gsm8k` 获取；provenance 见 `data/raw/provenance.json`。
- MATH：train 7498 / test 5000；7 个 config 合并；train 有 2 条无法从 solution 抽取 boxed 答案，已跳过并记录 `missing_answers`。
- 所有数据只放在 `data/raw/`（gitignored），不提交。

## 有效性边界

- 只使用公开数据；不把自造标签当作 ground truth。
- RFT/DPO 的合成轨迹必须由公开答案判对错；记录生成模型、采样参数和过滤规则。
- GSM8K/MATH 可能出现在预训练语料中；报告必须写污染风险，不能宣称“泛化到未见数学”。
- test 只用于最终评测；不参与训练、过滤、选模或调参。
- 评测脚本先用人工构造的小样本做自测，再跑全量。
