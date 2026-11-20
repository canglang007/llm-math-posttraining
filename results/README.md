# results/ —— 主表的机器可读版本

由 `scripts/summarize_results.py` 从原始评测 jsonl 重算生成，**不要手改**。
数字变了就重跑脚本，让代码成为唯一出处。

```bash
python scripts/summarize_results.py --output-dir results
python scripts/audit_data_integrity.py --json results/data-audit.json
```

| 文件 | 内容 |
|---|---|
| `summary.json` | 每个 run 的题数/正确数/准确率/抽取失败/平均字符数 + 数据集与解码口径 |
| `summary.csv` | 同一份内容的表格版，方便贴进表格工具 |
| `pairwise.csv` | 每个 run 与参照 run（默认 SFT）的配对四格 + 净差 + McNemar 精确 p 值 |
| `data-audit.json` | 数据纪律审计：dev 泄漏、id 唯一性、训练数据来源、评测集合一致性 |

口径：GSM8K **dev-200**（从 train 划出的 500 题中的前 200 题），greedy 解码，batch_size=1，
HF 4bit（生成于 2026-11-17）。判分统一走 `math_posttrain.answers.is_correct`。

> 本目录**不含 test 结果**。test 1319 题的权威数字见 `docs/main-results.md` 第一节，
> 逐题预测在 `experiments/eval/vllm-test/`；两套口径不可混用。

完整表格、结论与解读见 `docs/main-results.md`；
资产位置、复现命令与已知缺口见 `docs/experiment-artifacts.md`。
