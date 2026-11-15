# 面试用：调试与决策实录

更新时间：2026-11-15 15:53（Asia/Shanghai，UTC+08:00）

用途：把「现象 → 假设 → 实测 → 结论 / 反转」整理成可直接讲的材料。
`docs/issue-log.md` 是只追加的流水账（完整证据），本文按面试叙事重组，两者对照看。

项目动机：为找实习面试准备的作品，教育场景是叙事载体。第一优先是主链路闭环
（E0 Base → E1 SFT → E2 RFT 的 pass@1 对比）和能单独量化数据引擎贡献的 A1/A2 消融，
不追求刷榜或堆方法。

## 一句话主线

> 我在一天里把「评测口径」和「生成配置」两个隐形变量找出来并固定住；
> 期间做了两次自以为对的优化，都被小样本 A/B 数据否掉，最后宁可慢也不改口径。

## 时间线（2026-11-15，Asia/Shanghai）

| 时间 | 事件 | 结论 |
|---|---|---|
| 07:35 | 明确项目为面试作品 | 优先级改为「可信证据链」> 堆方法 |
| 07:36 | 核对 git | handoff 里「未提交」信息过时，修正 |
| 07:39 | 服务器体检 | RTX 4070 12GB；`/data` 持久盘；预置 Qwen3-4B 不是 Base |
| 07:44 | 代码/数据同步、装 tmux、换清华 pip 源 | pypi.org 直连会卡在 build dependencies |
| 07:46 | 依赖冲突 | numpy 1.26 ↔ pyarrow>=21，必须一起升 |
| 07:53 | Base 模型下载完成 | 按原设计用 Qwen3-4B-Base |
| 07:56 | 首次冒烟 5 题 | 3/5 正确，2 题输出跑飞并续写下一轮对话 |
| 08:01 | 修 eos 后重跑 | 正确样本逐字不变，跑飞被截断，输出字符 −42% |
| 08:13 | 发现 GPU 利用率仅 15%，决定停进程排查 | 不等慢进程跑完 |
| 08:20 | 反转一：去掉 stop_strings | 反而 106s → 145s |
| 08:25 | 反转二：batch=4 | 快 32%，但 accuracy 0.6 → 0.4，结果逐字不同 |
| 08:32 | 固定 batch=1，重启 200 题 dev probe | 加每 10 题进度日志，提交 `45aaeb1` |
| 08:40 | probe 跑到 20/200 | 后台继续 |
| 09:26 | 200 题 dev baseline 完成 | pass@1 0.77，抽取失败 27，见下节 |
| 13:43 | 服务器重启，host/port 更换 | `/data` 全部完好；重配免密与 tmux，三项自检通过 |
| 13:48 | 启动全量 E1 SFT | 实测 13.8 s/步 × 872 步 ≈ 3h20m，改跑 200 步 pilot |
| 14:42 | pilot v1 跑满 200 步后 Trainer dev eval OOM | 权重全丢，定位到多余的 epoch 末 eval |
| 15:33 | pilot v2 修复后重跑完成 | adapter 66MB + 4 个 checkpoint，train_loss ≈ 0.490 |
| 15:34 | E1 dev 前 50 题评测启动 | 与 E0 同批题（E0：0.76 / 抽取失败 9） |
| 15:42 | 账户余额用完，实例被释放 | 平台在 **15:43 保存了镜像**（用户确认） |
| 15:47 | 新旧 host 端口均超时 | E1 评测未完成；产物在 `/data`，待镜像恢复后校验并回传本地 |

## 七个可讲的决策

### 1. baseline 不能用 test（评测泄漏）

- **现象**：原计划「GSM8K test 200 题跑 baseline，并据此决定用 GSM8K 还是 MATH、怎么调 prompt」。
- **冲突**：与项目红线「test 只用于最终评测，不参与选模或调参」直接矛盾。
- **修复**：baseline probe 改用从 train 切出的 dev；test 只在最终统一评测时对每个方法（含 Base 无 adapter）各评一次。
- **讲法**：评测隔离要落在命令和文档上，不能只写在原则里。

### 2. 「有 Qwen 模型」≠「有 Base 权重」

- **现象**：服务器 `/data/models/Qwen3-4B` 的 model card 写 `base_model: Qwen/Qwen3-4B-Base`、Training Stage 为 Pretraining & Post-training，tokenizer 带 chat_template、generation_config 带采样参数。
- **判断**：这是后训练过的通用模型；用它做 E0 会把「数据带来的提升」讲成「模型自带的能力」。
- **修复**：另下载 `Qwen3-4B-Base`；原模型保留作对照，不当基线。
- **讲法**：定基线前先核对模型卡，目录名不算证据。

### 3. 依赖冲突要改「两端」

- **现象**：装完依赖 `pytest -q` 28 passed，但 `import transformers` 崩：transformers → sklearn → pyarrow → `pyarrow requires NumPy 2.0 or newer, found 1.26.4`。
- **根因**：conda 环境预置 numpy 1.26.4；datasets 4.8.5 要求 pyarrow>=21，而 pyarrow 21+ 要求 numpy>=2。只降级 pyarrow 会让 datasets 缺 `pa.json_()` 崩溃。
- **修复**：统一升到 numpy 2.5.4 + pyarrow 26.0.1，并把 `numpy>=2,<3` 写进 train extras；自检加「训练依赖全量 import」。
- **讲法**：单测通过 ≠ 环境可用；resolver 报的是两端约束，别只改报错最浅的一层。

### 4. 生成不停止（真正影响判分的 bug）

- **现象**：冒烟 5 题中 2 题输出 1753 / 2175 字符，末尾是 `assistant\n` 和复述的 `#### <final numeric answer>`，抽取失败。
- **根因**：Base 的 `generation_config` 只把 `<|endoftext|>`(151643) 当 eos，ChatML 结束符是 `<|im_end|>`(151645)；模型写完答案不停。而 `extract_gsm8k_answer` 取**最后一个** `####`，续写内容会污染判分。
- **修复**：`prompts.py` 合并 token 级 eos（`<|im_end|>`、`<|im_start|>`）+ `generation_stop_strings` + 生成后 `truncate_generation` 三层兜底；3 个新单测。
- **验证**：正确样本输出长度 558/495/1150 **逐字不变**，跑飞样本 1753/2175 → 569/765，总字符 6131 → 3537（−42%），accuracy 仍 0.6。
- **讲法**：证明修复没偷偷改结论，要靠「正确样本逐字一致 + 输出长度」两个不变量，不能只看最终分数。

### 5. 反转一：CPU 高 ≠ 停止词拖慢 GPU

- **假设**：GPU 15%、CPU 99%，怀疑 `StopStringCriteria` 每步 decode 整段输出把 GPU 饿住了。
- **实验**：去掉 `stop_strings`，只留 token 级 eos + 生成后截断。
- **数据**：同样 5 题 **106s → 145s**，GPU 平均 17%，accuracy 仍 0.6。
- **结论**：Base 模型并不生成 `<|im_start|>` 特殊 token，token 级 eos 拦不住续写，跑飞的题又写满 512 tokens；字符串停止词反而更省算力。15% 是 batch=1 的 4bit 自回归 decode 固有形态，不是故障。
- **讲法**：性能优化前先量「总生成 token 数」，别只盯利用率。

### 6. 反转二：批量解码会改变 greedy 结果

- **实验**：`--batch-size 4`（左填充 + 一次 generate 多条）。
- **数据**：107s → **73s**，GPU 13.8% → 24%，但 5 题 predictions **逐字不一致**，accuracy **0.6 → 0.4**。
- **原因**：左填充与 4bit 批 GEMM 的数值路径变化会让 greedy 解码分叉。
- **结论**：正式评测固定 `--batch-size 1`；参数保留但标注「>1 会改变结果、禁用」。若真要提速，必须把 batch=4 定成所有实验统一的新口径并重跑全部基线，收益却只有三成。
- **讲法**：能复现的方法间对比，比一张好看的利用率图重要。

### 7. 长训练必须先有「可恢复点」

- **现象**：pilot 200 步跑满 46 分钟，收尾时 Trainer 的 epoch 末 dev eval 在 `cross_entropy` 申请 2.14 GiB → CUDA OOM；进程退出，而 `save_strategy: epoch` 让 200 步内没有任何中间 checkpoint，权重全丢。
- **根因**：eval 不走 gradient checkpointing，显存峰值被低估；把「保存」和「评估」都绑在 epoch 上，短跑（`max_steps=200`）等于既没存档又多算一次评估。
- **修复**：pilot 去掉 `dev_file`（选模统一用 `evaluate_model.py` 的 dev pass@1）、改 `save_strategy: steps` + `save_steps: 50`、开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；重跑后每 50 步落盘，15:33 正常收尾。
- **讲法**：先保证「失败可恢复」再谈优化；46 分钟白跑换来的规则要写进配置和单测，不是记在脑子里。

## 关键数字对照（同一 dev 5 题，greedy，max_new_tokens=512）

| 配置 | 耗时 | GPU 平均利用率 | accuracy | 输出总字符 | 判定 |
|---|---|---|---|---|---|
| 原始（无修复） | 150s | ~15% | 0.6 | 6131 | 2 题跑飞 |
| eos + stop_strings（batch=1） | 106s | ~15% | 0.6 | 3537 | ✅ 采用 |
| 仅 token eos（去 stop_strings） | 145s | 17% | 0.6 | 3517 | ❌ 更慢 |
| 上述 + batch=4 | 73s | 24% | **0.4** | 结果逐字不同 | ❌ 口径漂移 |
| 最终 batch=1 复测 | 107s | 13.8% | 0.6 | 3517 | ✅ 正式口径 |

**不变量**：任何修复都不得改变正确样本的 prediction 文本（558 / 495 / 1150 三条始终逐字相同）。

## E0 baseline 正式结果（dev 200 题，batch=1 口径）

| 指标 | 数值 |
|---|---|
| pass@1 | **0.77**（154/200） |
| 抽取成功但答案错 | 19 |
| 答案抽取失败 | 27（13.5%） |
| 空输出 | 0 |
| 平均输出长度 | 583 字符（正确 520 / 抽取失败 841） |
| 正确答案中 `####` / `\boxed{}` | 152 / 2 |

失败类型（27 例）：

- 25 例是「有 `####` 但后面不是纯数字」：复述 prompt 的 `#### <final numeric answer>` 后跑飞、`#### $6` 带货币符号、`#### 3weeks` 带单位、退化成一长串 `0`；
- 2 例完全没有 `####`；
- 输出越长越容易跑飞（失败样本平均 841 字符 vs 正确样本 520）。

**三个可讲的判断**：

1. Base 在 GSM8K 上已经 0.77，headroom 有限——这正是「先测 baseline 再训练」的价值，不能默认小模型一定弱。
2. 27 个失败多数是**输出格式没对齐**，不是推理能力：SFT 用官方 CoT 教会 `#### <数字>` 的写法，很可能直接把这部分救回来。E1 的增益必须拆成「格式对齐」和「推理提升」两部分讲，否则会把格式收益吹成推理收益。
3. 因此差异化卖点更应放在 A1/A2 消融（单独量化数据引擎贡献）和失败分析，而不是绝对分数刷榜。

> 注意：dev 只是探针，正式对比表仍要在 test 上对每个方法各评一次；dev 数字不写进最终结论表。

## 方法论（可迁移）

1. 先定义不变量再优化：这里的硬约束是「predictions 逐字一致 + 同一评测口径」。
2. 小样本 A/B：5 题、同一 greedy 与解码参数，比跑完 200 题再发现便宜得多。
3. 一次只改一个变量：先 eos，再停止词，再 batch，才能定位是哪一步出的问题。
4. 允许推翻自己：两次优化都被数据否掉，如实记录比只讲成功更可信。
5. 长任务要可观测：tmux + 每 10 题进度日志 + 里程碑日志，断网也不丢。

## 1 分钟讲法（可直接背）

> 这个项目做 4B 模型的数学推理后训练，我想单独量化数据引擎的贡献。最花时间的不是训练，
> 是把评测口径固定住：我发现原计划用 test 跑 baseline 会泄漏，改成 dev；又发现 Base 模型的
> eos 配置和 ChatML 对不上，模型会一直写到 512 token 并续写下一轮对话，而答案抽取取最后一个
> `####`，等于判分被污染。修完之后我看到 GPU 只有 15%，试着去掉停止词、又试了批量解码，
> 结果一个更慢，一个虽然快三成却把 5 题准确率从 0.6 变成 0.4——batch=1 的 4bit 解码数值路径
> 变了。最后我固定 batch=1 宁可慢，因为面试里能被复现的对比表，比一张好看的利用率图重要。
> 整个过程的 A/B 数字和两次反转都记在 issue-log 和面试实录里。

## 可能被追问的 Q&A

- **Q：为什么不用 test 做 baseline？** A：baseline 结果会用来决定数据集和 prompt，属于调参；test 只能用于最终一次性评测，dev 从 train 切出才合法。
- **Q：GPU 15% 是不是浪费算力？** A：是 batch=1 自回归 decode 的固有形态，不是 bug。实测批量化能到 24%，但会改变结果；在「结果可复现」和「利用率」之间我选前者。
- **Q：batch 为什么会改变 greedy？** A：左填充改变了每条序列的 padding 位置，4bit 量化下批 GEMM 与单条 GEMV 的浮点累加路径不同，logits 微小差异会让 greedy 逐步分叉。
- **Q：怎么保证 E0/E1/E2 可比？** A：同一份 dev、同一 prompt 构造、greedy、max_new_tokens=512、batch=1、同一个 `evaluate_model.py` 版本；任何改动都必须重跑全部方法，不混用口径。
- **Q：4bit 量化会不会影响结论？** A：会，所以它是所有方法共享的固定条件，写进配置；报告里会注明量化设置，不拿它和全精度结果直接比。
- **Q：如果重来怎么省时间？** A：上服务器第一件事就跑 5 题冒烟并固化 prediction 快照，之后任何优化先做 A/B；这样两次反转能在 5 分钟内被发现，而不是等到 200 题跑一半。
- **Q：Base 已经 0.77，SFT 还有意义吗？** A：有意义，但要先拆指标：13.5% 的题是格式没对上而不是不会做；E1 的主要收益可能来自格式对齐。我会分别报告「抽取失败率下降」和「抽取成功样本里的准确率变化」，避免把格式收益讲成推理收益。若 headroom 仍不足，就换更难子集，或把重心放在 A1/A2 消融上。
- **Q：实例被释放、镜像没保住训练产物怎么办？** A：本地有完整代码、配置、公开数据、单测和 E0 预测，`configs/sft_qwen3_4b_12gb_pilot.yaml` + seed 42 可在 46 分钟内重训复现。但可复现不等于不用备份——恢复后第一件事是把 adapter 回传本地，省下这 46 分钟。

## 复现命令

```bash
export PATH=/data/miniconda/envs/torch/bin:$PATH
cd /data/llm-math-posttraining

# 5 题冒烟（正式口径）
python scripts/evaluate_model.py \
  --model-path /data/models/Qwen3-4B-Base \
  --data-file data/processed/sft/gsm8k-dev.raw.jsonl \
  --dataset gsm8k --split dev --limit 5 --max-new-tokens 512 \
  --batch-size 1 --output experiments/baseline/smoke5.jsonl

# 200 题 dev probe（后台）
tmux attach -t probe3        # 或 tail -f /data/probe-v3.log

# 环境自检
python -m pytest -q
python -c "import torch, transformers, peft, trl, datasets, accelerate, bitsandbytes"
```

证据索引：`docs/issue-log.md`（只追加，含每条的现象/根因/修复/验证/讲法）、
`/data/bootstrap.log`、`/data/probe.log`、`/data/probe-pre-fix.log`、
`/data/probe-v2.log`、`/data/probe-v2-pre-speedfix.log`、`/data/probe-v3.log`、
`/data/smoke-bs1.log`、`/data/smoke-bs4.log`、`/data/gpu-bs1.log`、`/data/gpu-bs4.log`。
