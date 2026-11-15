# 问题记录

更新时间：2026-11-15 07:46（Asia/Shanghai，UTC+08:00）

用途：面试时按“现象 → 根因 → 修复 → 验证 → 讲法”讲清楚一次真实决策。只追加，不删除历史。

## 2026-11-15 01:15 — 旧项目在 τ²-bench 上反复耗，偏离后训练主线

- **现象**：第二个项目本来用于补足大模型后训练覆盖面，但实际时间花在 τ²-bench 的 user simulator、数据、依赖和 benchmark 适配上，和已有 Agent 项目重复。
- **根因**：选题被 benchmark 和工具环境牵着走，没有从“后训练要展示什么”倒推。
- **修复**：新建独立项目 `llm-math-posttraining`，主问题是数学推理后训练；不再包含 agent/tool/user simulator/benchmark 复现。
- **验证**：新项目骨架、数据格式、答案抽取、评测和切分单测已建立。
- **讲法**：选题先对齐岗位覆盖面，再选数据；不要让 benchmark 的工程细节吞掉项目主线。

## 2026-11-15 01:15 — 自造数据当主证据，可信度低

- **现象**：为了造领域数据，容易把自建任务/自造标签当作主要训练和评测依据，面试时无法证明数据可信。
- **根因**：把“数据引擎”误解成“自己造数据”，没有区分主证据和增广。
- **修复**：主训练/评测用公开 GSM8K/MATH；合成数据只用于 RFT/DPO，并由公开答案验证；不自造 ground truth。
- **验证**：设计文档和 data-provenance 已明确边界。
- **讲法**：数据工程的价值在筛选、配比和验证，不是编标签。

## 2026-11-15 01:15 — 基础未搭建就上服务器

- **现象**：数据格式、答案抽取、评测和切分还没落地，就开始考虑服务器和 baseline。
- **根因**：把“跑模型”当成进度，跳过了本地 foundation。
- **修复**：先建项目骨架、单测和小样本评测；本地验证通过后再上 GPU。
- **验证**：18 个单测通过；不需要模型即可验证答案抽取、prompt、评测和切分。
- **讲法**：实验正确性先于算力；能在本地用小样本证伪的东西，不要带到服务器上。

## 2026-11-15 01:18 — GitHub raw 被重置，数据准备改走 HF 镜像

- **现象**：GSM8K 的 `raw.githubusercontent.com` 连接被 reset；HF fallback 又报 `httpx.InvalidURL: Invalid port ':1]'`。
- **根因**：本机代理/NO_PROXY 含 `[::1]`，httpx 解析失败；同时 MATH 的字段和预想不同，没有 `answer` 字段。
- **修复**：脚本默认 `HF_ENDPOINT=https://hf-mirror.com`，并把 NO_PROXY 收紧为 `localhost,127.0.0.1`；GSM8K 先 GitHub、失败自动走 HF；MATH 从 `solution` 的 `\boxed{}` 抽取答案。
- **验证**：GSM8K 7473/1319；MATH 7498/5000（train 2 条无 boxed 答案已跳过）；18 个单测通过。
- **讲法**：数据准备必须有镜像/fallback 和字段自检；网络失败不是数据问题，不能把下载失败当成模型或数据结论。

## 2026-11-15 01:15 — Qwen3-4B-Base 的 headroom 未知

- **现象**：没有可靠的 4B-Base GSM8K 分数，不能假设 SFT/RFT 一定有提升空间。
- **根因**：先入为主地认为“小模型一定弱”，没有先测 baseline。
- **修复**：服务器就绪后先跑 GSM8K test 200 题 baseline，确认 pass@1 和答案抽取失败率，再决定用 GSM8K 还是 MATH/更难子集。
- **验证**：待服务器恢复后执行。
- **讲法**：先测基线，再设计训练；不要用想象中的弱模型做实验。

## 2026-11-15 07:46 — baseline 探针原计划用 test 200 题，构成评测泄漏

- **现象**：handoff/README 的 baseline pilot 写的是「GSM8K test 200 题」，且明确要用这个结果决定用 GSM8K 还是 MATH、并据此调 prompt，与「test 只用于最终评测、不参与选模或调参」的项目红线冲突。
- **根因**：把「小样本探针」和「最终评测」混为同一批数据；赶进度时容易默认拿官方 test 当方便样本。
- **修复**：baseline probe 改用从 train 切出的 dev（`data/processed/sft/gsm8k-dev.raw.jsonl`，200 题），test 只在最终统一评测时对每个方法各评一次（含 Base 无 adapter）；同步修正 handoff、README、章程、实验设计和本记录。
- **验证**：待服务器环境就绪后执行 dev probe；最终评测前检查命令里的 `--data-file` 不含 `test.jsonl`。
- **讲法**：评测隔离要落在命令和文档上，而不是只写在原则里；面试时这比多跑一个方法更能说明实验可信度。

## 2026-11-15 07:46 — 服务器已有 Qwen3-4B 不是 Base 权重

- **现象**：`/data/models/Qwen3-4B` 的 model card 写明 `base_model: Qwen/Qwen3-4B-Base`、Training Stage 为 Pretraining & Post-training，tokenizer 带 chat_template、generation_config 带采样参数，是后训练过的通用模型。
- **根因**：目录名简写容易让「有 Qwen 模型」被误当成「有 Base 权重」；两者结构相同但训练阶段不同。
- **修复**：按原设计另下载 `Qwen/Qwen3-4B-Base` 到 `/data/models/Qwen3-4B-Base`；保留现有通用模型可作对照，但不作为 E0 基线。
- **验证**：检查 README/config/tokenizer_config/generation_config 后确认；下载完成后核对权重分片与 config。
- **讲法**：先核对模型卡再定基线；用已后训练的模型做 SFT 主线，会把「数据带来的提升」讲成「模型自带的能力」。

## 2026-11-15 07:49 — numpy 1.26 与 datasets 4.x 的 pyarrow>=21 双向冲突

- **现象**：pip 装完依赖后 `import transformers` 失败，链路是 transformers → sklearn → pyarrow → `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`；但 `pytest -q` 仍是 28 passed，单测没有覆盖训练依赖导入。
- **根因**：conda 环境 `torch` 预置 numpy 1.26.4，而 datasets 4.8.5 要求 pyarrow>=21，pyarrow 21+ 又要求 numpy>=2；只降级 pyarrow 会让 datasets 因缺少 `pa.json_()` 崩溃，只升级 numpy 也仍需把 pyarrow 还原，两头必须一起改。
- **修复**：统一升级到 `numpy>=2,<3` + `pyarrow>=21,<27`，保留 torch 2.10.0+cu128 不动；把 `numpy>=2,<3` 写进 pyproject 的 train extras 防止复现时再踩坑，并把「训练依赖全量 import」加入服务器环境自检。
- **验证**：torch/transformers/peft/trl/datasets/accelerate/bitsandbytes/sklearn/pyarrow 全部可导入，torch↔numpy 互操作正常，CUDA 可用，pytest 28 passed。
- **讲法**：环境自检不能只跑单测；依赖冲突要看 resolver 报出的两端约束，别只改报错最浅的那一层。

## 2026-11-15 08:01 — Base 用 ChatML 生成不停止，答案被续写污染

- **现象**：dev 5 题冒烟里 2 题输出跑到 1753 / 2175 字符，末尾出现 `assistant\n` 和复述的 `#### <final numeric answer>`，抽取失败；日志反复出现 `Setting pad_token_id to eos_token_id:151643`。
- **根因**：`Qwen3-4B-Base` 的 generation_config 只把 `<|endoftext|>`（151643）当 eos，而 chat template 结束符是 `<|im_end|>`（151645）；模型写完答案后不停，继续续写下一轮对话。加上 `extract_gsm8k_answer` 取**最后一个** `####`，跑飞内容会污染判分并白烧算力。
- **修复**：`prompts.py` 加 `resolve_eos_token_ids`（合并模型 eos 与 ChatML eos）和 `generation_stop_strings`（`<|im_start|>`、`assistant\n`、`\n\nQuestion:`）；`evaluate_model.py` 生成时显式传合并后的 eos、pad_token_id 与 stop_strings；新增 3 个单测。
- **验证**：本地与服务器均 31 passed；停掉旧代码跑的 200 题，用新代码重跑。修复前后同一 5 题：正确样本输出长度完全一致（558/495/1150），跑飞样本从 1753/2175 降到 569/765，总字符 6131 → 3537（-42%），pass@1 仍为 0.6（3/5）——修复只截断续写，不改变模型本身答不出的题。
- **讲法**：评测口径的坑常在生成配置里，不在判分函数里；先做小样本冒烟并对比「输出长度 + 正确样本是否不变」，才能证明修复没有偷偷改变结论。

## 2026-11-15 08:25 — 把 GPU 利用率低误判为性能 bug，两次优化都被实测否掉

- **现象**：200 题 dev probe 运行时 CPU ~99%、GPU 利用率仅 15%，怀疑是 `stop_strings` 每步 decode 整段输出拖慢了 GPU，于是停掉进程做优化。
- **根因（第一次反转）**：去掉 `stop_strings`、改用 token 级 eos + 生成后截断后，同样 5 题从 106s 变成 **145s**，GPU 平均 17%。原因是 Base 模型并不生成 `<|im_start|>` 特殊 token，token 级 eos 拦不住续写，跑飞的题又写满 512 tokens；字符串停止词反而更省算力。15% 是 batch=1 的 4bit 自回归 decode 固有形态，不是 bug。
- **第二次尝试**：改成批量解码（`--batch-size` 可配，左填充 + 一次 generate 多条）。batch=4 下 107s → **73s**、GPU 13.8% → 24%，但 5 题 predictions **逐字不一致**，accuracy 0.6 → 0.4；左填充与 4bit 批 GEMM 的数值路径改变会让 greedy 分叉。
- **修复**：正式评测固定 `--batch-size 1`，保留参数但标注「>1 会改变结果、禁用」；恢复 `generation_stop_strings` + token eos + `truncate_generation` 三层兜底；单测 33 passed（本地与服务器）。
- **验证**：A/B 数据见上；`batch=1` 的 5 题输出长度 558/495/1150/559/755、accuracy 0.6，与修复前一致。
- **讲法**：性能优化必须先定义不变量（这里是不改变任何一条 prediction），只看利用率或耗时会把评测口径优化坏；面试时主动讲「我做了两次优化都被数据否掉」比只讲成功更有说服力。

## 2026-11-15 09:26 — E0 baseline 完成：headroom 有限，失败主因是格式

- **现象**：dev 200 题（batch=1 正式口径）pass@1 **0.77**（154/200），抽取成功但答案错 19，抽取失败 27（13.5%），空输出 0；失败样本平均 841 字符，正确样本 520。
- **根因**：27 例里 25 例是「有 `####` 但后面不是纯数字」——复述 prompt 的 `#### <final numeric answer>` 后跑飞、`#### $6`、`#### 3weeks`、退化成一长串 `0`；只有 2 例完全没有 `####`。本质是 Base 未学过输出格式，不是不会推理。
- **修复/决策**：不为刷分放宽判分正则（会破坏口径）；把「抽取失败率」列为 E1 的核心指标，SFT 收益拆成格式对齐与推理提升两部分；正式对比表仍用 test 各评一次，dev 只是探针。
- **验证**：predictions 200 行、空输出 0；证据 `experiments/baseline/gsm8k-dev-probe200-final.jsonl` 与 `/data/probe-v3.log`，分析见 `docs/interview-debug-journal.md`。
- **讲法**：先测 baseline 才知道提升空间在哪；0.77 的 Base 让项目从「刷准确率」转向「讲清楚数据引擎到底贡献了什么」。

## 2026-11-15 13:43 — 服务器重启后 /data 完好，系统盘两项需重配

- **现象**：实例重启后 host/port/密码全变；`/data`（60G，余 31G）挂载正常，项目代码/git（`89a8548`）、data 32M、6 个 predictions、Base 模型 7.6G、conda env 8.5G、全部日志脚本都在；但 `/root/.ssh/authorized_keys` 不存在、`tmux` 命令缺失。
- **根因**：`/data` 是独立持久卷；`/`、`/root`、`/usr` 是随实例重建的容器系统盘，免密公钥和 apt 包都落在那里。
- **修复**：重新写入本机 `~/.ssh/id_ed25519.pub` 到 authorized_keys（700/600），`apt-get update -qq && apt-get install -y -qq tmux`；把恢复清单写进 handoff 第 9 节。
- **验证**：免密登录 OK、tmux 3.4、`pytest -q` 33 passed、训练依赖全量 import + CUDA 可用、`train_sft.py --config configs/sft_qwen3_4b_12gb.yaml --dry-run` 通过（6973/500 行、seq 2048/r8）、git 工作区干净、GPU 空闲。
- **讲法**：交付前要分清「持久盘」和「实例盘」；环境配置必须有一页可执行的恢复清单，不能依赖记忆。

## 2026-11-15 14:42 — E1 pilot 跑完 200 步却在 dev eval 时 OOM，权重全丢

- **现象**：pilot 200/200 步完成（46 分钟），但 `trainer.train()` 末尾 `_maybe_log_save_evaluate → trainer.evaluate → prediction_step → cross_entropy` 申请 2.14 GiB 时 CUDA OOM（当时仅剩 1.98 GiB、碎片 2.82 GiB），进程退出并标记 `SFT_PILOT_FAILED`；`experiments/sft/qwen3-4b-gsm8k-pilot200/` 为空，没有 checkpoint 可抢救。
- **根因**：`build_trainer` 在有 dev_rows 时设 `eval_strategy="epoch"`，pilot 用 `max_steps=200` 提前结束仍触发一次 dev loss eval；eval 不走 gradient checkpointing，logits/loss 的显存峰值把 12GB 卡撑爆。训练本身没崩，是多余的评估崩了；且 `save_strategy: epoch` 使 200 步内没有任何中间 checkpoint，一次失败就全丢。
- **修复**：pilot 配置去掉 `dev_file`（选模统一用 `evaluate_model.py` 的 dev pass@1）；改 `save_strategy: steps` + `save_steps: 50`，并在 `build_trainer` 里把 `save_steps` 传给 `TrainingArguments`；启动脚本加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减碎片；新增单测校验 pilot 配置。
- **验证**：待 v2 重跑；每 50 步会落一个 checkpoint，即使收尾再出问题也有权重可用。
- **讲法**：长训练必须先有「可恢复点」再谈优化；eval 不是免费的，显存峰值要单独算。46 分钟白跑换来一条写进配置的工程规则。
