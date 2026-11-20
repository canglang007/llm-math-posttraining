# 部署：把评测能力变成一个可调用的服务

两种运行方式，**接口完全一致**（`/v1/models`、`/v1/completions`），所以 `demo/live.html`
不需要改任何代码——只换端点。服务器不可用时直接切本地版继续演示。

| | 服务器版（推荐） | 本地版（兜底） |
|---|---|---|
| 引擎 | vLLM 0.34 + bf16，continuous batching | transformers + MPS/CPU |
| 硬件 | 单张 16 GB NVIDIA 卡 | Apple Silicon，16 GB 统一内存起 |
| 模型 | base + 4 个 LoRA 常驻，切换无开销 | 同上，用 `set_adapter` 切换 |
| 速度 | **2.7~4.3 秒/题**（4060 Ti / 5060 Ti 实测） | **20~45 秒/题**（M5 16 GB 实测，3~5 tok/s） |
| 适用 | 演示首选、并发请求 | 没机器 / 没卡 / 不想开机时 |

两版都**按需启停**，不需要常驻。

---

## 一、服务器版：vLLM

### 1. 启动（约 1~2 分钟，含权重加载）

```bash
cd /data/llm-math-posttraining
bash deploy/serve.sh                      # 默认 8000 端口，挂 sft / e2b / a2 / dpo 四个 LoRA + base
PORT=9000 bash deploy/serve.sh            # 换端口
GPUS=0.80 bash deploy/serve.sh            # 显存紧张时降占比
```

启动后会打印 `Uvicorn running on http://0.0.0.0:8000`，即就绪。

### 2. 验证（一条命令，10 秒）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sft","prompt":"<|im_start|>system\nYou are a careful math reasoning assistant. Show your steps and give a final answer.<|im_end|>\n<|im_start|>user\nAnne has 15 packs of 60 sweets, keeps 2 packs and divides the rest among 10 friends. How many each?\n\nSolve the problem step by step. On the last line write: #### <final numeric answer>.<|im_end|>\n<|im_start|>assistant\n","max_tokens":256,"temperature":0,"stop":["<|im_start|>","assistant\n","\n\nQuestion:"],"stop_token_ids":[151645,151644],"skip_special_tokens":true}'
```

期望输出里出现 `#### 78`。换 `"model":"a2"` 能看到不验证版本的回答（同一题常给出不同数字）。

> **`stop_token_ids` 不能省**：`skip_special_tokens=true` 会把 `<|im_start|>` 从解码文本里剥掉，
> 只靠 `stop` 字符串在 `<|im_end|>` 处停不住，模型会续写下一轮对话、`####` 抓到错的答案。
> 实测漏掉它时 SFT 与离线一致率从 20/20 掉到 18/20。

### 3. 停止

`Ctrl+C`，或 `pkill -f "vllm serve"`。服务无状态，重启不影响任何数据。

> 注意：别在**同一条命令**里既写 `pkill -f "vllm serve"` 又通过 ssh 远程执行——
> `pkill -f` 会匹配到那条 ssh 命令自身（命令行里含同样的字符串），把自己的会话杀掉。
> 远程操作时先 `ps -ef | grep vllm` 拿 pid，再 `kill <pid>`。

---

## 二、本地版：MPS / CPU（无 GPU 机器）

### 1. 准备（一次性）

```bash
pip install -e '.[train]'          # torch / transformers / peft / accelerate
bash deploy/restore_adapters.sh    # 从本地备份包解出 4 个 adapter（约 320 MB，3 秒）
```

`adapter` 不入库（体积），默认从 `data/server-backup/project-backup-*.tar.gz` 恢复；
已存在的目录会跳过。只跑 base 也可以：`bash deploy/run_local.sh --adapters ""`。

底座 `Qwen/Qwen3-4B-Base`（约 8 GB）首次运行会从 HuggingFace 下载。
国内网络可试 `export HF_ENDPOINT=https://hf-mirror.com`；**若在代理/企业网关后面，镜像域名可能不通，
此时不要设该变量**，走默认端点（本机实测：默认端点可用、镜像被挡）。

### 2. 启动

```bash
bash deploy/run_local.sh                 # 默认端口 8000，自动选 mps/cpu 与 dtype
MODEL=/path/to/Qwen3-4B-Base bash deploy/run_local.sh
bash deploy/run_local.sh --dry-run       # 只检查路径与 adapter，不加载模型
```

### 3. 验证

与服务器版同一个 `curl`（端口一致时就一样），或直接看日志里的
`[ready] 可用模型：['base', 'a2', 'dpo', 'e2b', 'sft']`。

已用**真实 4B 模型**实测（M5 16 GB，MPS bf16，peft 0.22 + transformers 5.7）：五个模型全部能出答案，
同一道题的本地结果与离线 vLLM 结果 **4/5 一致**：

| 模型 | 本地（MPS，eager） | 离线（vLLM，CUDA） | 一致 | 本地耗时 |
|---|---|---|---|---|
| Base | 78 | 78 | ✓ | 44 s |
| SFT | 78 | 84 | ✗ | 24 s |
| E2b（验证） | 78 | 78 | ✓ | 22 s |
| A2（不验证） | 84 | 84 | ✓ | 35 s |
| DPO（26 对） | 84 | 84 | ✓ | 40 s |

> **MPS 上必须用 `eager` 注意力**：transformers 默认走 `sdpa`，实测会让 Qwen3 输出退化成
> 无限重复（同一段话反复、永远不写 `####`），看起来像模型坏了。`local_server.py` 已对 MPS
> 自动强制 `eager`；如需覆盖用 `--attn-implementation sdpa|eager|flash_attention_2`。
> 踩坑记录见 `docs/issue-log.md` 2026-11-20 条目。

### 4. 与服务器版的差异（必须知道）

| 项 | 差异 | 影响 |
|---|---|---|
| 精度 | MPS/CPU 走 bf16/fp16，与 CUDA bf16 **不是同一条数值路径** | 个别题会翻转，属预期（实测 4/5 一致） |
| 速度 | 每条 20~45 秒（3~5 tok/s） | 演示时先说明，别让人以为是卡了 |
| 并发 | 单设备串行（内部有锁） | 页面上的 5 个模型是排队返回，不是同时 |
| 内存 | 16 GB 机器上 bf16 权重占 ~8 GB | 关掉其他大应用；必要时 `--dtype float16` |

**边界**：本地版只用于演示与自查，**不作为实验数字来源**。文档里的分数一律来自
`experiments/eval/` 的离线全量评测（同进程、同 LoRA）。

---

## 三、在本地浏览器查看演示页（两版通用）

`demo/live.html` 在**本地浏览器**里运行，它访问的 `localhost:8000` 指本机。

- **本地版**：服务就在本机，直接打开页面、端点填 `http://localhost:8000` 即可。
- **服务器版**：服务在远端，需要一条 SSH 端口转发（保持窗口不关）：

```bash
ssh -N -L 8000:localhost:8000 -p <端口> root@<服务器地址>
```

然后双击打开 `demo/live.html`，端点填 `http://localhost:8000`，点「连接」→ 输入题目 → 模型并排返回。

实机运行截图（同一道题：SFT 给 78 正确，RFT/RFT-不验证/DPO 三者都算成 `900-2*60=840` 给 84）：

![实时多模型对比](assets/demo-live-screenshot.png)

> 两个容易踩的点：
> 1. **不要在 IDE 的内嵌预览面板里点「测试连接」**——宿主应用的沙箱会拦截对 localhost 的请求，
>    页面只会显示连接失败。用浏览器直接打开 `file://` 路径。
> 2. **`--allowed-origins '["*"]'` 是必需项**：vLLM 0.34 起跨域默认关闭，不加这个参数浏览器报
>    `Disallowed CORS origin`。`serve.sh` 已内置；本地版由 `local_server.py` 直接返回 CORS 头。

---

## 显存与并发（服务器版）

| 项 | 估算 |
|---|---|
| Qwen3-4B bf16 权重 | ~8 GB |
| 4 个 LoRA adapter | < 1 GB（每个 ~63 MB，r=8） |
| KV cache（2048 上下文） | 剩余显存，约 5~6 GB |
| 结论 | **16 GB 卡即可**（4060 Ti / 5060 Ti 实测）；并发 8~16 路请求无压力 |

## 口径一致性（两版共用）

服务与评测必须用**同一套 prompt 与解码参数**，否则线上答案和文档里的分数对不上：

- prompt：`<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}\n\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n`
  （与 `src/math_posttrain/prompts.py` 的 `build_chat_messages` + ChatML 模板等价）
- 解码：`temperature=0`（greedy）、`max_tokens=512`、
  `stop=["<|im_start|>", "assistant\n", "\n\nQuestion:"]`（与评测 `generation_stop_strings()` 相同）
- **`stop_token_ids=[151645, 151644]`（`<|im_end|>`, `<|im_start|>`）必须一起给。**
  踩过的坑：`skip_special_tokens=true` 会把 `<|im_start|>` 从解码文本里剥掉，
  只靠 `stop` 字符串在 `<|im_end|>` 处停不下来，模型会续写下一轮对话，
  答案抽取抓到另一个 `####`（实测 SFT 从 20/20 掉到 18/20）。

本地版把这套口径直接复用 `math_posttrain.prompts`：`do_sample=False`、
`eos_token_id=resolve_eos_token_ids(...)`、`stop_strings=generation_stop_strings()`，
生成后再过同一个 `truncate_generation`——**没有第二份实现**。

`demo/live.html` 已内置这套口径；`tests/test_live_demo_prompt_parity.py` 会在每次跑单测时检查前端硬编码的
system prompt、指令串与 `stop_token_ids` 是否与 Python 侧一致，防止两边悄悄漂移。
`tests/test_local_server.py` 则用桩引擎跑真实 HTTP 往返，守住路由、CORS、greedy 守卫与响应格式。

### 线上与离线的实测一致率（2026-11-19 22:55，5060 Ti 16GB）

同一批 20 道 test 题，用与离线完全相同的采样参数分别请求线上服务、比对离线预测文件：

| 模型 | 逐题一致 | 说明 |
|---|---|---|
| SFT | **20/20** | 完全一致 |
| RFT（不验证，a2） | **17/20** | 3 题确定性地走了不同推理路径（如 `2971 → 221`），三次重测结果相同 |

差异来源：**多 LoRA 在线内核（`--max-loras 5`）与离线单 LoRA 进程的矩阵乘路径不同**，
bf16 下同精度也会在近阈值 token 上翻转，长思维链会把一次翻转放大成完全不同的解。
这不是 bug，但意味着：**服务响应与离线评测不能逐题对齐**，核对文档分数请用
`experiments/eval/` 下的预测文件（同一进程、同一 LoRA 跑出来的离线全量评测）。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 显存占比过高或 max_model_len 过大 | `GPUS=0.75 MAX_LEN=1536 bash deploy/serve.sh` |
| 启动报 `max_lora_rank` 不足 | adapter 的 r 大于设置值 | 本项目全为 r=8，若换了新 adapter 用 `LORA_RANK=16` |
| `404 model not found` | 模型名写错 | 可用名：`base` / `sft` / `e2b` / `a2` / `dpo` |
| 回答很啰嗦或跑飞 | 没带 stop 串 | 检查请求体里有 `stop` 字段 |
| 前端请求被浏览器拦截 | 跨域 | `serve.sh` 已带 `--allowed-origins '["*"]'`；本地版自带 CORS 头 |
| 本地版报 `MPS backend out of memory` | 统一内存不够 | 关掉其他大应用；`--dtype float16`；或只跑 1~2 个 adapter |
| 本地版很慢（>30 秒/题） | 退回 CPU 或内存吃紧 | 确认 `torch.backends.mps.is_available()` 为 True |

## 端到端自检（约 2 分钟）

1. 起服务（服务器版或本地版），用上面的 `curl` 验证一条请求（确认底座与 LoRA 都加载成功）
2. 打开 `demo/live.html`，同一道题并发问 `sft / e2b / a2 / dpo`
   （`a2` 与 `e2b` 的差异来自是否过滤掉 12.9% 的错误解）
3. 停服务（无状态，随时可重起）
