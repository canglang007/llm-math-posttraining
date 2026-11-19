# 部署：把评测能力变成一个可调用的服务

**按需启动，用完即停**——不需要常开。面试前一天起一次、跑通一条请求、录一段演示，面试当天不一定要现场验证。

## 1. 启动（约 1~2 分钟，含权重加载）

```bash
cd /data/llm-math-posttraining
bash deploy/serve.sh                      # 默认 8000 端口，挂 sft / e2b / a2 / dpo 四个 LoRA + base
PORT=9000 bash deploy/serve.sh            # 换端口
GPUS=0.80 bash deploy/serve.sh            # 显存紧张时降占比
```

启动后会打印 `Uvicorn running on http://0.0.0.0:8000`，即就绪。

## 2. 验证（一条命令，10 秒）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sft","prompt":"<|im_start|>system\nYou are a careful math reasoning assistant. Show your steps and give a final answer.<|im_end|>\n<|im_start|>user\nAnne has 15 packs of 60 sweets, keeps 2 packs and divides the rest among 10 friends. How many each?\n\nSolve the problem step by step. On the last line write: #### <final numeric answer>.<|im_end|>\n<|im_start|>assistant\n","max_tokens":256,"temperature":0,"stop":["<|im_start|>","assistant\n","\n\nQuestion:"],"stop_token_ids":[151645,151644],"skip_special_tokens":true}'
```

期望输出里出现 `#### 78`。换 `"model":"a2"` 就能看到不验证版本的回答（同一题常给出不同数字）。

> **`stop_token_ids` 不能省**：`skip_special_tokens=true` 会把 `<|im_start|>` 从解码文本里剥掉，
> 只靠 `stop` 字符串在 `<|im_end|>` 处停不住，模型会续写下一轮对话、`####` 抓到错的答案。
> 实测漏掉它时 SFT 与离线一致率从 20/20 掉到 18/20。

## 2.5 在自己电脑上看演示页（需要 SSH 端口转发）

`demo/live.html` 在**你本地浏览器**里跑，它访问的 `localhost:8000` 是你自己的电脑，
所以要在本地开一条隧道指向服务器：

```bash
ssh -N -L 8000:localhost:8000 -p <端口> root@<服务器地址>
```

然后双击打开 `demo/live.html`，端点填 `http://localhost:8000`，点「连接」→ 输入题目 → 五个模型并排返回。

实机运行截图（同一道题：SFT 给 78 正确，RFT/RFT-不验证/DPO 三者都算成 `900-2*60=840` 给 84）：

![实时多模型对比](assets/demo-live-screenshot.png)

> 注意：**不要在 IDE 的内嵌预览面板里点「测试连接」**——宿主应用的沙箱会拦截对 localhost 的请求，
> 页面会显示连接失败。用浏览器直接打开 `file://` 路径即可（已用 `--allowed-origins '["*"]'` 放开跨域）。

（服务已开 `--allowed-origins '["*"]'`；vLLM 0.34 起跨域默认关闭，不加这个参数浏览器会报
`Disallowed CORS origin`——这是必须项，不是安全可选项。）

## 3. 停止

`Ctrl+C`，或 `pkill -f "vllm serve"`。服务无状态，重启不影响任何数据。

> 注意：别在**同一条命令**里既写 `pkill -f "vllm serve"` 又通过 ssh 远程执行——
> `pkill -f` 会匹配到那条 ssh 命令自身（命令行里含同样的字符串），把自己的会话杀掉。
> 远程操作时先 `ps -ef | grep vllm` 拿 pid，再 `kill <pid>`。

## 显存与并发

| 项 | 估算 |
|---|---|
| Qwen3-4B bf16 权重 | ~8 GB |
| 4 个 LoRA adapter | < 1 GB（每个 ~63 MB，r=8） |
| KV cache（2048 上下文） | 剩余显存，约 5~6 GB |
| 结论 | **16 GB 卡即可**（4060 Ti / 5060 Ti 实测）；并发 8~16 路请求无压力 |

## 口径一致性（重要）

服务与评测必须用**同一套 prompt 与解码参数**，否则线上答案和文档里的分数对不上：

- prompt：`<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}\n\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n`
  （与 `src/math_posttrain/prompts.py` 的 `build_chat_messages` + ChatML 模板等价）
- 解码：`temperature=0`（greedy）、`max_tokens=512`、
  `stop=["<|im_start|>", "assistant\n", "\n\nQuestion:"]`（与评测 `generation_stop_strings()` 相同）
- **`stop_token_ids=[151645, 151644]`（`<|im_end|>`, `<|im_start|>`）必须一起给。**
  踩过的坑：`skip_special_tokens=true` 会把 `<|im_start|>` 从解码文本里剥掉，
  只靠 `stop` 字符串在 `<|im_end|>` 处停不下来，模型会续写下一轮对话，
  答案抽取抓到另一个 `####`（实测 SFT 从 20/20 掉到 18/20）。

`demo/live.html` 已内置这套口径；`tests/test_live_demo_prompt_parity.py` 会在每次跑单测时检查前端硬编码的
system prompt、指令串与 `stop_token_ids` 是否与 Python 侧一致，防止两边悄悄漂移。

### 线上与离线的实测一致率（2026-11-20，5060 Ti 16GB）

同一批 20 道 test 题，用与离线完全相同的采样参数分别请求线上服务、比对离线预测文件：

| 模型 | 逐题一致 | 说明 |
|---|---|---|
| SFT | **20/20** | 完全一致 |
| RFT（不验证，a2） | **17/20** | 3 题确定性地走了不同推理路径（如 `2971 → 221`），三次重测结果相同 |

差异来源：**多 LoRA 在线内核（`--max-loras 5`）与离线单 LoRA 进程的矩阵乘路径不同**，
bf16 下同精度也会在近阈值 token 上翻转，长思维链会把一次翻转放大成完全不同的解。
这不是 bug，但意味着：**演示以线上实际输出为准，不要用演示页面去验证文档里的分数**；
文档分数来自离线全量评测（同一进程、同一 LoRA）。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 显存占比过高或 max_model_len 过大 | `GPUS=0.75 MAX_LEN=1536 bash deploy/serve.sh` |
| 启动报 `max_lora_rank` 不足 | adapter 的 r 大于设置值 | 本项目全为 r=8，若换了新 adapter 用 `LORA_RANK=16` |
| `404 model not found` | 模型名写错 | 可用名：`base` / `sft` / `e2b` / `a2` / `dpo` |
| 回答很啰嗦或跑飞 | 没带 stop 串 | 检查请求体里有 `stop` 字段 |
| 前端请求被浏览器拦截 | 跨域 | 启动时用 `--allowed-origins "*"`（演示用），或用同源页面 |

## 面试演示建议（2 分钟脚本）

1. 起服务 → `curl` 验证一条（30 秒，证明真的能跑）
2. 打开 `demo/live.html`，同一道题并发问 `sft / e2b / a2 / dpo`（40 秒，重点看 `a2` 与 `e2b` 的差异——
   这是"答案验证"那 1.74pt 的直观来源）
3. 关掉服务（10 秒，说明"按需启停、不常挂"）
