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
  -d '{"model":"sft","prompt":"<|im_start|>system\nYou are a careful math reasoning assistant. Show your steps and give a final answer.<|im_end|>\n<|im_start|>user\nAnne has 15 packs of 60 sweets, keeps 2 packs and divides the rest among 10 friends. How many each?\n\nSolve the problem step by step. On the last line write: #### <final numeric answer>.<|im_end|>\n<|im_start|>assistant\n","max_tokens":256,"temperature":0}'
```

期望输出里出现 `#### 78`。换 `"model":"a2"` 就能看到不验证版本的回答（同一题常给出不同数字）。

## 3. 停止

`Ctrl+C`，或 `pkill -f "vllm serve"`。服务无状态，重启不影响任何数据。

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

`demo/live.html` 已内置这套口径；`tests/test_live_demo_prompt_parity.py` 会在每次跑单测时检查前端硬编码的
system prompt 与指令串是否与 Python 侧一致，防止两边悄悄漂移。

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
