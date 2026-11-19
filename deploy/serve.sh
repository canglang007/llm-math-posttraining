#!/bin/bash
# 一键启动数学推理服务：Qwen3-4B-Base + 5 个后训练 LoRA（vLLM OpenAI 兼容接口）
#
# 用法：
#   bash deploy/serve.sh                 # 默认端口 8000，挂全部 5 个 adapter
#   PORT=9000 bash deploy/serve.sh       # 换端口
#   GPUS=0.80 bash deploy/serve.sh       # 降显存占用（OOM 时用）
#
# 停止：Ctrl+C（前台）或 kill 掉进程；服务无状态，重启即可。
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/models/Qwen3-4B-Base}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/llm-math-posttraining}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0.85}"          # gpu_memory_utilization
MAX_LEN="${MAX_LEN:-2048}"    # prompt≤1024 + 生成≤512，留余量
MAX_LORAS="${MAX_LORAS:-5}"
LORA_RANK="${LORA_RANK:-8}"   # 本项目全部 adapter 的 r=8

# 模型名 → adapter 目录（只读，启动时一次性加载；base 本身由 --served-model-name 提供）
declare -a MODULES=(
  "sft=$PROJECT_ROOT/experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400"
  "e2b=$PROJECT_ROOT/experiments/rft/qwen3-4b-gsm8k-rft-300q"
  "a2=$PROJECT_ROOT/experiments/rft/qwen3-4b-gsm8k-rft-a2nf-300q"
  "dpo=$PROJECT_ROOT/experiments/dpo/qwen3-4b-gsm8k-dpo-from-sft"
)

# 检查 adapter 是否齐全（缺哪个直接报出来，不要等到运行时 404）
missing=0
for entry in "${MODULES[@]}"; do
  path="${entry#*=}"
  if [ ! -f "$path/adapter_config.json" ]; then
    echo "[ERROR] 缺 adapter: $path" >&2
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "先把 adapter 同步到服务器（或改 MODULES 路径），再启动。" >&2
  exit 1
fi

echo "=== 启动服务 ==="
echo "base      : $BASE_MODEL"
echo "port      : $PORT    gpu_util: $GPUS    max_len: $MAX_LEN"
for entry in "${MODULES[@]}"; do echo "lora      : $entry"; done

exec vllm serve "$BASE_MODEL" \
  --served-model-name base \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPUS" \
  --enable-lora \
  --max-loras "$MAX_LORAS" \
  --max-lora-rank "$LORA_RANK" \
  --lora-modules "${MODULES[@]}" \
  --trust-remote-code
