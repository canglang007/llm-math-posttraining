#!/usr/bin/env bash
# 本地（Mac / 无 GPU 机器）一键启动演示服务。
#
#   bash deploy/run_local.sh                 # 默认 Qwen/Qwen3-4B-Base，端口 8000
#   MODEL=/path/to/Qwen3-4B-Base bash deploy/run_local.sh
#   bash deploy/run_local.sh --dry-run       # 只检查环境与 adapter 路径
#
# 与服务器版的关系：接口完全一致（/v1/models、/v1/completions），
# demo/live.html 不需要改，只把端点换成 http://localhost:8000。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
PORT="${PORT:-8000}"

# 自动挑一个装了 torch/transformers/peft 的解释器：
# 机器上常有多个 python（系统自带 / Homebrew / conda），默认的 python3 未必有依赖。
pick_python() {
  local candidates=()
  [ -n "${PYTHON:-}" ] && candidates+=("$PYTHON")
  candidates+=("python3.13" "python3" "/opt/anaconda3/bin/python3" "/opt/homebrew/bin/python3")
  for cand in "${candidates[@]}"; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import torch, transformers, peft" >/dev/null 2>&1; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

echo "== 环境检查 =="
if ! PYTHON="$(pick_python)"; then
  echo "  没有一个解释器同时具备 torch / transformers / peft。"
  echo "  先安装（注意用同一个解释器）：pip install -e '.[train]'"
  echo "  或显式指定：PYTHON=/path/to/python bash deploy/run_local.sh"
  for cand in python3.13 python3 /opt/anaconda3/bin/python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    echo "    $cand -> $("$cand" -c "
import importlib
missing=[m for m in ('torch','transformers','peft') if not importlib.util.find_spec(m)]
print('缺 ' + ', '.join(missing) if missing else '齐全')
" 2>/dev/null || echo '无法执行')"
  done
  exit 1
fi
echo "  解释器：$PYTHON"
"$PYTHON" - <<'PY'
for name in ("torch", "transformers", "peft"):
    mod = __import__(name)
    print(f"  {name:14s} {getattr(mod, '__version__', '?')}")
import torch
print(f"  device        mps={torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    print("  [warn] 未检测到 MPS，将退回 CPU（4B 模型会非常慢，仅适合验证接口）")
PY

echo
echo "== 底座模型 =="
if [ -d "$MODEL" ]; then
  echo "  本地目录：$MODEL"
else
  echo "  将从 HuggingFace 拉取：$MODEL（约 8 GB，首次较慢）"
  echo "  国内网络可试：export HF_ENDPOINT=https://hf-mirror.com"
  echo "  （若有代理/企业网关挡住镜像域名，就不要设该变量，走默认端点）"
fi

echo
echo "== adapter =="
for spec in \
  "sft=experiments/sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400" \
  "e2b=experiments/rft/qwen3-4b-gsm8k-rft-300q" \
  "a2=experiments/rft/qwen3-4b-gsm8k-rft-a2nf-300q" \
  "dpo=experiments/dpo/qwen3-4b-gsm8k-dpo-from-sft"; do
  path="${spec#*=}"
  if [ -f "$path/adapter_config.json" ]; then
    echo "  ✓ ${spec%%=*}  $path"
  else
    echo "  - ${spec%%=*}  缺失（跳过）：$path"
  fi
done

echo
echo "== 启动 =="
exec "$PYTHON" deploy/local_server.py --model-path "$MODEL" --port "$PORT" "$@"
