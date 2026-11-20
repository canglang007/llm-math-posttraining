#!/usr/bin/env bash
# 从本地备份包恢复「演示所需」的 4 个 LoRA adapter（约 320 MB），而不是解包整个 1.4 GB。
#
#   bash deploy/restore_adapters.sh                     # 默认读 data/server-backup/project-backup-20261118.tar.gz
#   BACKUP=/path/to/backup.tar.gz bash deploy/restore_adapters.sh
#
# 为什么需要：本地版 demo 要并排对比 sft / e2b / a2 / dpo 四个后训练结果，
# 而 adapter 不入库（体积），只有备份包里有。已存在的目录会被跳过。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKUP="${BACKUP:-data/server-backup/project-backup-20261118.tar.gz}"
PREFIX="llm-math-posttraining/experiments"

if [ ! -f "$BACKUP" ]; then
  echo "找不到备份包：$BACKUP" >&2
  echo "可用 BACKUP=/path/to/project-backup-*.tar.gz 指定" >&2
  exit 1
fi

TARGETS=(
  "sft/qwen3-4b-gsm8k-sft-v3/checkpoint-400"
  "rft/qwen3-4b-gsm8k-rft-300q"
  "rft/qwen3-4b-gsm8k-rft-a2nf-300q"
  "dpo/qwen3-4b-gsm8k-dpo-from-sft"
)

missing=()
for rel in "${TARGETS[@]}"; do
  dst="experiments/$rel"
  if [ -f "$dst/adapter_config.json" ]; then
    echo "  ✓ 已存在，跳过：$dst"
  else
    missing+=("$rel")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "  → 一次解出 ${#missing[@]} 个目录（单次读包，避免重复解压）"
  args=()
  for rel in "${missing[@]}"; do
    args+=("$PREFIX/$rel")
  done
  # --strip-components=1 去掉顶层 llm-math-posttraining/
  if ! tar xzf "$BACKUP" -C . --strip-components=1 "${args[@]}" 2>/dev/null; then
    wild=()
    for rel in "${missing[@]}"; do wild+=("$PREFIX/$rel/*"); done
    tar xzf "$BACKUP" -C . --strip-components=1 --wildcards "${wild[@]}"
  fi
fi

echo
echo "== 结果 =="
for rel in "${TARGETS[@]}"; do
  dst="experiments/$rel"
  if [ -f "$dst/adapter_config.json" ]; then
    size=$(du -sh "$dst" 2>/dev/null | cut -f1)
    echo "  ✓ $rel  ($size)"
  else
    echo "  ✗ $rel  仍然缺失"
  fi
done
echo
echo "下一步：bash deploy/run_local.sh"
