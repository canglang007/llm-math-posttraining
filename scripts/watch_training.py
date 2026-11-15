"""训练过程观察器：从日志里读出进度、loss 趋势和健康判断。

用途是「不盲跑」——长时间训练时必须能随时回答三个问题：
  1. loss 在降吗？还是已经平了 / 在涨 / 出现 NaN？
  2. 还剩多久？
  3. 现在该继续，还是该停下来调配置？

用法：
    python scripts/watch_training.py --log /data/sft-v2.log --total-steps 400
    python scripts/watch_training.py --log /data/sft-v2.log --total-steps 400 --json

健康判断规则（保守，宁可早停）：
  - 出现 NaN / inf        → BAD，立即停
  - 最近 10 个点均值 > 最早 10 个点均值 → STUCK，学习率或数据可能有问题
  - 最近 10 个点均值比最早 10 个点低不到 5%  → FLAT，考虑加步数或调 lr
  - 其余                   → OK
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# HF Trainer 的日志行形如：
# {'loss': 0.4898, 'grad_norm': 1.23, 'learning_rate': 0.0002, 'epoch': 0.05}
LOSS_PATTERN = re.compile(r"['\"]loss['\"]\s*:\s*([0-9.eE+-]+|nan|inf)")
STEP_PATTERN = re.compile(r"['\"]epoch['\"]\s*:\s*([0-9.eE+-]+)")


def parse_log(path: Path) -> tuple[list[float], list[float]]:
    losses: list[float] = []
    epochs: list[float] = []
    if not path.exists():
        return losses, epochs
    text = path.read_text(encoding="utf-8", errors="ignore")
    # tqdm 用 \r 刷新，先统一成 \n
    for line in text.replace("\r", "\n").split("\n"):
        match = LOSS_PATTERN.search(line)
        if not match:
            continue
        raw = match.group(1)
        try:
            losses.append(float(raw))
        except ValueError:
            losses.append(float("nan"))
        epoch_match = STEP_PATTERN.search(line)
        if epoch_match:
            try:
                epochs.append(float(epoch_match.group(1)))
            except ValueError:
                pass
    return losses, epochs


def summarize(losses: list[float], total_steps: int, window: int = 10) -> dict:
    if not losses:
        return {"status": "NO_DATA", "logged_points": 0}

    if any(value != value or value in (float("inf"), float("-inf")) for value in losses):
        return {"status": "BAD", "reason": "loss 出现 NaN/inf", "logged_points": len(losses)}

    head = losses[:window]
    tail = losses[-window:]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    change = (tail_mean - head_mean) / head_mean if head_mean else 0.0

    if len(losses) >= 2 * window and change > 0.02:
        status, reason = "STUCK", "最近均值反而上升，可能没在学或 lr 过大"
    elif len(losses) >= 2 * window and change > -0.05:
        status, reason = "FLAT", "降幅不足 5%，收敛已趋平"
    else:
        status, reason = "OK", "loss 在下降"

    return {
        "status": status,
        "reason": reason,
        "logged_points": len(losses),
        "first_mean": round(head_mean, 4),
        "last_mean": round(tail_mean, 4),
        "relative_change": round(change, 4),
        "min_loss": round(min(losses), 4),
        "last_loss": round(losses[-1], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    losses, epochs = parse_log(args.log)
    summary = summarize(losses, args.total_steps, args.window)
    if epochs:
        summary["last_epoch"] = round(epochs[-1], 3)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
        if summary["status"] in {"BAD", "STUCK"}:
            sys.exit(1)
        return

    if summary["status"] == "NO_DATA":
        print(f"还没有 loss 记录（日志：{args.log}）")
        return

    print(f"已记录 {summary['logged_points']} 个 loss 点")
    print(f"最初 {args.window} 点均值：{summary['first_mean']}    最近 {args.window} 点均值：{summary['last_mean']}")
    print(f"相对变化：{summary['relative_change']:+.1%}    最低：{summary['min_loss']}    最新：{summary['last_loss']}")
    if "last_epoch" in summary:
        print(f"当前 epoch：{summary['last_epoch']}")
    print(f"判断：{summary['status']} —— {summary['reason']}")
    if summary["status"] in {"BAD", "STUCK"}:
        print("建议：停下来看数据和配置，不要继续烧卡。")
        sys.exit(1)


if __name__ == "__main__":
    main()
