"""训练过程观察器的判定逻辑。

这条测试守的是「不盲跑」：出现 NaN 或 loss 不下探时必须能报警，
否则长时间训练会白烧算力。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "watch_training.py"


def _run(tmp_path: Path, lines: list[str]) -> tuple[dict, int]:
    log = tmp_path / "train.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--log", str(log), "--json"],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1]), result.returncode


def test_descending_loss_is_ok(tmp_path: Path) -> None:
    lines = [f"{{'loss': {1.0 - i * 0.05:.3f}, 'epoch': {i * 0.01:.2f}}}" for i in range(30)]
    payload, _ = _run(tmp_path, lines)
    assert payload["status"] == "OK"
    assert payload["relative_change"] < -0.5


def test_nan_loss_is_flagged_bad(tmp_path: Path) -> None:
    lines = ["{'loss': 0.9, 'epoch': 0.1}", "{'loss': nan, 'epoch': 0.2}"]
    payload, code = _run(tmp_path, lines)
    assert payload["status"] == "BAD"
    assert code == 1


def test_rising_loss_is_stuck(tmp_path: Path) -> None:
    lines = [f"{{'loss': {1.0 + i * 0.05:.3f}, 'epoch': {i * 0.01:.2f}}}" for i in range(30)]
    payload, code = _run(tmp_path, lines)
    assert payload["status"] == "STUCK"
    assert code == 1


def test_flat_loss_is_reported_flat(tmp_path: Path) -> None:
    lines = [f"{{'loss': {0.50 - (i % 2) * 0.001:.4f}, 'epoch': {i * 0.01:.2f}}}" for i in range(30)]
    payload, _ = _run(tmp_path, lines)
    assert payload["status"] == "FLAT"


def test_missing_log_reports_no_data(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--log", str(tmp_path / "nope.log"), "--json"],
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.strip())["status"] == "NO_DATA"
