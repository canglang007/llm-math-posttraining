"""采样产物 → 候选级数据的展开逻辑。

重点守护的是「数据纪律」：dev 产物必须被强制改名，避免误用于训练。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "expand_sampling_candidates.py"


def _write_sampling(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _run(tmp_path: Path, *extra: str) -> dict:
    sampling = tmp_path / "sampling.jsonl"
    _write_sampling(
        sampling,
        [
            {"id": "q1", "question": "1+1?", "reference": "2", "candidates": ["#### 2", "#### 3", "#### 2"]},
            {"id": "q2", "question": "2+2?", "reference": "4", "candidates": ["#### 5", "#### 6"]},
        ],
    )
    output = tmp_path / "candidates.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sampling-file",
            str(sampling),
            "--dataset",
            "gsm8k",
            "--split",
            "train",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_drops_questions_without_any_correct_candidate(tmp_path: Path) -> None:
    stats = _run(tmp_path)
    # q2 全错，被整题丢弃；q1 保留 3 条
    assert stats["kept"] == 3
    assert stats["dropped_questions"] == 1
    assert stats["correct"] == 2


def test_correct_only_filters_wrong_candidates(tmp_path: Path) -> None:
    stats = _run(tmp_path, "--correct-only")
    assert stats["kept"] == 2
    rows = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["correct"] is True for row in rows)
    assert {row["id"] for row in rows} == {"q1"}


def test_train_split_output_name_is_untouched(tmp_path: Path) -> None:
    stats = _run(tmp_path)
    assert Path(stats["output"]).name == "candidates.jsonl"
