"""Download public math data with provenance. No model calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from math_posttrain.answers import extract_boxed, extract_gsm8k_answer  # noqa: E402
from math_posttrain.data import write_jsonl  # noqa: E402

GSM8K_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
LICENSES = {"gsm8k": "MIT", "math": "MIT"}


def download_jsonl(url: str, destination: Path, timeout: int = 30) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    destination.write_bytes(data)
    return len(data)


def _gsm8k_from_hf(split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [(str(item["question"]), str(item["answer"])) for item in dataset]


def prepare_gsm8k(output: Path) -> dict:
    provenance = {"license": LICENSES["gsm8k"], "urls": GSM8K_URLS, "splits": {}}
    for split, url in GSM8K_URLS.items():
        source = url
        try:
            raw_path = output / "gsm8k" / f"{split}.raw.jsonl"
            download_jsonl(url, raw_path)
            items = []
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    items.append((str(row["question"]), str(row["answer"])))
        except Exception as raw_error:
            try:
                items = _gsm8k_from_hf(split)
                source = "hf:openai/gsm8k"
            except Exception as hf_error:
                raise RuntimeError(
                    f"could not download GSM8K {split} from GitHub or HF: {raw_error}; {hf_error}"
                ) from hf_error
        rows = []
        for index, (question, solution) in enumerate(items):
            final = extract_gsm8k_answer(solution)
            if final is None:
                raise ValueError(f"GSM8K {split} row {index} has no final answer")
            rows.append(
                {
                    "id": f"gsm8k-{split}-{index}",
                    "question": question,
                    "answer": final,
                    "solution": solution,
                }
            )
        destination = output / "gsm8k" / f"{split}.jsonl"
        write_jsonl(destination, rows)
        provenance["splits"][split] = {"count": len(rows), "path": str(destination), "source": source}
    return provenance


def prepare_math(output: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("MATH requires datasets: pip install -e '.[eval]'") from error
    provenance = {"license": LICENSES["math"], "splits": {}}
    for split in ("train", "test"):
        rows = []
        missing_answers = 0
        for config in MATH_CONFIGS:
            dataset = load_dataset("EleutherAI/hendrycks_math", config, split=split)
            for index, item in enumerate(dataset):
                solution = str(item["solution"])
                answer = extract_boxed(solution)
                if answer is None:
                    missing_answers += 1
                    continue
                rows.append(
                    {
                        "id": f"math-{config}-{split}-{index}",
                        "question": item["problem"],
                        "answer": answer,
                        "solution": solution,
                        "category": config,
                        "level": item.get("level"),
                    }
                )
        destination = output / "math" / f"{split}.jsonl"
        write_jsonl(destination, rows)
        provenance["splits"][split] = {
            "count": len(rows),
            "missing_answers": missing_answers,
            "path": str(destination),
        }
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gsm8k", "math", "all"), default="gsm8k")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "gsm8k": {"license": LICENSES["gsm8k"], "urls": GSM8K_URLS},
        "math": {"license": LICENSES["math"], "repo": "https://github.com/hendrycks/math", "configs": MATH_CONFIGS},
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    if args.dataset in ("gsm8k", "all"):
        provenance["gsm8k"] = prepare_gsm8k(args.output)
    if args.dataset in ("math", "all"):
        provenance["math"] = prepare_math(args.output)
    provenance["downloaded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
