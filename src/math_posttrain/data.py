import json
from pathlib import Path

from .types import Example


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_examples(path: str | Path, source: str, split: str) -> list[Example]:
    examples = []
    for index, row in enumerate(read_jsonl(path)):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"{path}: row {index} needs string question/answer")
        examples.append(
            Example(
                id=str(row.get("id", index)),
                question=question,
                answer=answer,
                solution=row.get("solution"),
                source=source,
                split=split,
                category=row.get("category"),
                level=row.get("level"),
            )
        )
    return examples
