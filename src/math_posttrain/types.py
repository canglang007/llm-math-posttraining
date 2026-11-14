from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    solution: str | None = None
    source: str = "gsm8k"
    split: str = "train"
    category: str | None = None
    level: str | None = None


@dataclass
class EvalSummary:
    dataset: str
    total: int
    correct: int
    extraction_failures: int
    accuracy: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
