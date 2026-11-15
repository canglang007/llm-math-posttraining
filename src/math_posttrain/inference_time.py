"""推理时候选聚合：公开答案归一化后的多数投票与 oracle 上界。

纯逻辑模块，不加载模型，可在 Mac 上单测。候选解答由模型采样产生，
这里只负责把它们聚合成一个可判分的答案，不产生任何 ground truth。
"""
from __future__ import annotations

from collections import Counter

from .answers import extract_final_answer, is_correct


def majority_vote(candidates: list[str], dataset: str) -> str | None:
    """抽取每个候选的最终答案后投票。

    抽取失败的候选直接忽略；平票时取最早出现的答案，保证结果可复现。
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, text in enumerate(candidates):
        value = extract_final_answer(text, dataset)
        if value is None:
            continue
        counts[value] += 1
        first_seen.setdefault(value, index)
    if not counts:
        return None
    best = max(counts.values())
    tied = [value for value, count in counts.items() if count == best]
    return min(tied, key=lambda value: first_seen[value])


def vote_is_correct(candidates: list[str], reference: str, dataset: str) -> bool:
    """多数投票得到的答案是否等于公开答案。"""
    chosen = majority_vote(candidates, dataset)
    if chosen is None:
        return False
    wrapped = f"#### {chosen}" if dataset == "gsm8k" else f"\\boxed{{{chosen}}}"
    return is_correct(wrapped, reference, dataset)


def oracle_pass_at_k(candidates: list[str], reference: str, dataset: str) -> bool:
    """上界：k 个候选里是否至少有一个答对（验证信号天花板）。"""
    return any(is_correct(text, reference, dataset) for text in candidates)


def valid_candidate_count(candidates: list[str], dataset: str) -> int:
    """成功抽取出答案的候选数，用于衡量格式过滤的代价。"""
    return sum(1 for text in candidates if extract_final_answer(text, dataset) is not None)


def sampling_cost(num_questions: int, k: int, avg_new_tokens: float) -> dict[str, int]:
    """采样预算估算：总候选数与新增 token 数（不含 prompt）。"""
    samples = num_questions * k
    return {
        "questions": num_questions,
        "k": k,
        "samples": samples,
        "estimated_new_tokens": round(samples * avg_new_tokens),
    }
