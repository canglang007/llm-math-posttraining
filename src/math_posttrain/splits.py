import hashlib
from collections.abc import Iterable, Sequence


def stable_key(value: str, seed: int = 42) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def carve_dev(ids: Iterable[str], dev_size: int = 500, seed: int = 42) -> tuple[list[str], list[str]]:
    unique = list(dict.fromkeys(str(item) for item in ids))
    if dev_size <= 0 or dev_size >= len(unique):
        raise ValueError(f"dev_size must be between 1 and {len(unique) - 1}, got {dev_size}")
    ordered = sorted(unique, key=lambda value: stable_key(value, seed))
    return ordered[dev_size:], ordered[:dev_size]


def assert_disjoint(*groups: Sequence[str]) -> None:
    sets = [set(group) for group in groups]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"split overlap: {sorted(overlap)[:5]}")
