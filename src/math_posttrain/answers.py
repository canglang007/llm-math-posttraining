import re

_GSM8K_ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def _normalize_number(value: str) -> str | None:
    value = value.strip().replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else str(number)


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    body_start = index
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index].strip()
        index += 1
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    matches = _GSM8K_ANSWER.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    boxed = extract_boxed(text)
    return _normalize_number(boxed) if boxed else None


def normalize_math(text: str) -> str:
    value = text.strip().strip("$").strip()
    value = re.sub(r"\\left|\\right", "", value)
    value = re.sub(r"\\[,;!]", "", value)
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".")


def math_equivalent(prediction: str, reference: str) -> bool:
    left = normalize_math(prediction)
    right = normalize_math(reference)
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except ValueError:
        pass
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
    except Exception:
        return False
    try:
        return simplify(parse_latex(left) - parse_latex(right)) == 0
    except Exception:
        return False


def extract_final_answer(text: str, dataset: str) -> str | None:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(text)
    if dataset == "math":
        return extract_boxed(text)
    raise ValueError(f"unknown dataset: {dataset}")


def is_correct(prediction: str, reference: str, dataset: str) -> bool:
    extracted = extract_final_answer(prediction, dataset)
    if extracted is None:
        return False
    if dataset == "gsm8k":
        expected = _normalize_number(reference)
        if expected is None:
            return False
        return extracted == expected or abs(float(extracted) - float(expected)) <= 1e-9
    if dataset == "math":
        return math_equivalent(extracted, reference)
    raise ValueError(f"unknown dataset: {dataset}")
