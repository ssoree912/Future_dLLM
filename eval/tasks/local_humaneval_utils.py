import math
import os


def _estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total < k:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod(
        1.0 - k / denominator
        for denominator in range(total - correct + 1, total + 1)
    )


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
) -> dict[str, float]:
    """Execute HumanEval candidates with lm-eval's guarded local executor."""
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError("HumanEval requires HF_ALLOW_CODE_EVAL=1")

    from lm_eval.tasks.cruxeval.utils import check_correctness

    requested = [k] if isinstance(k, int) else (k or [1])
    scores = {f"pass@{value}": [] for value in requested}
    for reference, candidates in zip(references, predictions, strict=True):
        outcomes = [
            check_correctness(f"{candidate}\n{reference}", timeout=3)
            for candidate in candidates
        ]
        for value in requested:
            scores[f"pass@{value}"].append(
                _estimate_pass_at_k(len(outcomes), sum(outcomes), value)
            )

    return {
        name: sum(values) / len(values) if values else 0.0
        for name, values in scores.items()
    }


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + response for response in responses]
        for responses, doc in zip(resps, docs, strict=True)
    ]
