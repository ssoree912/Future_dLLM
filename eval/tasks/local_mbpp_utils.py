"""MBPP scoring, reading local parquet and executing candidates locally.

lm-eval's own ``tasks/mbpp/utils.py`` scores through ``evaluate.load("code_eval")``,
which reaches the Hub at import time and refuses to import at all without
HF_ALLOW_CODE_EVAL already set. Every other task here runs offline, so this uses
the same guarded executor ``local_humaneval_utils`` uses and keeps the semantics
identical -- candidate, newline, then that item's three asserts.

``list_fewshot_samples`` is lm-eval's list verbatim (dumped from
``lm_eval.tasks.mbpp.utils``), not a paraphrase: the three shots are part of the
benchmark, and retyping them would quietly make this a different one.
"""

import math
import os


def _estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    """Same estimator as local_humaneval_utils, repeated rather than imported.

    lm-eval loads a task's ``!function`` module by file path, not as part of a
    package, so one task utils file cannot import another by name.
    """
    if total < k:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod(
        1.0 - k / denominator
        for denominator in range(total - correct + 1, total + 1)
    )


def pass_at_1(references, predictions) -> float:
    """pass@1 for one item: run each candidate against that item's asserts."""
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError("MBPP requires HF_ALLOW_CODE_EVAL=1")

    from lm_eval.tasks.cruxeval.utils import check_correctness

    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]

    scores = []
    for reference, candidates in zip(references, predictions, strict=True):
        outcomes = [check_correctness(f"{c}\n{reference}", timeout=3) for c in candidates]
        scores.append(_estimate_pass_at_k(len(outcomes), sum(outcomes), 1))
    return sum(scores) / len(scores) if scores else 0.0


def list_fewshot_samples():
    return [   {   'task_id': 2,
        'text': 'Write a function to find the similar elements from the given two '
                'tuple lists.',
        'code': 'def similar_elements(test_tup1, test_tup2):\r\n'
                '  res = tuple(set(test_tup1) & set(test_tup2))\r\n'
                '  return (res) ',
        'test_list': [   'assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, '
                         '5)',
                         'assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)',
                         'assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) '
                         '== (13, 14)'],
        'is_fewshot': True},
    {   'task_id': 3,
        'text': 'Write a python function to identify non-prime numbers.',
        'code': 'import math\r\n'
                'def is_not_prime(n):\r\n'
                '    result = False\r\n'
                '    for i in range(2,int(math.sqrt(n)) + 1):\r\n'
                '        if n % i == 0:\r\n'
                '            result = True\r\n'
                '    return result',
        'test_list': [   'assert is_not_prime(2) == False',
                         'assert is_not_prime(10) == True',
                         'assert is_not_prime(35) == True'],
        'is_fewshot': True},
    {   'task_id': 4,
        'text': 'Write a function to find the largest integers from a given list of '
                'numbers using heap queue algorithm.',
        'code': 'import heapq as hq\r\n'
                'def heap_queue_largest(nums,n):\r\n'
                '  largest_nums = hq.nlargest(n, nums)\r\n'
                '  return largest_nums',
        'test_list': [   'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                         '58],3)==[85, 75, 65] ',
                         'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                         '58],2)==[85, 75] ',
                         'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, '
                         '58],5)==[85, 75, 65, 58, 35]'],
        'is_fewshot': True}]
