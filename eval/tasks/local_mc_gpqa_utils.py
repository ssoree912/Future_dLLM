"""GPQA choice shuffling, copied from lm-eval's gpqa/n_shot/utils.py.

The module-level RNG and its seed are part of the task definition: the same seed
and the same document order give the same answer positions as a stock lm-eval
run, so scores stay comparable.
"""

from __future__ import annotations

import random

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


rng = random.Random(42)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        rng.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)
