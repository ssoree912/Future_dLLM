import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_eviction_similarity.py"
SPEC = importlib.util.spec_from_file_location("compare_eviction_similarity", SCRIPT)
similarity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(similarity)


def record(doc_id, text):
    return {
        "doc_id": doc_id,
        "doc_hash": f"hash-{doc_id}",
        "filter": "none",
        "filtered_resps": [text],
    }


class TextSimilarityTest(unittest.TestCase):
    def test_sequence_preserves_order_while_jaccard_uses_token_sets(self):
        left, right = "alpha beta", "beta alpha"
        self.assertLess(similarity.sequence_similarity(left, right), 1.0)
        self.assertEqual(similarity.jaccard_similarity(left, right), 1.0)

    def test_empty_outputs_are_identical(self):
        self.assertEqual(similarity.sequence_similarity("", ""), 1.0)
        self.assertEqual(similarity.jaccard_similarity("", ""), 1.0)

    def test_punctuation_is_a_token_and_case_is_folded(self):
        self.assertEqual(similarity.jaccard_similarity("Answer: FOUR", "answer: four"), 1.0)
        self.assertLess(similarity.jaccard_similarity("answer four", "answer: four"), 1.0)


class RunComparisonTest(unittest.TestCase):
    def test_compares_raw_generation_instead_of_task_filtered_answer(self):
        sample = record(0, "raw chain of thought")
        sample["resps"] = [["raw chain of thought"]]
        sample["filtered_resps"] = ["42"]
        self.assertEqual(similarity._response(sample), "raw chain of thought")

    def test_aligns_by_hash_not_line_order_and_recommends_current(self):
        full = {
            ("hash-0", "raw-generation"): record(0, "the exact answer"),
            ("hash-1", "raw-generation"): record(1, "return x + 1"),
        }
        future = {
            ("hash-1", "raw-generation"): record(1, "unrelated tokens"),
            ("hash-0", "raw-generation"): record(0, "wrong result"),
        }
        current = {
            ("hash-0", "raw-generation"): record(0, "the exact answer"),
            ("hash-1", "raw-generation"): record(1, "return x + 1"),
        }
        summary, rows = similarity.compare_runs(
            full, future, current, low_threshold=0.9, min_gain=0.01
        )
        self.assertEqual(summary["aligned_samples"], 2)
        self.assertEqual(summary["held_out_dataset_recommendation"], "current")
        self.assertEqual(summary["low_future_switched_to_current"], 2)
        self.assertTrue(all(row["diagnostic_choice"] == "current" for row in rows))

    def test_loads_nested_lm_eval_samples_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested"
            path.mkdir()
            sample = path / "samples_task.jsonl"
            sample.write_text(json.dumps(record(7, "result")) + "\n", encoding="utf-8")
            loaded = similarity.load_samples(tmp)
        self.assertEqual(loaded[("hash-7", "raw-generation")]["doc_id"], 7)

    def test_lm_eval_filters_do_not_duplicate_the_same_generation(self):
        strict = record(7, "same raw response")
        strict["filter"] = "strict-match"
        strict["resps"] = [["same raw response"]]
        flexible = dict(strict, filter="flexible-extract")
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples_task.jsonl"
            sample.write_text(
                json.dumps(strict) + "\n" + json.dumps(flexible) + "\n",
                encoding="utf-8",
            )
            loaded = similarity.load_samples(tmp)
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
