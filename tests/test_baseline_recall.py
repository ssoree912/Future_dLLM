import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/baseline_recall.py"
SPEC = importlib.util.spec_from_file_location("baseline_recall", SCRIPT)
baseline_recall = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline_recall)


class RankingSimilarityTest(unittest.TestCase):
    def test_jaccard_uses_union_of_equal_budget_topk_sets(self):
        prediction = torch.tensor([9.0, 8.0, 1.0, 0.0])
        target = torch.tensor([9.0, 1.0, 8.0, 0.0])
        # At a 50% budget, {0, 1} intersects {0, 2} once and their union is 3.
        actual = baseline_recall.jaccard_at(prediction, target, ratios=(0.5,))
        self.assertAlmostEqual(actual[0], 1 / 3)

    def test_matching_topk_has_unit_recall_and_jaccard(self):
        prediction = torch.tensor([3.0, 2.0, 1.0])
        self.assertEqual(
            baseline_recall.recall_at(prediction, prediction, ratios=(2 / 3,)), [1.0]
        )
        self.assertEqual(
            baseline_recall.jaccard_at(prediction, prediction, ratios=(2 / 3,)), [1.0]
        )

    def test_per_head_metrics_average_heads_and_keep_budget_per_head(self):
        prediction = torch.tensor([[9.0, 8.0, 1.0, 0.0],
                                   [9.0, 8.0, 1.0, 0.0]])
        target = torch.tensor([[9.0, 8.0, 1.0, 0.0],
                               [9.0, 1.0, 8.0, 0.0]])
        # Head 0 is exact; head 1 has recall 1/2 and Jaccard 1/3.
        self.assertEqual(baseline_recall.recall_at(prediction, target, (0.5,)), [0.75])
        self.assertAlmostEqual(
            baseline_recall.jaccard_at(prediction, target, (0.5,))[0], 2 / 3
        )


if __name__ == "__main__":
    unittest.main()
