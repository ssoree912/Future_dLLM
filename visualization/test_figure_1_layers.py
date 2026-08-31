import unittest

import torch

from visualization.figure_1_layers import (
    average_retained_attention,
    selection_frequency,
)


class LayerAggregationTest(unittest.TestCase):
    def test_average_applies_each_layer_mask_before_reduction(self):
        rows = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ])
        keeps = torch.tensor([[0, 2], [1, 2]])
        actual = average_retained_attention(rows, keeps)
        torch.testing.assert_close(actual, torch.tensor([
            [0.5, 4.0, 6.0],
            [2.0, 5.5, 9.0],
        ]))

    def test_selection_frequency_counts_layers_not_union(self):
        keeps = torch.tensor([[0, 2], [1, 2], [2, 3]])
        actual = selection_frequency(keeps, candidate_count=4)
        torch.testing.assert_close(
            actual, torch.tensor([1 / 3, 1 / 3, 1.0, 1 / 3])
        )


if __name__ == "__main__":
    unittest.main()
