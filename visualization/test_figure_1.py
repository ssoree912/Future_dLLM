import unittest

import torch

from future_dllm import (CustomCache, sparse_dllm_current_score,
                         teacher_current_attention_score)
from visualization.figure_1 import (
    aggregate_future_rows,
    candidate_regions,
    mask_future_attention,
    topk_metrics,
)


class SparseDLLMScoreTest(unittest.TestCase):
    def test_matches_query_then_head_average(self):
        q = torch.tensor([[[[1.0, 0.0], [3.0, 0.0]],
                           [[0.0, 2.0], [0.0, 4.0]]]])
        k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
        actual = sparse_dllm_current_score(q, k, pool_kernel_size=None)
        expected = torch.tensor([[1.0, 1.5, 2.5]])
        torch.testing.assert_close(actual, expected)

    def test_gqa_and_pooling_keep_candidate_length(self):
        q = torch.randn(1, 4, 3, 2)
        k = torch.randn(1, 2, 7, 2)
        score = sparse_dllm_current_score(q, k, pool_kernel_size=3)
        self.assertEqual(tuple(score.shape), (1, 7))

    def test_even_pooling_kernel_is_rejected(self):
        with self.assertRaises(ValueError):
            sparse_dllm_current_score(
                torch.randn(1, 1, 2, 2), torch.randn(1, 1, 3, 2),
                pool_kernel_size=2,
            )

    def test_cache_capture_uses_candidates_outside_current_block(self):
        cache = CustomCache(
            n_layers=1,
            device=torch.device("cpu"),
            keep_ratio=1.0,
            capture_current_scores=True,
            current_score_pool_kernel=None,
        )
        keys = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
        cache.update_cache(0, keys, torch.zeros_like(keys))
        queries = torch.tensor([[[[1.0], [2.0]]]])
        cache.filter_cache(0, queries, cur_filtered_len=2, block_len=2)
        torch.testing.assert_close(
            cache.current_scores[0], torch.tensor([[0.0, 1.5, 6.0, 7.5]])
        )
        torch.testing.assert_close(
            cache.get_cache(0)["k"].flatten(), torch.tensor([0.0, 1.0, 4.0, 5.0])
        )


class TeacherCurrentScoreTest(unittest.TestCase):
    def test_matches_teacher_kernel_with_current_queries(self):
        torch.manual_seed(7)
        q = torch.randn(1, 4, 3, 2)
        candidate_k = torch.randn(1, 2, 5, 2)
        block_k = torch.randn(1, 2, 3, 2)

        teacher = CustomCache(n_layers=1, device=torch.device("cpu"))
        teacher.capture_rows = True
        teacher.capture_per_head = True
        teacher.group_reduce = "mean"
        teacher.record_attention(0, q, torch.cat([candidate_k, block_k], dim=-2))
        expected = teacher.pending_rows[0].amax(dim=1)

        actual = teacher_current_attention_score(
            q, candidate_k, block_k, row_reduce="max",
            group_reduce="mean", per_head=True,
        ).squeeze(0)
        torch.testing.assert_close(actual, expected)

    def test_current_eviction_keeps_teacher_style_topk_per_kv_head(self):
        torch.manual_seed(11)
        cache = CustomCache(
            n_layers=1, device=torch.device("cpu"), keep_ratio=0.5,
            eviction_method="current", current_reduce=("max", "mean", True),
            baseline_order=True,
        )
        keys = torch.randn(1, 2, 8, 3)
        values = torch.arange(16, dtype=torch.float32).view(1, 2, 8, 1)
        queries = torch.randn(1, 4, 2, 3)
        cache.update_cache(0, keys, values)

        candidate_k = torch.cat([keys[:, :, :3], keys[:, :, 5:]], dim=2)
        candidate_v = torch.cat([values[:, :, :3], values[:, :, 5:]], dim=2)
        scores = teacher_current_attention_score(
            queries, candidate_k, keys[:, :, 3:5], row_reduce="max",
            group_reduce="mean", per_head=True,
        )
        keep = torch.topk(scores, k=3, dim=-1).indices.squeeze(0)
        heads = torch.arange(2)[:, None]
        expected = candidate_v[:, heads, keep]

        cache.filter_cache(0, queries, cur_filtered_len=3, block_len=2)
        torch.testing.assert_close(cache.get_cache(0)["v"], expected)


class FigureMetricsTest(unittest.TestCase):
    def test_mass_and_recall_use_rowmax_oracle(self):
        prediction = torch.tensor([0.1, 0.8, 0.7, 0.2])
        future = torch.tensor([0.9, 0.8, 0.1, 0.2])
        keep, mass, recall = topk_metrics(prediction, future, cache_budget=2)
        torch.testing.assert_close(keep, torch.tensor([1, 2]))
        self.assertAlmostEqual(mass, 0.9 / 2.0)
        self.assertAlmostEqual(recall, 0.5)

    def test_candidate_regions_exclude_current_block(self):
        block = {
            "prompt_length": 5,
            "block_start": 9,
            "block_length": 4,
            "candidate_indices": torch.arange(17),
        }
        regions = candidate_regions(block)
        self.assertEqual([region["count"] for region in regions], [5, 4, 8])
        self.assertEqual(sum(region["count"] for region in regions), 17)

    def test_sentence_aggregation_uses_teacher_aligned_rowmax(self):
        rows = torch.tensor([
            [0.1, 0.4],
            [0.3, 0.2],
            [0.5, 0.1],
            [0.2, 0.6],
        ])
        groups = [
            {"start": 0, "end": 2, "label": "S1"},
            {"start": 2, "end": 4, "label": "S2"},
        ]
        actual = aggregate_future_rows(rows, groups)
        torch.testing.assert_close(
            actual, torch.tensor([[0.3, 0.4], [0.5, 0.6]])
        )

    def test_retained_attention_heatmap_zeros_evicted_columns(self):
        rows = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ])
        actual = mask_future_attention(rows, torch.tensor([1, 3]))
        torch.testing.assert_close(actual, torch.tensor([
            [0.0, 0.2, 0.0, 0.4],
            [0.0, 0.6, 0.0, 0.8],
        ]))

if __name__ == "__main__":
    unittest.main()
