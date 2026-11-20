"""Lightweight behavioral tests for model evaluation; require NumPy and PyTorch.

These are skipped if PyTorch is not available; CSV validation still runs.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import numpy as np
    from methods import (Interaction, build_user_histories, evaluate_model,
                         make_episode_for_user, ndcg_from_scores,
                         train_test_user_split)


@unittest.skipUnless(HAS_TORCH, "Install requirements.txt to run model tests")
class ExperimentBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.histories = {
            user_id: [(item_id, float((item_id + user_id) % 5 + 1), item_id)
                      for item_id in range(1, 31)] for user_id in range(1, 7)
        }
        self.embeddings = {item_id: np.array([float(item_id), 1.0])
                           for item_id in range(1, 31)}

    def test_user_split_disjoint_and_seeded(self):
        train, test = train_test_user_split(self.histories, min_ratings=25, seed=42)
        self.assertTrue(set(train).isdisjoint(test))
        self.assertEqual((train, test), train_test_user_split(self.histories, min_ratings=25, seed=42))

    def test_support_query_non_overlap(self):
        import random
        episode = make_episode_for_user(1, self.histories, support_size=5, rng=random.Random(42))
        self.assertEqual(len(episode.support_item_ids), 5)
        self.assertEqual(len(episode.query_item_ids), 10)
        self.assertTrue(set(episode.support_item_ids).isdisjoint(episode.query_item_ids))

    def test_ndcg_is_one_for_perfect_ranking(self):
        relevance = np.array([5., 1., 4., 2., 3.])
        self.assertAlmostEqual(ndcg_from_scores(relevance, relevance), 1.0)
        self.assertLess(ndcg_from_scores(relevance, -relevance), 1.0)

    def test_same_seed_produces_identical_test_episodes(self):
        observed_a, observed_b = [], []

        def record_into(destination):
            def predictor(episode, embeddings):
                destination.append((episode.user_id, tuple(episode.support_item_ids),
                                    tuple(episode.query_item_ids)))
                return np.zeros(len(episode.query_item_ids), dtype=np.float32)
            return predictor

        for store in (observed_a, observed_b):
            evaluate_model("test", record_into(store), [1, 2, 3],
                           self.histories, self.embeddings, support_size=5, seed=47)
        self.assertEqual(observed_a, observed_b)
        self.assertEqual(len(observed_a), 3)


if __name__ == "__main__":
    unittest.main()
