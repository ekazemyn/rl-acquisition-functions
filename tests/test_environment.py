import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from rlbo import PoolOptimizationEnv
from rlbo.datasets import _to_tabular_dataset, sample_pool, sample_pool_indices


class PoolOptimizationEnvTests(unittest.TestCase):
    def setUp(self):
        self.pool_size = 60
        self.features = np.arange(self.pool_size * 2, dtype=np.float32).reshape(
            self.pool_size, 2
        )
        self.labels = 1000.0 + np.arange(self.pool_size, dtype=np.float64)
        self.provider_calls = []
        self.env = PoolOptimizationEnv(
            features=self.features,
            labels=self.labels,
            candidate_provider=self.candidate_provider,
            observation_size=2,
            initial_points=3,
            budget=30,
            shortlist_size=20,
        )

    def candidate_provider(
        self,
        evaluated_indices,
        observed_targets,
        available_indices,
        candidate_features,
        evaluated_features,
    ):
        self.provider_calls.append((evaluated_indices.copy(), observed_targets.copy()))
        scores = candidate_features[:, 0].astype(np.float64)
        distances = np.linalg.norm(
            candidate_features[:, None, :] - evaluated_features[None, :, :], axis=2
        ).min(axis=1)
        return np.column_stack([scores, distances]), scores

    def test_budget_shortlist_and_reward_contract(self):
        observation, reset_info = self.env.reset(
            seed=7, options={"initial_indices": [0, 1, 2]}
        )
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(observation["candidates"].shape, (20, 2))
        self.assertEqual(observation["budget_fraction"].tolist(), [1.0])

        initial_indices = reset_info["initial_indices"]
        initial_best = float(np.max(self.labels[initial_indices]))
        evaluated = set(initial_indices.tolist())
        total_reward = 0.0

        for step_number in range(30):
            observation, reward, terminated, truncated, info = self.env.step(0)
            selected_index = info["selected_index"]
            self.assertEqual(selected_index, info["shortlist_indices"][0])
            self.assertNotIn(selected_index, evaluated)
            evaluated.add(selected_index)
            total_reward += reward
            self.assertEqual(reward, info["raw_improvement"])
            self.assertFalse(truncated)
            self.assertEqual(terminated, step_number == 29)
            self.assertTrue(self.env.observation_space.contains(observation))

        final_best = float(np.max(self.labels[list(evaluated)]))
        self.assertEqual(len(evaluated), 33)
        self.assertAlmostEqual(total_reward, final_best - initial_best)
        self.assertEqual(observation["budget_fraction"].tolist(), [0.0])

        for indices, observed_targets in self.provider_calls:
            self.assertEqual(len(indices), len(observed_targets))
            np.testing.assert_array_equal(observed_targets, self.labels[indices])

    def test_duplicate_initial_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct valid"):
            self.env.reset(options={"initial_indices": [1, 1, 2]})

    def test_gymnasium_api(self):
        self.assertIsInstance(self.env, gym.Env)
        check_env(self.env, skip_render_check=True)


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _to_tabular_dataset(
            "synthetic",
            np.array([[0.0, 1.0], [2.0, 3.0], [np.nan, 4.0], [6.0, 7.0]]),
            np.array([[10.0], [11.0], [12.0], [np.inf]]),
            "synthetic test fixture",
        )

    def test_nonfinite_rows_are_removed_and_arrays_aligned(self):
        self.assertEqual(self.dataset.features.shape, (2, 2))
        np.testing.assert_array_equal(self.dataset.targets, [10.0, 11.0])
        self.assertEqual(self.dataset.feature_names, ("feature_0", "feature_1"))
        self.assertEqual(self.dataset.target_name, "target")

    def test_pool_sampling_is_reproducible_and_without_replacement(self):
        indices_a = sample_pool_indices(n_rows=20, pool_size=8, seed=31)
        indices_b = sample_pool_indices(n_rows=20, pool_size=8, seed=31)
        np.testing.assert_array_equal(indices_a, indices_b)
        self.assertEqual(len(np.unique(indices_a)), 8)

        pool, source_indices = sample_pool(self.dataset, pool_size=2, seed=31)
        np.testing.assert_array_equal(pool.features, self.dataset.features[source_indices])
        np.testing.assert_array_equal(pool.targets, self.dataset.targets[source_indices])

    def test_pool_larger_than_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            sample_pool_indices(n_rows=3, pool_size=4, seed=0)


if __name__ == "__main__":
    unittest.main()