import unittest

import numpy as np

from rlbo.baselines import BASELINE_METHODS, run_baseline_episode
from rlbo.surrogate import EICandidateProvider


class BaselineEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.features = np.column_stack(
            [np.linspace(-2.0, 2.0, 18), np.cos(np.linspace(-2.0, 2.0, 18))]
        )
        self.targets = np.square(self.features[:, 0]) + 0.25 * self.features[:, 1]
        self.initial_indices = np.array([1, 6, 12])
        self.budget = 4

    def _run(self, method, targets=None, seed=17):
        return run_baseline_episode(
            features=self.features,
            targets=self.targets if targets is None else targets,
            method=method,
            initial_indices=self.initial_indices,
            budget=self.budget,
            seed=seed,
        )

    def test_all_methods_use_same_start_and_return_regret_curve(self):
        results = {method: self._run(method) for method in BASELINE_METHODS}
        first_best_values = [result.best_values[0] for result in results.values()]
        first_regrets = [result.simple_regret[0] for result in results.values()]
        self.assertTrue(np.allclose(first_best_values, first_best_values[0]))
        self.assertTrue(np.allclose(first_regrets, first_regrets[0]))

        for result in results.values():
            self.assertEqual(len(result.selected_indices), self.budget)
            self.assertEqual(len(result.best_values), self.budget + 1)
            self.assertEqual(len(result.simple_regret), self.budget + 1)
            self.assertTrue(np.all(np.diff(result.best_values) >= 0.0))
            self.assertTrue(np.all(np.diff(result.simple_regret) <= 1e-12))
            self.assertEqual(len(np.unique(result.selected_indices)), self.budget)
            self.assertFalse(
                np.isin(result.selected_indices, self.initial_indices).any()
            )

    def test_policies_do_not_use_unrevealed_targets_for_selection(self):
        altered_targets = self.targets.copy()
        hidden = np.setdiff1d(np.arange(len(self.targets)), self.initial_indices)
        altered_targets[hidden] += np.linspace(100.0, 200.0, len(hidden))

        for method in BASELINE_METHODS:
            with self.subTest(method=method):
                original = run_baseline_episode(
                    self.features,
                    self.targets,
                    method,
                    self.initial_indices,
                    budget=1,
                    seed=23,
                )
                altered = run_baseline_episode(
                    self.features,
                    altered_targets,
                    method,
                    self.initial_indices,
                    budget=1,
                    seed=23,
                )
                self.assertEqual(original.selected_indices[0], altered.selected_indices[0])

    def test_ei_baseline_matches_provider_rank_zero(self):
        available = np.setdiff1d(np.arange(len(self.targets)), self.initial_indices)
        provider = EICandidateProvider()
        _, scores = provider(
            self.initial_indices,
            self.targets[self.initial_indices],
            available,
            self.features[available],
            self.features[self.initial_indices],
        )
        result = run_baseline_episode(
            features=self.features,
            targets=self.targets,
            method="ei",
            initial_indices=self.initial_indices,
            budget=1,
        )
        self.assertEqual(result.selected_indices[0], available[int(np.argmax(scores))])

    def test_random_baseline_is_reproducible_for_a_seed(self):
        first = self._run("random", seed=42)
        second = self._run("random", seed=42)
        np.testing.assert_array_equal(first.selected_indices, second.selected_indices)


if __name__ == "__main__":
    unittest.main()