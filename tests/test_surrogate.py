import unittest

import numpy as np

from rlbo import PoolOptimizationEnv
from rlbo.surrogate import (
    CANDIDATE_STATE_FEATURES,
    EICandidateProvider,
    expected_improvement,
)


class ExpectedImprovementTests(unittest.TestCase):
    def test_zero_uncertainty_reduces_to_positive_part(self):
        values = expected_improvement(
            mean=np.array([1.0, 2.0, 3.0]),
            standard_deviation=np.zeros(3),
            incumbent=2.0,
            xi=0.1,
        )
        np.testing.assert_allclose(values, [0.0, 0.0, 0.9], atol=1e-12)

    def test_invalid_uncertainty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            expected_improvement(np.array([1.0]), np.array([-0.1]), incumbent=0.0)


class EICandidateProviderTests(unittest.TestCase):
    def setUp(self):
        self.features = np.column_stack(
            [np.linspace(-2.0, 2.0, 12), np.sin(np.linspace(-2.0, 2.0, 12))]
        )
        self.targets = np.square(self.features[:, 0]) - self.features[:, 1]
        self.evaluated_indices = np.array([0, 3, 6, 9])
        self.available_indices = np.setdiff1d(
            np.arange(len(self.targets)), self.evaluated_indices
        )

    def _provide(self, targets):
        provider = EICandidateProvider(xi=0.01)
        return provider(
            self.evaluated_indices,
            targets[self.evaluated_indices],
            self.available_indices,
            self.features[self.available_indices],
            self.features[self.evaluated_indices],
        )

    def test_returns_finite_five_feature_state_and_ei_scores(self):
        candidate_state, scores = self._provide(self.targets)
        self.assertEqual(candidate_state.shape, (len(self.available_indices), 5))
        self.assertEqual(len(CANDIDATE_STATE_FEATURES), 5)
        self.assertTrue(np.isfinite(candidate_state).all())
        self.assertTrue(np.all(scores >= 0.0))
        np.testing.assert_array_equal(candidate_state[:, 2], scores)
        self.assertTrue(np.all(candidate_state[:, 4] >= 0.0))

    def test_state_is_invariant_to_positive_target_unit_changes(self):
        original_state, original_scores = self._provide(self.targets)
        rescaled_state, rescaled_scores = self._provide(100.0 * self.targets + 37.0)
        np.testing.assert_allclose(rescaled_state, original_state, atol=1e-8)
        np.testing.assert_allclose(rescaled_scores, original_scores, atol=1e-8)

    def test_provider_runs_inside_pool_environment(self):
        environment = PoolOptimizationEnv(
            features=self.features,
            labels=self.targets,
            candidate_provider=EICandidateProvider(),
            observation_size=len(CANDIDATE_STATE_FEATURES),
            initial_points=3,
            budget=3,
            shortlist_size=5,
        )
        observation, reset_info = environment.reset(
            seed=5, options={"initial_indices": [0, 3, 6]}
        )
        self.assertEqual(observation["candidates"].shape, (5, 5))
        self.assertTrue(environment.observation_space.contains(observation))

        evaluated = set(reset_info["initial_indices"].tolist())
        for _ in range(3):
            observation, _, terminated, truncated, info = environment.step(0)
            self.assertEqual(info["selected_index"], info["shortlist_indices"][0])
            self.assertNotIn(info["selected_index"], evaluated)
            evaluated.add(info["selected_index"])
            self.assertFalse(truncated)
            self.assertTrue(environment.observation_space.contains(observation))
        self.assertTrue(terminated)


if __name__ == "__main__":
    unittest.main()