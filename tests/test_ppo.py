import unittest

import numpy as np

from rlbo.ppo import PPOEpisodeResult, train_ppo_agent


class PPOTrainingTests(unittest.TestCase):
    def setUp(self):
        self.features = np.column_stack(
            [
                np.linspace(-2.0, 2.0, 40),
                np.cos(np.linspace(-2.0, 2.0, 40)),
                np.sin(np.linspace(-2.0, 2.0, 40)),
            ]
        )
        self.targets = np.square(self.features[:, 0]) + 0.5 * self.features[:, 1]
        self.initial_indices = np.array([3, 15, 27])
        self.budget = 4

    def test_ppo_training_smoke_test(self):
        result = train_ppo_agent(
            features=self.features,
            targets=self.targets,
            initial_indices=self.initial_indices,
            budget=self.budget,
            seed=11,
            total_timesteps=128,
            n_steps=16,
            batch_size=8,
            learning_rate=3e-3,
        )

        self.assertIsInstance(result, PPOEpisodeResult)
        self.assertEqual(len(result.selected_indices), self.budget)
        self.assertEqual(len(result.best_values), self.budget + 1)
        self.assertEqual(len(result.simple_regret), self.budget + 1)
        self.assertTrue(np.all(np.diff(result.best_values) >= -1e-12))
        self.assertTrue(np.all(np.diff(result.simple_regret) <= 1e-12))
        self.assertEqual(result.seed, 11)

    def test_ppo_uses_shared_initial_points(self):
        result = train_ppo_agent(
            features=self.features,
            targets=self.targets,
            initial_indices=self.initial_indices,
            budget=self.budget,
            seed=7,
            total_timesteps=96,
            n_steps=12,
            batch_size=8,
            learning_rate=3e-3,
        )

        np.testing.assert_array_equal(result.initial_indices, self.initial_indices)
        self.assertFalse(np.isin(result.selected_indices, self.initial_indices).any())


if __name__ == "__main__":
    unittest.main()
