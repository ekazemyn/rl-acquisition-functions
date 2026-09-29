"""Gymnasium environment for sequential selection from a fixed candidate pool."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PoolOptimizationEnv(gym.Env):
    """Pool-based maximization environment with labels revealed on selection."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features,
        labels,
        candidate_provider,
        observation_size,
        initial_points=3,
        budget=30,
        shortlist_size=20,
        reward_scale=1.0,
    ):
        super().__init__()
        self._features = np.asarray(features, dtype=np.float32)
        self._labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if self._features.ndim != 2 or len(self._features) != len(self._labels):
            raise ValueError("features must be 2D and aligned with labels")
        if not np.isfinite(self._features).all() or not np.isfinite(self._labels).all():
            raise ValueError("features and labels must be finite")
        if initial_points < 1 or budget < 1 or shortlist_size < 1:
            raise ValueError("initial_points, budget, and shortlist_size must be positive")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        minimum_pool_size = initial_points + budget - 1 + shortlist_size
        if len(self._labels) < minimum_pool_size:
            raise ValueError("pool is too small to provide a full shortlist through the budget")

        self.candidate_provider = candidate_provider
        self.initial_points = int(initial_points)
        self.budget = int(budget)
        self.shortlist_size = int(shortlist_size)
        self.reward_scale = float(reward_scale)
        self.observation_size = int(observation_size)
        self.action_space = spaces.Discrete(self.shortlist_size)
        self.observation_space = spaces.Dict(
            {
                "candidates": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.shortlist_size, self.observation_size),
                    dtype=np.float32,
                ),
                "budget_fraction": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        self._evaluated_indices = None
        self._observed_targets = None
        self._candidate_indices = None
        self._steps_taken = 0

    def _get_observation(self):
        evaluated_indices = self._evaluated_indices.copy()
        observed_targets = self._observed_targets.copy()
        available = np.setdiff1d(
            np.arange(len(self._labels)), evaluated_indices, assume_unique=False
        )
        candidate_observations, scores = self.candidate_provider(
            evaluated_indices,
            observed_targets,
            available,
            self._features[available].copy(),
            self._features[evaluated_indices].copy(),
        )
        candidate_observations = np.asarray(candidate_observations, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if candidate_observations.shape != (len(available), self.observation_size):
            raise ValueError("candidate_provider returned an invalid observation shape")
        if len(scores) != len(available) or not np.isfinite(scores).all():
            raise ValueError("candidate_provider returned invalid scores")
        if not np.isfinite(candidate_observations).all():
            raise ValueError("candidate observations must be finite")

        ranked_positions = np.argsort(-scores, kind="stable")[: self.shortlist_size]
        self._candidate_indices = available[ranked_positions]
        return {
            "candidates": candidate_observations[ranked_positions],
            "budget_fraction": np.asarray(
                [(self.budget - self._steps_taken) / self.budget], dtype=np.float32
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "initial_indices" in options:
            initial = np.asarray(options["initial_indices"], dtype=np.int64)
        else:
            initial = self.np_random.choice(
                len(self._labels), size=self.initial_points, replace=False
            )
        if (
            initial.shape != (self.initial_points,)
            or len(np.unique(initial)) != self.initial_points
            or np.any(initial < 0)
            or np.any(initial >= len(self._labels))
        ):
            raise ValueError("initial_indices must be distinct valid pool indices")

        self._evaluated_indices = initial.copy()
        self._observed_targets = self._labels[initial].copy()
        self._steps_taken = 0
        self._best_target = float(np.max(self._observed_targets))
        observation = self._get_observation()
        return observation, {"initial_indices": initial.copy()}

    def step(self, action):
        if self._evaluated_indices is None:
            raise RuntimeError("reset() must be called before step()")
        if self._steps_taken >= self.budget:
            raise RuntimeError("episode is complete; call reset() before stepping again")
        if not self.action_space.contains(action):
            raise ValueError("action must be a valid shortlist rank")

        shortlist_indices = self._candidate_indices.copy()
        selected_index = int(shortlist_indices[int(action)])
        previous_best = self._best_target
        selected_target = float(self._labels[selected_index])
        self._evaluated_indices = np.append(self._evaluated_indices, selected_index)
        self._observed_targets = np.append(self._observed_targets, selected_target)
        self._best_target = max(previous_best, selected_target)
        raw_improvement = self._best_target - previous_best
        reward = raw_improvement / self.reward_scale
        self._steps_taken += 1
        terminated = self._steps_taken == self.budget

        if terminated:
            observation = {
                "candidates": np.zeros(
                    (self.shortlist_size, self.observation_size), dtype=np.float32
                ),
                "budget_fraction": np.zeros(1, dtype=np.float32),
            }
        else:
            observation = self._get_observation()
        info = {
            "selected_index": selected_index,
            "shortlist_indices": shortlist_indices,
            "raw_improvement": raw_improvement,
        }
        return observation, reward, terminated, False, info