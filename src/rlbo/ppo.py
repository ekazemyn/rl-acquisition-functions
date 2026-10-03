"""PPO-based policy for pool-based Bayesian optimization."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from .datasets import load_dataset, sample_pool
from .environment import PoolOptimizationEnv
from .surrogate import CANDIDATE_STATE_FEATURES, EICandidateProvider


@dataclass(frozen=True)
class PPOEpisodeResult:
    """The deterministic PPO rollout from a shared initial pool state."""

    method: str
    seed: int
    initial_indices: np.ndarray
    selected_indices: np.ndarray
    best_values: np.ndarray
    simple_regret: np.ndarray


class FlattenPoolObservation(gym.ObservationWrapper):
    """Convert the dict observation into a flat vector for SB3 MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        candidate_shape = env.observation_space["candidates"].shape
        flattened_size = int(np.prod(candidate_shape)) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flattened_size,),
            dtype=np.float32,
        )

    def observation(self, observation):
        candidate_part = np.asarray(observation["candidates"], dtype=np.float32).reshape(-1)
        budget_fraction = np.asarray(observation["budget_fraction"], dtype=np.float32).reshape(-1)
        return np.concatenate([candidate_part, budget_fraction], axis=0).astype(np.float32)


def _make_training_env(
    features,
    targets,
    initial_indices,
    budget,
    shortlist_size,
    candidate_provider,
    observation_size,
    seed,
):
    env = PoolOptimizationEnv(
        features=features,
        labels=targets,
        candidate_provider=candidate_provider,
        observation_size=observation_size,
        initial_points=len(initial_indices),
        budget=budget,
        shortlist_size=shortlist_size,
    )
    env.reset(seed=seed, options={"initial_indices": initial_indices.copy()})
    return FlattenPoolObservation(env)


def _evaluate_ppo_rollout(model, env, initial_indices, budget):
    observation, _ = env.reset(options={"initial_indices": initial_indices.copy()})
    underlying = env.unwrapped
    selected_indices = []
    best_values = [float(np.max(underlying._observed_targets))]
    pool_optimum = float(np.max(underlying._labels))

    for _ in range(budget):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, _, info = env.step(int(action))
        selected_indices.append(int(info["selected_index"]))
        best_values.append(float(underlying._best_target))
        if terminated:
            break

    best_values = np.asarray(best_values, dtype=np.float64)
    simple_regret = np.maximum(pool_optimum - best_values, 0.0)
    return np.asarray(selected_indices, dtype=np.int64), best_values, simple_regret


def train_ppo_on_dataset(
    dataset_name,
    pool_size=500,
    initial_points=3,
    budget=30,
    seed=0,
    total_timesteps=4096,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    shortlist_size=20,
    candidate_provider=None,
    observation_size=None,
    policy_kwargs=None,
):
    """Train PPO on one dataset using a sampled pool and a seed-matched initial state."""
    dataset = load_dataset(dataset_name)
    pool, _ = sample_pool(dataset, pool_size=pool_size, seed=seed)
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(len(pool.targets), size=initial_points, replace=False)
    return train_ppo_agent(
        features=pool.features,
        targets=pool.targets,
        initial_indices=initial_indices,
        budget=budget,
        seed=seed,
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        shortlist_size=shortlist_size,
        candidate_provider=candidate_provider,
        observation_size=observation_size,
        policy_kwargs=policy_kwargs,
    )


def train_ppo_on_datasets(
    dataset_names=("concrete", "superconductivity"),
    seeds=(0,),
    **kwargs,
):
    """Train PPO for one or more datasets and seeds, returning a dictionary of results."""
    results = {}
    for dataset_name in dataset_names:
        for seed in seeds:
            result = train_ppo_on_dataset(dataset_name=dataset_name, seed=seed, **kwargs)
            results[(dataset_name, int(seed))] = result
    return results


def train_ppo_agent(
    features,
    targets,
    initial_indices=None,
    budget=30,
    seed=0,
    total_timesteps=4096,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    shortlist_size=20,
    candidate_provider=None,
    observation_size=None,
    policy_kwargs=None,
):
    """Train a PPO policy on the pool-based BO environment and return a rollout result."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be 2D and aligned with targets")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if budget < 1:
        raise ValueError("budget must be positive")
    if shortlist_size < 1 or shortlist_size > len(targets):
        raise ValueError("shortlist_size must be positive and no larger than the pool")
    if initial_indices is None:
        initial_indices = np.random.default_rng(seed).choice(
            len(targets), size=min(3, len(targets)), replace=False
        )
    initial_indices = np.asarray(initial_indices, dtype=np.int64)
    if initial_indices.ndim != 1 or len(initial_indices) == 0:
        raise ValueError("initial_indices must be a non-empty one-dimensional array")
    if len(np.unique(initial_indices)) != len(initial_indices):
        raise ValueError("initial_indices must be distinct")
    if np.any(initial_indices < 0) or np.any(initial_indices >= len(targets)):
        raise ValueError("initial_indices must be valid pool indices")
    if len(targets) < len(initial_indices) + budget:
        raise ValueError("the pool is too small for the requested budget")

    if candidate_provider is None:
        candidate_provider = EICandidateProvider()
    if observation_size is None:
        observation_size = len(CANDIDATE_STATE_FEATURES)

    env = _make_training_env(
        features=features,
        targets=targets,
        initial_indices=initial_indices,
        budget=budget,
        shortlist_size=shortlist_size,
        candidate_provider=candidate_provider,
        observation_size=observation_size,
        seed=seed,
    )

    policy_kwargs = dict(policy_kwargs or {})
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=min(batch_size, n_steps),
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)

    eval_env = _make_training_env(
        features=features,
        targets=targets,
        initial_indices=initial_indices,
        budget=budget,
        shortlist_size=shortlist_size,
        candidate_provider=candidate_provider,
        observation_size=observation_size,
        seed=seed,
    )
    selected_indices, best_values, simple_regret = _evaluate_ppo_rollout(
        model=model,
        env=eval_env,
        initial_indices=initial_indices,
        budget=budget,
    )

    return PPOEpisodeResult(
        method="ppo",
        seed=int(seed),
        initial_indices=initial_indices.copy(),
        selected_indices=selected_indices,
        best_values=best_values,
        simple_regret=simple_regret,
    )
