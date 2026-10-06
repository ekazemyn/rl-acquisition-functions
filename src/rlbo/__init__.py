"""Reinforcement-learning tools for pool-based Bayesian optimization."""

from .environment import PoolOptimizationEnv
from .datasets import TabularDataset, load_dataset, sample_pool, sample_pool_indices
from .baselines import BASELINE_METHODS, BaselineEpisodeResult, run_baseline_episode
from .surrogate import (
    CANDIDATE_STATE_FEATURES,
    EICandidateProvider,
    GaussianProcessSurrogate,
    expected_improvement,
)
from .ppo import (
    PPOEpisodeResult,
    train_ppo_agent,
    train_ppo_on_dataset,
    train_ppo_on_datasets,
)
from .benchmark import run_shared_pool_benchmark

__all__ = [
    "PoolOptimizationEnv",
    "TabularDataset",
    "load_dataset",
    "sample_pool",
    "sample_pool_indices",
    "CANDIDATE_STATE_FEATURES",
    "EICandidateProvider",
    "GaussianProcessSurrogate",
    "expected_improvement",
    "BASELINE_METHODS",
    "BaselineEpisodeResult",
    "run_baseline_episode",
    "PPOEpisodeResult",
    "train_ppo_agent",
    "train_ppo_on_dataset",
    "train_ppo_on_datasets",
    "run_shared_pool_benchmark",
]