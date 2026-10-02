"""Common-seed baseline evaluation for pool-based Bayesian optimization."""

from dataclasses import dataclass

import numpy as np

from .surrogate import GaussianProcessSurrogate, expected_improvement


BASELINE_METHODS = ("random", "greedy_mean", "ei", "ucb", "thompson")


@dataclass(frozen=True)
class BaselineEpisodeResult:
    method: str
    seed: int
    initial_indices: np.ndarray
    selected_indices: np.ndarray
    best_values: np.ndarray
    simple_regret: np.ndarray


def _sample_joint_posterior(mean, covariance, rng):
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    noise = eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(len(mean))
    )
    return mean + noise


def run_baseline_episode(
    features,
    targets,
    method,
    initial_indices,
    budget=30,
    seed=0,
    ucb_beta=2.0,
    ei_xi=0.01,
):
    """Run one baseline episode and return best-so-far and simple-regret curves.

    `initial_indices` should be shared across methods in paired comparisons.
    Policies only use targets at those indices and at points they subsequently select.
    """
    if method not in BASELINE_METHODS:
        raise ValueError(f"method must be one of {BASELINE_METHODS}")
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    initial_indices = np.asarray(initial_indices, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("features must be 2D and row-aligned with targets")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if initial_indices.ndim != 1 or len(initial_indices) == 0:
        raise ValueError("initial_indices must be a non-empty one-dimensional array")
    if (
        len(np.unique(initial_indices)) != len(initial_indices)
        or np.any(initial_indices < 0)
        or np.any(initial_indices >= len(targets))
    ):
        raise ValueError("initial_indices must be distinct valid pool indices")
    if budget < 1 or len(targets) < len(initial_indices) + budget:
        raise ValueError("pool must contain enough candidates for the requested budget")
    if not np.isfinite(ucb_beta) or ucb_beta < 0:
        raise ValueError("ucb_beta must be finite and non-negative")
    if not np.isfinite(ei_xi) or ei_xi < 0:
        raise ValueError("ei_xi must be finite and non-negative")

    rng = np.random.default_rng(seed)
    evaluated_indices = initial_indices.copy()
    observed_targets = targets[evaluated_indices].copy()
    selected_indices = []
    best_values = [float(np.max(observed_targets))]
    pool_optimum = float(np.max(targets))

    for _ in range(budget):
        available = np.setdiff1d(
            np.arange(len(targets)), evaluated_indices, assume_unique=False
        )
        candidate_features = features[available]

        if method == "random":
            selected_index = int(rng.choice(available))
        else:
            surrogate = GaussianProcessSurrogate().fit(
                features[evaluated_indices], observed_targets
            )
            if method == "thompson":
                mean, _, covariance = surrogate.predict_distribution(candidate_features)
                scores = _sample_joint_posterior(mean, covariance, rng)
            else:
                mean, standard_deviation = surrogate.predict(candidate_features)
                if method == "greedy_mean":
                    scores = mean
                elif method == "ei":
                    incumbent = (
                        float(np.max(observed_targets)) - surrogate.target_center
                    ) / surrogate.target_scale
                    scores = expected_improvement(
                        mean, standard_deviation, incumbent, xi=ei_xi
                    )
                else:
                    scores = mean + ucb_beta * standard_deviation
            selected_index = int(available[int(np.argmax(scores))])

        selected_indices.append(selected_index)
        evaluated_indices = np.append(evaluated_indices, selected_index)
        observed_targets = np.append(observed_targets, targets[selected_index])
        best_values.append(float(np.max(observed_targets)))

    best_values = np.asarray(best_values, dtype=np.float64)
    simple_regret = np.maximum(pool_optimum - best_values, 0.0)
    return BaselineEpisodeResult(
        method=method,
        seed=int(seed),
        initial_indices=initial_indices.copy(),
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        best_values=best_values,
        simple_regret=simple_regret,
    )