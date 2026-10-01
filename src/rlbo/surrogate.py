"""Gaussian-process predictions and EI-ranked candidate state features."""

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler


CANDIDATE_STATE_FEATURES = (
    "mean_standardized",
    "std_standardized",
    "ei_standardized",
    "gap_to_best_standardized",
    "nearest_distance_standardized",
)


class GaussianProcessSurrogate:
    """Small fixed-kernel GP, fit only to labels revealed in the episode."""

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.model = None
        self.target_center = None
        self.target_scale = None

    def fit(self, features, targets):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets) or len(targets) == 0:
            raise ValueError("features and targets must be non-empty and row-aligned")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("training features and targets must be finite")

        scaled_features = self.feature_scaler.fit_transform(features)
        self.target_center = float(np.mean(targets))
        target_scale = float(np.std(targets))
        self.target_scale = target_scale if target_scale > np.finfo(float).eps else 1.0
        scaled_targets = (targets - self.target_center) / self.target_scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0, length_scale_bounds="fixed", nu=2.5
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None,
            normalize_y=False,
        )
        self.model.fit(scaled_features, scaled_targets)
        return self

    def transform_features(self, features):
        if self.model is None:
            raise RuntimeError("fit() must be called before transforming features")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.feature_scaler.n_features_in_:
            raise ValueError("features have an unexpected shape")
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")
        return self.feature_scaler.transform(features)

    def predict(self, features):
        """Return predictive mean and standard deviation in standardized target units."""
        scaled_features = self.transform_features(features)
        mean, standard_deviation = self.model.predict(
            scaled_features, return_std=True
        )
        return mean, np.maximum(standard_deviation, 0.0)


def expected_improvement(mean, standard_deviation, incumbent, xi=0.01):
    """Expected improvement for maximization, with inputs in standardized units."""
    mean = np.asarray(mean, dtype=np.float64)
    standard_deviation = np.asarray(standard_deviation, dtype=np.float64)
    if mean.shape != standard_deviation.shape:
        raise ValueError("mean and standard_deviation must have matching shapes")
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise ValueError("mean and standard_deviation must be finite")
    if (
        np.any(standard_deviation < 0)
        or not np.isfinite(incumbent)
        or not np.isfinite(xi)
        or xi < 0
    ):
        raise ValueError("standard deviations and xi must be non-negative")

    improvement = mean - incumbent - xi
    positive_uncertainty = standard_deviation > 0.0
    denominator = np.maximum(standard_deviation, np.abs(improvement) / 38.0)
    z_score = np.divide(
        improvement,
        denominator,
        out=np.zeros_like(improvement),
        where=positive_uncertainty,
    )
    density = np.exp(-0.5 * np.square(z_score)) / np.sqrt(2.0 * np.pi)
    values = improvement * ndtr(z_score) + standard_deviation * density
    values = np.where(positive_uncertainty, values, np.maximum(improvement, 0.0))
    return np.maximum(values, 0.0)


class EICandidateProvider:
    """Build the five candidate-state features and rank candidates by EI."""

    def __init__(self, xi=0.01):
        if xi < 0:
            raise ValueError("xi must be non-negative")
        self.xi = float(xi)
        self.surrogate = GaussianProcessSurrogate()

    def __call__(
        self,
        evaluated_indices,
        observed_targets,
        available_indices,
        candidate_features,
        evaluated_features,
    ):
        evaluated_indices = np.asarray(evaluated_indices)
        observed_targets = np.asarray(observed_targets, dtype=np.float64).reshape(-1)
        available_indices = np.asarray(available_indices)
        candidate_features = np.asarray(candidate_features, dtype=np.float64)
        evaluated_features = np.asarray(evaluated_features, dtype=np.float64)
        if len(evaluated_indices) != len(observed_targets):
            raise ValueError("each evaluated index must have one revealed target")
        if len(available_indices) != len(candidate_features):
            raise ValueError("each available index must have one feature row")
        if len(evaluated_indices) != len(evaluated_features):
            raise ValueError("each evaluated index must have one feature row")
        if candidate_features.shape[1] != evaluated_features.shape[1]:
            raise ValueError("candidate and evaluated features must have matching columns")

        self.surrogate.fit(evaluated_features, observed_targets)
        mean, standard_deviation = self.surrogate.predict(candidate_features)
        incumbent = (
            float(np.max(observed_targets)) - self.surrogate.target_center
        ) / self.surrogate.target_scale
        improvement = expected_improvement(
            mean, standard_deviation, incumbent, xi=self.xi
        )

        scaled_candidates = self.surrogate.transform_features(candidate_features)
        scaled_evaluated = self.surrogate.transform_features(evaluated_features)
        distances = np.sqrt(
            np.sum(
                np.square(scaled_candidates[:, None, :] - scaled_evaluated[None, :, :]),
                axis=2,
            )
        ).min(axis=1) / np.sqrt(candidate_features.shape[1])
        candidate_state = np.column_stack(
            [mean, standard_deviation, improvement, mean - incumbent, distances]
        )
        return candidate_state, improvement