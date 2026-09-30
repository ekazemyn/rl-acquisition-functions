"""Loading and reproducible pool sampling for the project datasets."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabularDataset:
    name: str
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple
    target_name: str
    source: str


def _to_tabular_dataset(name, features, targets, source):
    feature_values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if feature_values.ndim != 2:
        raise ValueError("features must be a two-dimensional table")
    if target_values.ndim == 2 and target_values.shape[1] == 1:
        target_values = target_values[:, 0]
    if target_values.ndim != 1 or len(feature_values) != len(target_values):
        raise ValueError("targets must contain one value for each feature row")

    valid_rows = np.isfinite(feature_values).all(axis=1) & np.isfinite(target_values)
    feature_values = feature_values[valid_rows]
    target_values = target_values[valid_rows]
    if len(target_values) == 0:
        raise ValueError("dataset has no rows with finite features and target")

    columns = getattr(features, "columns", None)
    feature_names = tuple(str(column) for column in columns) if columns is not None else tuple(
        f"feature_{index}" for index in range(feature_values.shape[1])
    )
    target_columns = getattr(targets, "columns", None)
    target_name = str(target_columns[0]) if target_columns is not None else "target"
    return TabularDataset(
        name=name,
        features=feature_values,
        targets=target_values,
        feature_names=feature_names,
        target_name=target_name,
        source=source,
    )


def load_dataset(name, data_home=None):
    """Load one approved dataset; optional data dependencies are imported lazily."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"concrete", "concrete_strength", "concrete_compressive_strength"}:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "Loading UCI datasets requires the 'data' extra: pip install -e .[data]"
            ) from exc
        dataset = fetch_ucirepo(id=165)
        return _to_tabular_dataset(
            "concrete",
            dataset.data.features,
            dataset.data.targets,
            "UCI Concrete Compressive Strength (dataset 165)",
        )

    if key in {"superconductivity", "superconductor"}:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "Loading UCI datasets requires the 'data' extra: pip install -e .[data]"
            ) from exc
        dataset = fetch_ucirepo(id=464)
        return _to_tabular_dataset(
            "superconductivity",
            dataset.data.features,
            dataset.data.targets,
            "UCI Superconductivity (dataset 464)",
        )

    if key in {"california", "california_housing"}:
        try:
            from sklearn.datasets import fetch_california_housing
        except ImportError as exc:
            raise ImportError(
                "Loading California Housing requires the 'data' extra: pip install -e .[data]"
            ) from exc
        housing = fetch_california_housing(data_home=data_home, as_frame=True)
        return _to_tabular_dataset(
            "california_housing",
            housing.data,
            housing.target.to_frame(name=housing.target.name or "MedHouseVal"),
            "scikit-learn California Housing dataset",
        )

    raise ValueError(
        "unknown dataset; choose 'concrete', 'superconductivity', or 'california_housing'"
    )


def sample_pool_indices(n_rows, pool_size=500, seed=0):
    """Choose a reproducible pool without replacement from a dataset's rows."""
    if n_rows < 1 or pool_size < 1:
        raise ValueError("n_rows and pool_size must be positive")
    if pool_size > n_rows:
        raise ValueError("pool_size cannot exceed the number of dataset rows")
    return np.random.default_rng(seed).choice(n_rows, size=pool_size, replace=False)


def sample_pool(dataset, pool_size=500, seed=0):
    """Return a sampled dataset pool and its row indices in the source dataset."""
    indices = sample_pool_indices(len(dataset.targets), pool_size=pool_size, seed=seed)
    pool = TabularDataset(
        name=dataset.name,
        features=dataset.features[indices].copy(),
        targets=dataset.targets[indices].copy(),
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
        source=dataset.source,
    )
    return pool, indices