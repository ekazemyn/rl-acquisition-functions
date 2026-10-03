# Learning Acquisition Functions with Reinforcement Learning

Use reinforcement learning to choose evaluations in pool-based Bayesian optimization, then compare it with standard acquisition strategies on datasets held out by domain.

## Project layout

```text
notebooks/results.ipynb   Results tables, regret curves, and interpretation
src/rlbo/                 Reusable experiment code
  environment.py          Pool-based Gymnasium environment
  datasets.py             Dataset loading and reproducible pool sampling
  surrogate.py            Gaussian-process predictions and EI state features
  baselines.py            Shared-pool baseline episode runner
  ppo.py                  PPO policy trainer and rollout evaluator
tests/                    Automated checks for experiment components
pyproject.toml            Package metadata and runtime dependencies
```

The notebook is intentionally a reporting surface. Put reusable implementation in `src/rlbo/` and focused behavior checks in `tests/`.

## Setup and tests

From the project root in PowerShell:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Current status

The pool environment, reproducible dataset preparation, EI-ranked GP state features, shared-pool baseline episode runner, and PPO training wrapper are implemented. Install the project with `python -m pip install -e ".[data]"` to include the dataset loaders and the PPO runtime. The planned training datasets are Concrete Compressive Strength and Superconductivity; California Housing is reserved for held-out evaluation. The PPO runner trains on the same shared-pool protocol used by the handcrafted baselines and returns a deterministic rollout summary for reporting.