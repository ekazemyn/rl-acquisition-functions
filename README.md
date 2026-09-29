# Learning Acquisition Functions with Reinforcement Learning

Use reinforcement learning to choose evaluations in pool-based Bayesian optimization, then compare it with standard acquisition strategies on datasets held out by domain.

## Project layout

```text
notebooks/results.ipynb   Results tables, regret curves, and interpretation
src/rlbo/                 Reusable experiment code
  environment.py          Pool-based Gymnasium environment
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

The pool environment and its synthetic contract tests are implemented. Dataset loading, surrogate and acquisition strategies, PPO training, and benchmark results are still to come. The planned training datasets are Concrete Compressive Strength and Superconductivity; California Housing is reserved for held-out evaluation.