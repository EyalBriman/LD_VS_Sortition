# Liquid Democracy vs. Sortition

Reproducible simulations comparing Liquid Democracy (LD) with Sortition.

The repository contains three experiment families:

1. **Singleton loss bound:** tests whether the empirical singleton-conditioned loss reaches `1/2` on one-dimensional clique instances.
2. **Experiment 1:** studies how one-dimensional population geometry affects LD and Sortition, and numerically checks the singleton variance-gap identity.
3. **Experiment 2:** compares the mechanisms on two-dimensional stochastic block model (SBM) graphs for several target committee sizes.

## Main conventions

- Delegation probabilities are proportional to inverse distance.
- Self-delegation is excluded.
- For `k = 1`, LD is evaluated conditional on obtaining exactly one representative (`M = 1`).
- For `k > 1`, selection probabilities are calibrated so that `E[M | M > 0] = k`; empty committees are rejected.
- Sortition selects exactly `k` representatives uniformly without replacement.
- Negative normalized MSE gaps favor LD; positive gaps favor Sortition.

## Installation

Python 3.10 or newer is required.

```bash
cd /c/Users/User/Downloads/LD_VS_Sortition

python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

## Running the simulations

Run a small validation profile:

```bash
python -m ld_vs_sortition all \
  --profile quick \
  --output-dir outputs/quick
```

Run the full parameter grids:

```bash
python -m ld_vs_sortition all \
  --profile full \
  --output-dir outputs/full
```

Run one experiment only:

```bash
python -m ld_vs_sortition loss-bound \
  --profile full \
  --output-dir outputs/loss_bound

python -m ld_vs_sortition experiment1 \
  --profile full \
  --output-dir outputs/experiment1

python -m ld_vs_sortition experiment2 \
  --profile full \
  --output-dir outputs/experiment2
```

The full Experiment 2 run is computationally intensive.

## Outputs

Each experiment writes its results to the selected output directory, including:

- draw-level CSV files;
- summary CSV files;
- generated figures;
- `config.json` with the exact run parameters.

Generated output files are ignored by Git, except for `outputs/.gitkeep`.

## Repository structure

```text
LD_VS_Sortition/
├── pyproject.toml
├── requirements.txt
├── scripts/
├── src/ld_vs_sortition/
│   ├── core.py
│   ├── delegation.py
│   ├── selection.py
│   └── experiments/
├── tests/
└── outputs/
```

## Reproducibility

- All commands accept `--seed`.
- NumPy's `Generator` is used throughout.
- The `quick` profile is intended for validation.
- The `full` profile preserves the supplied experiment grids.
- Experiment 2 records unresolved delegation-cycle diagnostics in its CSV output.
