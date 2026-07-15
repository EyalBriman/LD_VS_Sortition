# Liquid Democracy vs. Sortition — Simulation Project

This repository contains a coherent, reproducible implementation of three simulation families comparing Liquid Democracy (LD) with Sortition:

1. **Singleton-conditioned loss-bound check** on a one-dimensional clique.
2. **Experiment 1: population geometry and amplification** on a one-dimensional clique, including a numerical verification of the singleton variance-gap identity.
3. **Experiment 2: community structure and variable committee size** on two-dimensional stochastic block model (SBM) graphs.

The project consolidates the supplied scripts into shared modules, removes duplicated implementations, adds a command-line interface, saves every run configuration, and includes automated tests.

---

## 1. Model and conventions

### Population positions

There are `n` agents with positions

\[
x_1,\ldots,x_n \in [0,1]^d.
\]

Random instances in the loss-bound experiment use i.i.d. uniform positions on `[0,1]`. The geometry experiments use a deterministic grid transformed by the symmetric gamma warp

\[
T_\gamma(u)
= \frac12 + \frac12\operatorname{sign}(2u-1)|2u-1|^\gamma.
\]

Interpretation:

- `gamma < 1`: points move toward the boundary, producing an edge-dense population;
- `gamma = 1`: the original grid;
- `gamma > 1`: points move toward the center, producing a center-dense population.

### Delegation

For a clique, agent `i` delegates to another agent `j` with probability proportional to inverse distance:

\[
w_{ij}=\frac{1}{\lVert x_i-x_j\rVert_1+\varepsilon},
\qquad
\Pr(i\to j)=\frac{w_{ij}}{\sum_{\ell\neq i}w_{i\ell}}.
\]

Self-delegation is excluded.

For Experiment 2, an undirected SBM graph is sampled first. Delegation is then restricted to graph neighbors and remains proportional to inverse distance. An isolated vertex falls back to the full inverse-distance clique.

### Singleton selection and conditioning on `M = 1`

For a realized delegation graph, let `indeg_i` be agent `i`'s indegree and define

\[
p_i=\frac{\operatorname{indeg}_i}{n}.
\]

Agents independently self-select with probabilities `p_i`. Let `M` denote the number of selected representatives. For a fixed delegation graph,

\[
q^{\mathrm{raw}}_i
=\Pr(\mathrm{Com}=\{i\}\mid \mathrm{DelG})
=p_i\prod_{\ell\neq i}(1-p_\ell),
\]

and

\[
\Pr(M=1\mid \mathrm{DelG})=\sum_i q^{\mathrm{raw}}_i.
\]

The globally conditioned singleton distribution is therefore

\[
q_i^{(1)}
=\Pr(\mathrm{Com}=\{i\}\mid M=1)
=\frac{\mathbb{E}_{\mathrm{DelG}}[q_i^{\mathrm{raw}}]}
       {\mathbb{E}_{\mathrm{DelG}}[\Pr(M=1\mid\mathrm{DelG})]}.
\]

The loss-bound code estimates this ratio directly. It does **not** average graph-by-graph conditional distributions, because that would weight every graph equally instead of weighting it by its probability of producing a singleton committee.

### Variable committee size for `k > 1`

For Experiment 2, `k=1` uses the exact `M=1` conditioning convention above. For `k>1`, the Bernoulli probabilities are calibrated as

\[
p_i(\lambda)
=\min\!\left\{
\lambda\frac{\zeta_i^\alpha}{\sum_j\zeta_j^\alpha},1
\right\},
\]

where `zeta_i` is based on indegree and the implementation uses `alpha=1`. A binary search chooses `lambda` so that

\[
\mathbb{E}[M\mid M>0]=k.
\]

Empty LD committees are rejected. Sortition always samples exactly `k` agents without replacement.

### Aggregation and error

The repository supports coordinate-wise weighted means and weighted medians. The direct-democracy outcome is the same aggregation rule applied to all agents with equal weight.

The normalized MSE gap is

\[
\delta
=\frac{\lVert \mathrm{Out}_{\mathrm{LD}}-\mu\rVert_2^2
      -\lVert \mathrm{Out}_{\mathrm{Sort}}-\mu\rVert_2^2}{S^2},
\]

where

\[
S^2=\frac1n\sum_i\lVert x_i-\mu\rVert_2^2.
\]

Thus:

- `delta < 0`: LD has lower squared error;
- `delta > 0`: Sortition has lower squared error.

---

## 2. Experiments

### A. Singleton loss-bound check

The experiment samples random one-dimensional populations and estimates

\[
B^{(1)}=\mathbb{E}[\mathrm{Out}_{\mathrm{LD}}\mid M=1]-\mu,
\qquad
L^{(1)}=|B^{(1)}|.
\]

It checks empirically whether any sampled instance reaches or exceeds `1/2`.

Full-profile parameters reproduce the supplied specification:

- `n in {5,10,20,50,100}`;
- 500 random instances per `n`;
- 500 delegation graphs per instance;
- seed 2025.

Main outputs:

- `loss_bound_results.csv`;
- `loss_bound_summary.csv`;
- `loss_bound_violations.csv`;
- histogram, scatter, and maximum-loss figures.

### B. Experiment 1: clique geometry

This experiment varies `gamma` and `n` in one dimension. It reports normalized LD-minus-Sortition MSE gaps for mean and median reference outcomes.

It also verifies the singleton variance-gap identity

\[
\Delta_{\mathrm{Var}}^{(1)}
=\sum_i\left(q_i^{(1)}-\frac1n\right)(x_i-\mu)^2
\]

against an independent Monte Carlo estimate.

Because the realized committee has one member, the LD and Sortition committee outcomes are individual positions. The mean/median choice changes the direct-democracy reference point and the normalization.

Main outputs:

- `variance_identity.csv` and `variance_identity_M1.png`;
- `exp1_mean_M1.csv`;
- `exp1_median_M1.csv`;
- `exp1_diagnostics.csv`;
- `sampling_vs_amplification_M1.png`;
- `mse_gap_lines_exp1_M1.png`.

### C. Experiment 2: two-dimensional SBM and variable `k`

This experiment uses two graph regimes:

- strong community structure: `B=2`, `p_in=0.30`, `p_out=0.05`;
- weak community structure: `B=2`, `p_in=0.15`, `p_out=0.10`.

It varies population geometry, population size, and target committee size.

The LD outcome follows delegation chains to selected representatives. A delegation chain can enter a directed cycle that contains no representative. To preserve the supplied simulation convention while making it transparent:

- such agents are counted as unresolved;
- they contribute no basin weight;
- if no agent resolves to any representative, the selected representatives receive equal fallback weights.

The CSV output records `unresolved_agents` and `conditioning_attempts`, and the summary reports how often unresolved cycles occur. This diagnostic should be inspected before interpreting Experiment 2 substantively.

Main outputs:

- `exp2_mean.csv`;
- `exp2_median.csv`;
- `exp2_summary.csv`;
- line plots for every `(n, SBM)` pair;
- one compact summary figure.

---

## 3. Repository structure

```text
LD_VS_Sortition/
├── pyproject.toml
├── requirements.txt
├── README.md
├── CHANGELOG_NOTES.md
├── scripts/
│   ├── run_quick.sh
│   ├── run_full.sh
│   └── run_quick.bat
├── src/ld_vs_sortition/
│   ├── __main__.py
│   ├── cli.py
│   ├── core.py
│   ├── delegation.py
│   ├── selection.py
│   └── experiments/
│       ├── loss_bound.py
│       ├── experiment1.py
│       └── experiment2.py
├── tests/
└── outputs/
```

Generated results are ignored by Git, except for `outputs/.gitkeep`.

---

## 4. Installation on Windows using Git Bash

Assuming the extracted folder is

```text
C:\Users\User\Downloads\LD_VS_Sortition
```

open **Git Bash** and run:

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

The project was validated with all nine included tests passing.

---

## 5. Running the simulations

### Quick validation run

This runs reduced parameter grids for all three experiment families:

```bash
python -m ld_vs_sortition all \
  --profile quick \
  --output-dir outputs/quick
```

Equivalent script:

```bash
bash scripts/run_quick.sh
```

### Full supplied parameter grids

```bash
python -m ld_vs_sortition all \
  --profile full \
  --output-dir outputs/full
```

Equivalent script:

```bash
bash scripts/run_full.sh
```

The full Experiment 2 grid is computationally substantial because it samples a new SBM graph, delegation graph, representative set, and Sortition committee for every Monte Carlo draw.

### Run experiments separately

```bash
python -m ld_vs_sortition loss-bound --profile full --output-dir outputs/loss_bound

python -m ld_vs_sortition experiment1 --profile full --output-dir outputs/experiment1

python -m ld_vs_sortition experiment2 --profile full --output-dir outputs/experiment2
```

### Custom loss-bound run

```bash
python -m ld_vs_sortition loss-bound \
  --profile quick \
  --n-list 5,10,20 \
  --n-instances 100 \
  --graphs-per-instance 1000 \
  --batch-size 250 \
  --seed 2025 \
  --output-dir outputs/loss_custom
```

### Custom Experiment 1 run

```bash
python -m ld_vs_sortition experiment1 \
  --profile quick \
  --n-list 10,50,100 \
  --gamma-list 0.1,0.3,1.0,2.0,3.0 \
  --n-mc 2000 \
  --theorem-n-list 10,20,50 \
  --theorem-gamma-list 0.3,1.0,2.0 \
  --theorem-graphs 50000 \
  --theorem-mc 50000 \
  --q-graphs 20000 \
  --seed 123 \
  --output-dir outputs/experiment1_custom
```

### Custom Experiment 2 run

```bash
python -m ld_vs_sortition experiment2 \
  --profile quick \
  --n-list 50,100 \
  --gamma-list 0.3,1.0,2.0 \
  --k-list 1,5,10 \
  --n-mc 500 \
  --seed 456 \
  --output-dir outputs/experiment2_custom
```

Every output directory contains a `config.json` recording the exact parameters used.

---

## 6. Uploading this project to GitHub

The safest workflow is to clone the existing repository, copy this project into the clone, and then commit. This preserves any existing GitHub history.

From Git Bash:

```bash
cd /c/Users/User/Downloads

# This only removes the temporary upload clone if it already exists.
rm -rf LD_VS_Sortition_repo_upload

git clone https://github.com/EyalBriman/LD_VS_Sortition.git \
  LD_VS_Sortition_repo_upload

# Replace the checked-out repository contents while preserving its .git history.
find LD_VS_Sortition_repo_upload -mindepth 1 -maxdepth 1 \
  ! -name .git -exec rm -rf {} +
cp -a LD_VS_Sortition/. LD_VS_Sortition_repo_upload/

cd LD_VS_Sortition_repo_upload

git status
git add -A
git commit -m "Add coherent LD vs Sortition simulation project"
git push origin HEAD
```

If GitHub requests authentication, complete the browser sign-in or use your configured personal access token. Do not place a token inside a script or commit it to the repository.

For later updates, work inside `LD_VS_Sortition_repo_upload` and use:

```bash
git add -A
git commit -m "Describe the simulation update"
git push origin HEAD
```

---

## 7. Reproducibility notes

- NumPy's `Generator` is used throughout.
- Each command accepts `--seed`.
- The full profiles preserve the supplied top-level parameter grids.
- CSV files contain draw-level data rather than only plotted summaries.
- Each run saves its configuration in JSON.
- Figures are saved without requiring an interactive display.
- The tests cover grid construction, aggregation, delegation sampling, indegrees, singleton conditioning, conditional-size calibration, and smoke runs for all experiments.
