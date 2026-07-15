from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ..core import (
    direct_democracy_outcome,
    grid_points_cube_gamma,
    instance_spread_s2,
)
from ..delegation import clique_probability_cdf
from ..selection import (
    estimate_singleton_distribution_analytic,
    sample_singleton_indices_conditioned,
)


def sample_sortition_values(
    points_1d: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return points_1d[rng.integers(0, len(points_1d), size=n_draws)]


def sample_ld_singleton_values(
    points_1d: np.ndarray,
    cdf: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, float]:
    indices, p_m1 = sample_singleton_indices_conditioned(
        cdf, n_draws, rng, batch_size=batch_size
    )
    return points_1d[indices], p_m1


def paired_mse_gap_draws(
    points: np.ndarray,
    center: np.ndarray,
    spread_s2: float,
    cdf: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    points_1d = points[:, 0]
    center_scalar = float(center[0])
    sortition = sample_sortition_values(points_1d, n_mc, rng)
    liquid, p_m1 = sample_ld_singleton_values(points_1d, cdf, n_mc, rng)
    draws = ((liquid - center_scalar) ** 2 - (sortition - center_scalar) ** 2) / spread_s2
    return draws.astype(float), p_m1


def run_geometry_experiment(
    n_list: Iterable[int],
    gamma_list: Iterable[float],
    *,
    n_mc: int,
    base_seed: int,
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(base_seed)
    rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []

    for n in n_list:
        for gamma in gamma_list:
            points = grid_points_cube_gamma(n=int(n), d=1, gamma=float(gamma))
            center = direct_democracy_outcome(points, aggregation)
            spread = instance_spread_s2(points, center)
            cdf = clique_probability_cdf(points)
            draws, p_m1 = paired_mse_gap_draws(
                points, center, spread, cdf, n_mc, rng
            )
            for draw_index, value in enumerate(draws):
                rows.append(
                    {
                        "shape": f"grid_gamma(g={gamma},d=1)",
                        "n": int(n),
                        "d": 1,
                        "k": 1,
                        "gamma": float(gamma),
                        "draw_index": int(draw_index),
                        "normalized_mse_gap": float(value),
                        "aggregation": aggregation,
                    }
                )
            diagnostic_rows.append(
                {
                    "n": int(n),
                    "gamma": float(gamma),
                    "aggregation": aggregation,
                    "estimated_p_m1": float(p_m1),
                    "direct_democracy_outcome": float(center[0]),
                    "spread_s2": float(spread),
                }
            )
            print(
                f"  Experiment 1 | aggregation={aggregation:6s} | n={n:4d} | "
                f"gamma={gamma:g} | P(M=1)≈{p_m1:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows), pd.DataFrame(diagnostic_rows)


def theorem3_rhs(points_1d: np.ndarray, q: np.ndarray) -> float:
    n = len(points_1d)
    mean = float(points_1d.mean())
    return float(np.sum((q - 1.0 / n) * (points_1d - mean) ** 2))


def theorem3_lhs_mc(
    points_1d: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    cdf = clique_probability_cdf(points_1d)
    mean = float(points_1d.mean())
    liquid, p_m1 = sample_ld_singleton_values(points_1d, cdf, n_draws, rng)
    sortition = sample_sortition_values(points_1d, n_draws, rng)
    lhs = float(((liquid - mean) ** 2 - (sortition - mean) ** 2).mean())
    return lhs, p_m1


def verify_variance_identity(
    n_list: Iterable[int],
    gamma_list: Iterable[float],
    *,
    n_graphs: int,
    n_mc: int,
    base_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(base_seed)
    rows: list[dict[str, float | int]] = []

    for n in n_list:
        for gamma in gamma_list:
            points = grid_points_cube_gamma(int(n), 1, float(gamma))[:, 0]
            cdf = clique_probability_cdf(points)
            q, p_m1_analytic = estimate_singleton_distribution_analytic(
                cdf, n_graphs, rng, batch_size=min(1000, n_graphs)
            )
            rhs = theorem3_rhs(points, q)
            lhs, p_m1_rejection = theorem3_lhs_mc(points, n_mc, rng)
            rows.append(
                {
                    "n": int(n),
                    "gamma": float(gamma),
                    "rhs_q1": rhs,
                    "lhs_mc": lhs,
                    "absolute_difference": abs(rhs - lhs),
                    "p_m1_analytic": p_m1_analytic,
                    "p_m1_rejection": p_m1_rejection,
                }
            )
            print(
                f"  Identity | n={n:4d} | gamma={gamma:g} | "
                f"formula={rhs:+.5f} | MC={lhs:+.5f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def plot_variance_identity(df: pd.DataFrame, path: Path) -> None:
    n_values = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.0))
    if len(n_values) == 1:
        axes = [axes]
    for axis, n in zip(axes, n_values):
        subset = df[df["n"] == n].sort_values("gamma")
        axis.plot(subset["gamma"], subset["rhs_q1"], "o-", label=r"formula using $q_i^{(1)}$")
        axis.plot(subset["gamma"], subset["lhs_mc"], "s--", label="Monte Carlo")
        axis.axhline(0.0, linewidth=0.8, linestyle=":")
        axis.set_xlabel(r"$\gamma$")
        axis.set_ylabel(r"$\Delta_{\mathrm{Var}}^{(1)}$")
        axis.set_title(f"n={n}")
        axis.legend(fontsize=8)
    fig.suptitle(r"Variance-gap identity under conditioning on $M=1$")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_vs_amplification(
    path: Path,
    *,
    n: int,
    gamma_edge: float,
    gamma_center: float,
    q_graphs: int,
    base_seed: int,
    bandwidth: float = 0.07,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    column_titles = [
        "Population",
        "Sortition committee\n(mirrors population)",
        "LD committee\n(conditioned on M=1)",
    ]
    configurations = [
        (gamma_edge, f"Edge-dense (gamma={gamma_edge})"),
        (gamma_center, f"Center-dense (gamma={gamma_center})"),
    ]
    x_grid = np.linspace(-0.05, 1.05, 500)
    rng = np.random.default_rng(base_seed)

    for row, (gamma, row_label) in enumerate(configurations):
        x = np.sort(grid_points_cube_gamma(n, 1, gamma)[:, 0])
        cdf = clique_probability_cdf(x)
        q, _ = estimate_singleton_distribution_analytic(
            cdf, q_graphs, rng, batch_size=min(1000, q_graphs)
        )
        population_kde = gaussian_kde(x, bw_method=bandwidth)
        ld_kde = gaussian_kde(x, bw_method=bandwidth, weights=q)
        population_density = np.maximum(population_kde(x_grid), 0.0)
        ld_density = np.maximum(ld_kde(x_grid), 0.0)
        mask = (x_grid >= 0) & (x_grid <= 1)
        ymax = max(population_density[mask].max(), ld_density[mask].max()) * 1.22

        for column, density in enumerate([population_density, population_density, ld_density]):
            axis = axes[row, column]
            if column > 0:
                axis.fill_between(x_grid[mask], population_density[mask], alpha=0.18)
                axis.plot(x_grid[mask], population_density[mask], linewidth=1.2, linestyle="--", label="Population")
            axis.fill_between(x_grid[mask], density[mask], alpha=0.45)
            axis.plot(x_grid[mask], density[mask], linewidth=2.0, label=["Population", "Sortition", "LD"][column])
            axis.plot(x, np.full(n, -ymax * 0.045), "|", markersize=6, alpha=0.6)
            axis.set_xlim(0, 1)
            axis.set_ylim(-ymax * 0.08, ymax)
            axis.set_xlabel("Position on [0,1]")
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[column], fontweight="bold")
            if column == 0:
                axis.set_ylabel(row_label)
            if column == 2:
                axis.legend(fontsize=8, loc="upper right")

    fig.suptitle("Sortition mirrors the population; singleton-conditioned LD amplifies delegation structure", y=1.01)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mse_gap_lines(
    mean_df: pd.DataFrame,
    median_df: pd.DataFrame,
    path: Path,
) -> None:
    n_values = sorted(mean_df["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, frame, title in zip(axes, [mean_df, median_df], ["Weighted mean", "Weighted median"]):
        for n in n_values:
            subset = frame[frame["n"] == n]
            gamma_plot: list[float] = []
            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for gamma in sorted(subset["gamma"].unique()):
                values = subset.loc[subset["gamma"] == gamma, "normalized_mse_gap"].to_numpy()
                if values.size == 0:
                    continue
                gamma_plot.append(float(gamma))
                means.append(float(values.mean()))
                lower.append(float(np.percentile(values, 25)))
                upper.append(float(np.percentile(values, 75)))
            axis.plot(gamma_plot, means, "o-", label=f"n={n}")
            if n >= 50:
                axis.fill_between(gamma_plot, lower, upper, alpha=0.10)
        axis.axhline(0.0, linewidth=1.5, linestyle="--")
        axis.set_xlabel(r"$\gamma$ (population geometry)")
        axis.set_title(title)
        axis.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Mean normalized MSE gap (LD - Sortition)")
    fig.suptitle(r"Clique experiment in one dimension, conditioned on $M=1$")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_and_save(
    output_dir: str | Path,
    *,
    n_list: list[int],
    gamma_list: list[float],
    n_mc: int,
    theorem_n_list: list[int],
    theorem_gamma_list: list[float],
    theorem_graphs: int,
    theorem_mc: int,
    q_graphs: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "experiment1_clique_geometry",
        "n_list": n_list,
        "gamma_list": gamma_list,
        "n_mc": n_mc,
        "theorem_n_list": theorem_n_list,
        "theorem_gamma_list": theorem_gamma_list,
        "theorem_graphs": theorem_graphs,
        "theorem_mc": theorem_mc,
        "q_graphs": q_graphs,
        "base_seed": base_seed,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    identity = verify_variance_identity(
        theorem_n_list,
        theorem_gamma_list,
        n_graphs=theorem_graphs,
        n_mc=theorem_mc,
        base_seed=base_seed + 100,
    )
    identity.to_csv(output / "variance_identity.csv", index=False)
    plot_variance_identity(identity, output / "variance_identity_M1.png")

    mean_df, mean_diagnostics = run_geometry_experiment(
        n_list,
        gamma_list,
        n_mc=n_mc,
        base_seed=base_seed,
        aggregation="mean",
    )
    median_df, median_diagnostics = run_geometry_experiment(
        n_list,
        gamma_list,
        n_mc=n_mc,
        base_seed=base_seed + 1,
        aggregation="median",
    )
    mean_df.to_csv(output / "exp1_mean_M1.csv", index=False)
    median_df.to_csv(output / "exp1_median_M1.csv", index=False)
    pd.concat([mean_diagnostics, median_diagnostics], ignore_index=True).to_csv(
        output / "exp1_diagnostics.csv", index=False
    )

    plot_sampling_vs_amplification(
        output / "sampling_vs_amplification_M1.png",
        n=max(20, min(80, max(n_list))),
        gamma_edge=0.3,
        gamma_center=2.0,
        q_graphs=q_graphs,
        base_seed=base_seed + 2,
    )
    plot_mse_gap_lines(mean_df, median_df, output / "mse_gap_lines_exp1_M1.png")
    return mean_df, median_df, identity
