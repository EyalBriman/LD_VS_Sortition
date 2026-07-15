from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core import aggregate, direct_democracy_outcome, grid_points_cube_gamma, instance_spread_s2
from ..delegation import (
    basin_weights_to_representatives,
    block_probability_matrix,
    compute_indegrees,
    inverse_distance_weight_matrix,
    make_blocks,
    sample_sbm_adjacency,
    sample_targets_from_allowed_weights,
)
from ..selection import sample_representatives_corrected


SBM_CONFIGS: list[dict[str, float | int | str]] = [
    {
        "label": "Strong community (B=2, p_in=0.30, p_out=0.05)",
        "blocks": 2,
        "p_in": 0.30,
        "p_out": 0.05,
    },
    {
        "label": "Weak community (B=2, p_in=0.15, p_out=0.10)",
        "blocks": 2,
        "p_in": 0.15,
        "p_out": 0.10,
    },
]


def compute_ld_outcome(
    points: np.ndarray,
    targets: np.ndarray,
    representatives: np.ndarray,
    aggregation: str,
) -> tuple[np.ndarray, int]:
    weights, unresolved = basin_weights_to_representatives(targets, representatives)
    outcome = aggregate(points[representatives], weights, aggregation)
    return outcome, unresolved


def one_mse_gap_draw(
    points: np.ndarray,
    center: np.ndarray,
    spread_s2: float,
    k: int,
    graph_probability_matrix: np.ndarray,
    distance_weights: np.ndarray,
    rng: np.random.Generator,
    aggregation: str,
    *,
    max_attempts: int = 10_000,
) -> tuple[float, int, int, int]:
    """Generate one LD-sortition MSE-gap draw.

    Returns the normalized gap, realized LD committee size, unresolved cycle
    count, and number of conditioning attempts.
    """
    n = points.shape[0]
    k_effective = min(int(k), n)

    for attempt in range(1, max_attempts + 1):
        adjacency = sample_sbm_adjacency(graph_probability_matrix, rng)
        targets = sample_targets_from_allowed_weights(distance_weights, adjacency, rng)
        indegrees = compute_indegrees(targets, n)
        representatives = sample_representatives_corrected(indegrees, rng, k_effective)
        if representatives is not None:
            break
    else:
        raise RuntimeError("too many rejected LD draws under the conditioning rule")

    ld_outcome, unresolved = compute_ld_outcome(
        points, targets, representatives, aggregation
    )
    sortition_representatives = rng.choice(n, size=k_effective, replace=False)
    sortition_outcome = aggregate(
        points[sortition_representatives],
        np.ones(k_effective, dtype=float),
        aggregation,
    )
    ld_error = float(np.sum((ld_outcome - center) ** 2))
    sortition_error = float(np.sum((sortition_outcome - center) ** 2))
    return (
        (ld_error - sortition_error) / spread_s2,
        int(len(representatives)),
        int(unresolved),
        attempt,
    )


def run_sbm_experiment(
    n_list: Iterable[int],
    gamma_list: Iterable[float],
    k_list: Iterable[int],
    *,
    sbm_configs: list[dict[str, float | int | str]],
    n_mc: int,
    base_seed: int,
    aggregation: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(base_seed)
    rows: list[dict[str, float | int | str]] = []

    # Materialize iterables once so generators are also supported.
    n_values = list(n_list)
    gamma_values = list(gamma_list)
    k_values = list(k_list)
    total = len(n_values) * len(gamma_values) * len(sbm_configs) * len(k_values)
    completed = 0

    for n in n_values:
        for gamma in gamma_values:
            points = grid_points_cube_gamma(int(n), 2, float(gamma))
            center = direct_democracy_outcome(points, aggregation)
            spread = instance_spread_s2(points, center)
            distance_weights = inverse_distance_weight_matrix(points, metric="l1")

            for sbm in sbm_configs:
                blocks = make_blocks(int(n), int(sbm["blocks"]))
                probability_matrix = block_probability_matrix(
                    blocks, float(sbm["p_in"]), float(sbm["p_out"])
                )
                for k in k_values:
                    for draw_index in range(n_mc):
                        gap, realized_size, unresolved, attempts = one_mse_gap_draw(
                            points,
                            center,
                            spread,
                            int(k),
                            probability_matrix,
                            distance_weights,
                            rng,
                            aggregation,
                        )
                        rows.append(
                            {
                                "shape": f"grid_gamma(g={gamma},d=2)",
                                "n": int(n),
                                "d": 2,
                                "gamma": float(gamma),
                                "sbm": str(sbm["label"]),
                                "k_target": int(k),
                                "draw_index": int(draw_index),
                                "normalized_mse_gap": float(gap),
                                "aggregation": aggregation,
                                "realized_ld_size": realized_size,
                                "unresolved_agents": unresolved,
                                "conditioning_attempts": attempts,
                            }
                        )
                    completed += 1
                    print(
                        f"  [{completed}/{total}] Experiment 2 | aggregation={aggregation:6s} | "
                        f"n={n} | gamma={gamma:g} | k={k} | {sbm['label']}",
                        flush=True,
                    )

    return pd.DataFrame(rows)


def plot_mean_gap_lines(
    mean_df: pd.DataFrame,
    median_df: pd.DataFrame,
    *,
    n_fixed: int,
    sbm_label: str,
    path: Path,
) -> None:
    k_values = sorted(mean_df["k_target"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for axis, frame, title in zip(
        axes, [mean_df, median_df], ["Weighted mean", "Weighted median"]
    ):
        subset = frame[(frame["n"] == n_fixed) & (frame["sbm"] == sbm_label)]
        for k in k_values:
            by_k = subset[subset["k_target"] == k]
            x_values: list[float] = []
            means: list[float] = []
            low: list[float] = []
            high: list[float] = []
            for gamma in sorted(by_k["gamma"].unique()):
                values = by_k.loc[
                    by_k["gamma"] == gamma, "normalized_mse_gap"
                ].to_numpy()
                if values.size == 0:
                    continue
                mean = float(values.mean())
                standard_error = (
                    float(values.std(ddof=1) / np.sqrt(values.size))
                    if values.size > 1 else 0.0
                )
                x_values.append(float(gamma))
                means.append(mean)
                low.append(mean - 1.96 * standard_error)
                high.append(mean + 1.96 * standard_error)
            axis.plot(x_values, means, "o-", label=f"k={k}")
            axis.fill_between(x_values, low, high, alpha=0.13)
        axis.axhline(0.0, linewidth=1.2, linestyle="--")
        axis.set_xlabel(r"$\gamma$ (population geometry)")
        axis.set_ylabel("Mean normalized MSE gap (LD - Sortition)")
        axis.set_title(title)
        axis.legend(fontsize=9, loc="lower left", title="Target size")

    fig.suptitle(f"Experiment 2: n={n_fixed}; {sbm_label}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compact_summary(
    mean_df: pd.DataFrame,
    median_df: pd.DataFrame,
    *,
    n_fixed: int,
    sbm_configs: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    k_values = sorted(mean_df["k_target"].unique())
    n_rows = len(sbm_configs)
    n_columns = len(k_values)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.2 * n_columns, 3.8 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for row, sbm in enumerate(sbm_configs):
        for column, k in enumerate(k_values):
            axis = axes[row, column]
            for frame, linestyle, label in [
                (mean_df, "-", "Weighted mean"),
                (median_df, "--", "Weighted median"),
            ]:
                subset = frame[
                    (frame["n"] == n_fixed)
                    & (frame["sbm"] == sbm["label"])
                    & (frame["k_target"] == k)
                ]
                x_values: list[float] = []
                means: list[float] = []
                lower: list[float] = []
                upper: list[float] = []
                for gamma in sorted(subset["gamma"].unique()):
                    values = subset.loc[
                        subset["gamma"] == gamma, "normalized_mse_gap"
                    ].to_numpy()
                    if values.size == 0:
                        continue
                    x_values.append(float(gamma))
                    means.append(float(values.mean()))
                    lower.append(float(np.percentile(values, 25)))
                    upper.append(float(np.percentile(values, 75)))
                axis.plot(x_values, means, linestyle, marker="o", label=label)
                if linestyle == "-":
                    axis.fill_between(x_values, lower, upper, alpha=0.12)
            axis.axhline(0.0, linewidth=1.2, linestyle="--")
            axis.set_xlabel(r"$\gamma$")
            axis.set_title(f"k={k}")
            if column == 0:
                axis.set_ylabel(f"{sbm['label']}\nMean normalized gap")
            if row == 0 and column == n_columns - 1:
                axis.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        f"Experiment 2 compact summary (n={n_fixed}, d=2)\n"
        "Negative: LD lower MSE; positive: sortition lower MSE",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["aggregation", "n", "gamma", "sbm", "k_target"])
        .agg(
            mean_gap=("normalized_mse_gap", "mean"),
            se_gap=(
                "normalized_mse_gap",
                lambda values: values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
            ),
            mean_realized_ld_size=("realized_ld_size", "mean"),
            mean_unresolved_agents=("unresolved_agents", "mean"),
            fraction_with_unresolved=("unresolved_agents", lambda values: np.mean(values > 0)),
            mean_conditioning_attempts=("conditioning_attempts", "mean"),
            n_draws=("normalized_mse_gap", "count"),
        )
        .reset_index()
    )


def run_and_save(
    output_dir: str | Path,
    *,
    n_list: list[int],
    gamma_list: list[float],
    k_list: list[int],
    n_mc: int,
    base_seed: int,
    sbm_configs: list[dict[str, float | int | str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configurations = SBM_CONFIGS if sbm_configs is None else sbm_configs
    config = {
        "experiment": "experiment2_sbm_variable_k",
        "n_list": n_list,
        "gamma_list": gamma_list,
        "k_list": k_list,
        "n_mc": n_mc,
        "base_seed": base_seed,
        "sbm_configs": configurations,
        "cycle_policy": "agents in representative-free cycles are unresolved and carry no basin weight",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    mean_df = run_sbm_experiment(
        n_list,
        gamma_list,
        k_list,
        sbm_configs=configurations,
        n_mc=n_mc,
        base_seed=base_seed,
        aggregation="mean",
    )
    median_df = run_sbm_experiment(
        n_list,
        gamma_list,
        k_list,
        sbm_configs=configurations,
        n_mc=n_mc,
        base_seed=base_seed + 1,
        aggregation="median",
    )
    mean_df.to_csv(output / "exp2_mean.csv", index=False)
    median_df.to_csv(output / "exp2_median.csv", index=False)
    combined = pd.concat([mean_df, median_df], ignore_index=True)
    summarize_diagnostics(combined).to_csv(output / "exp2_summary.csv", index=False)

    for sbm_index, sbm in enumerate(configurations):
        for n in n_list:
            plot_mean_gap_lines(
                mean_df,
                median_df,
                n_fixed=n,
                sbm_label=str(sbm["label"]),
                path=output / f"exp2_lines_sbm{sbm_index + 1}_n{n}.png",
            )
    plot_compact_summary(
        mean_df,
        median_df,
        n_fixed=n_list[min(len(n_list) // 2, len(n_list) - 1)],
        sbm_configs=configurations,
        path=output / "exp2_compact_summary.png",
    )
    return mean_df, median_df
