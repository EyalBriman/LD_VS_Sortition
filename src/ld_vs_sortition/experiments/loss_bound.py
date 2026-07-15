from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..delegation import clique_probability_cdf
from ..selection import estimate_singleton_distribution_analytic


def sample_random_instance(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=n)


def estimate_bias_for_instance(
    points: np.ndarray,
    n_graphs: int,
    rng: np.random.Generator,
    *,
    batch_size: int = 500,
) -> tuple[float, float, float, float]:
    """Estimate B^(1), |B^(1)|, P(M=1), and P(M!=1)."""
    points = np.asarray(points, dtype=float)
    mean = float(points.mean())
    cdf = clique_probability_cdf(points)
    q, p_m1 = estimate_singleton_distribution_analytic(
        cdf, n_graphs, rng, batch_size=batch_size
    )
    bias = float(np.dot(q, points - mean))
    return bias, abs(bias), p_m1, 1.0 - p_m1


def run_loss_bound_experiment(
    n_list: Iterable[int],
    *,
    n_instances: int = 500,
    graphs_per_instance: int = 500,
    base_seed: int = 2025,
    batch_size: int = 500,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    master_rng = np.random.default_rng(base_seed)

    for n in n_list:
        print(f"  n={n:4d} | graph=Clique", flush=True)
        for instance_index in range(n_instances):
            instance_seed = int(master_rng.integers(0, 2**31 - 1))
            instance_rng = np.random.default_rng(instance_seed)
            points = sample_random_instance(int(n), instance_rng)
            bias, loss, p_m1, not_m1 = estimate_bias_for_instance(
                points,
                graphs_per_instance,
                instance_rng,
                batch_size=batch_size,
            )
            rows.append(
                {
                    "graph": "Clique",
                    "n": int(n),
                    "instance_index": int(instance_index),
                    "instance_seed": instance_seed,
                    "mu": float(points.mean()),
                    "B_hat": bias,
                    "L_hat": loss,
                    "mean_p_m1": p_m1,
                    "mean_not_m1_rate": not_m1,
                    "exceeds_or_equals_half": int(loss >= 0.5),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["graph", "n", "instance_index"])
        .reset_index(drop=True)
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["graph", "n"])
        .agg(
            max_L_hat=("L_hat", "max"),
            mean_L_hat=("L_hat", "mean"),
            p90_L_hat=("L_hat", lambda values: np.quantile(values, 0.90)),
            frac_ge_half=("exceeds_or_equals_half", "mean"),
            mean_p_m1=("mean_p_m1", "mean"),
            mean_not_m1_rate=("mean_not_m1_rate", "mean"),
            n_instances=("L_hat", "count"),
        )
        .reset_index()
    )


def plot_loss_distribution(df: pd.DataFrame, path: Path) -> None:
    n_values = sorted(df["n"].unique())
    fig, axes = plt.subplots(
        1, len(n_values), figsize=(3.2 * len(n_values), 3.2), sharey=False
    )
    if len(n_values) == 1:
        axes = [axes]
    for axis, n in zip(axes, n_values):
        values = df.loc[df["n"] == n, "L_hat"]
        axis.hist(values, bins=30, alpha=0.7, edgecolor="none")
        axis.axvline(0.5, linewidth=1.5, linestyle="--", label=r"$L=1/2$")
        axis.set_xlabel(r"$\hat{L}^{(1)}$")
        axis.set_ylabel("Count")
        axis.set_title(f"Clique\nn={n}")
        axis.set_xlim(left=0.0)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        r"Empirical singleton loss $\hat{L}^{(1)}=|\hat{B}^{(1)}|$"
        "\npositions sampled from U[0,1]",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 4.0))
    for n in sorted(df["n"].unique()):
        subset = df[df["n"] == n]
        axis.scatter(subset["mu"], subset["L_hat"], s=10, alpha=0.4, label=f"n={n}")
    axis.axhline(0.5, linewidth=1.5, linestyle="--", label=r"$L=1/2$")
    axis.set_xlabel(r"$\mu$ (instance mean)")
    axis.set_ylabel(r"$\hat{L}^{(1)}$")
    axis.set_title("Clique")
    axis.legend(fontsize=8, markerscale=2)
    fig.suptitle(r"$\hat{L}^{(1)}$ versus instance mean")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_max_loss_vs_n(summary_df: pd.DataFrame, path: Path) -> None:
    subset = summary_df.sort_values("n")
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(subset["n"], subset["max_L_hat"], marker="o", label="maximum")
    axis.plot(
        subset["n"], subset["p90_L_hat"], marker="o", linestyle="--", label="90th percentile"
    )
    axis.axhline(0.5, linewidth=1.5, linestyle="-.", label=r"bound $1/2$")
    axis.set_xlabel("n")
    axis.set_ylabel(r"$\hat{L}^{(1)}$")
    axis.set_title("Empirical singleton loss versus n")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_and_save(
    output_dir: str | Path,
    *,
    n_list: list[int],
    n_instances: int,
    graphs_per_instance: int,
    base_seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "singleton_loss_bound",
        "n_list": n_list,
        "n_instances": n_instances,
        "graphs_per_instance": graphs_per_instance,
        "base_seed": base_seed,
        "batch_size": batch_size,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = run_loss_bound_experiment(
        n_list,
        n_instances=n_instances,
        graphs_per_instance=graphs_per_instance,
        base_seed=base_seed,
        batch_size=batch_size,
    )
    results.to_csv(output / "loss_bound_results.csv", index=False)
    summary_df = summarize(results)
    summary_df.to_csv(output / "loss_bound_summary.csv", index=False)

    violations = results[results["L_hat"] >= 0.5]
    violations.to_csv(output / "loss_bound_violations.csv", index=False)
    plot_loss_distribution(results, output / "hist_loss_by_n.png")
    plot_loss_scatter(results, output / "scatter_loss_vs_mu.png")
    plot_max_loss_vs_n(summary_df, output / "max_loss_vs_n.png")

    print("\n=== Loss-bound summary ===")
    print(summary_df.to_string(index=False))
    if violations.empty:
        print("\nNo sampled instance had L_hat^(1) >= 1/2.")
    else:
        print(f"\nObserved {len(violations)} sampled instances with L_hat^(1) >= 1/2.")
    return results, summary_df
