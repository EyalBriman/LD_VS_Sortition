from __future__ import annotations

import numpy as np


def symmetric_gamma_warp(u: np.ndarray, gamma: float) -> np.ndarray:
    """Warp [0,1]^d symmetrically around 1/2.

    gamma < 1 pushes mass toward the boundary; gamma > 1 pushes mass toward
    the center. gamma=1 leaves the grid unchanged.
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive")
    values = np.asarray(u, dtype=float)
    centered = 2.0 * values - 1.0
    return 0.5 + 0.5 * np.sign(centered) * np.abs(centered) ** gamma


def grid_points_cube_gamma(n: int, d: int, gamma: float) -> np.ndarray:
    """Create n deterministic points in [0,1]^d and apply a gamma warp."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if d < 1:
        raise ValueError("d must be at least 1")

    side = int(np.ceil(n ** (1.0 / d)))
    axes = [np.linspace(0.0, 1.0, side) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=1)[:n].copy()
    points = symmetric_gamma_warp(points, gamma)
    points = np.nan_to_num(points, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(points, 0.0, 1.0)


def weighted_mean_nd(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones(len(positions), dtype=float)
        total = float(weights.sum())
    normalized = weights / total
    return (positions * normalized[:, None]).sum(axis=0)


def weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones(len(values), dtype=float)
        total = float(weights.sum())

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    position = int(np.searchsorted(cumulative, total / 2.0, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def weighted_median_nd(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    return np.array(
        [weighted_median_1d(positions[:, coordinate], weights)
         for coordinate in range(positions.shape[1])],
        dtype=float,
    )


def aggregate(
    positions: np.ndarray,
    weights: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    if aggregation == "mean":
        return weighted_mean_nd(positions, weights)
    if aggregation == "median":
        return weighted_median_nd(positions, weights)
    raise ValueError("aggregation must be 'mean' or 'median'")


def direct_democracy_outcome(points: np.ndarray, aggregation: str) -> np.ndarray:
    return aggregate(points, np.ones(points.shape[0], dtype=float), aggregation)


def instance_spread_s2(points: np.ndarray, center: np.ndarray) -> float:
    """Mean squared Euclidean distance from the reference outcome."""
    differences = points - np.asarray(center, dtype=float).reshape(1, -1)
    return max(float((differences * differences).sum(axis=1).mean()), 1e-12)
