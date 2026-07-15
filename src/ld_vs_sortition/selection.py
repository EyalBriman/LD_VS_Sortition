from __future__ import annotations

import numpy as np

from .delegation import compute_indegrees_batch, sample_targets_batch_from_cdf


def singleton_raw_probabilities(
    indegrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(Com={i}|DelG) and P(M=1|DelG) row by row."""
    indegrees = np.asarray(indegrees, dtype=float)
    n = indegrees.shape[1]
    probabilities = indegrees / n
    q_raw = np.zeros_like(probabilities, dtype=float)

    certain_rows = np.isclose(probabilities.max(axis=1), 1.0)
    if np.any(certain_rows):
        selected = np.argmax(probabilities[certain_rows], axis=1)
        q_raw[certain_rows, selected] = 1.0

    ordinary_rows = ~certain_rows
    if np.any(ordinary_rows):
        ordinary = probabilities[ordinary_rows]
        all_fail = np.prod(1.0 - ordinary, axis=1)
        q_raw[ordinary_rows] = (
            ordinary * all_fail[:, None] / (1.0 - ordinary)
        )

    p_m1 = q_raw.sum(axis=1)
    return q_raw, p_m1


def estimate_singleton_distribution_analytic(
    cdf: np.ndarray,
    n_graphs: int,
    rng: np.random.Generator,
    *,
    batch_size: int = 500,
) -> tuple[np.ndarray, float]:
    """Estimate q_i^(1)=P(Com={i}|M=1) by analytic within-graph conditioning."""
    if n_graphs < 1:
        raise ValueError("n_graphs must be positive")
    n = cdf.shape[0]
    numerator = np.zeros(n, dtype=float)
    denominator = 0.0
    completed = 0

    while completed < n_graphs:
        current_batch = min(batch_size, n_graphs - completed)
        targets = sample_targets_batch_from_cdf(cdf, current_batch, rng)
        indegrees = compute_indegrees_batch(targets, n)
        q_raw, p_m1 = singleton_raw_probabilities(indegrees)
        numerator += q_raw.sum(axis=0)
        denominator += float(p_m1.sum())
        completed += current_batch

    if denominator <= 0:
        raise RuntimeError("estimated P(M=1) is zero")
    return numerator / denominator, denominator / n_graphs


def sample_singleton_indices_conditioned(
    cdf: np.ndarray,
    n_valid: int,
    rng: np.random.Generator,
    *,
    batch_size: int = 4096,
    max_batches: int = 100_000,
) -> tuple[np.ndarray, float]:
    """Rejection-sample unique representatives from the full process | M=1."""
    if n_valid < 1:
        raise ValueError("n_valid must be positive")
    n = cdf.shape[0]
    accepted_chunks: list[np.ndarray] = []
    accepted_count = 0
    accepted_total = 0
    total_draws = 0

    for _ in range(max_batches):
        targets = sample_targets_batch_from_cdf(cdf, batch_size, rng)
        indegrees = compute_indegrees_batch(targets, n)
        self_selection = rng.random((batch_size, n)) < (indegrees / n)
        counts = self_selection.sum(axis=1)
        valid = counts == 1
        valid_count = int(valid.sum())
        accepted_total += valid_count
        total_draws += batch_size

        if valid_count:
            chunk = np.argmax(self_selection[valid], axis=1).astype(np.int32)
            accepted_chunks.append(chunk)
            accepted_count += len(chunk)
            if accepted_count >= n_valid:
                indices = np.concatenate(accepted_chunks)[:n_valid]
                return indices, accepted_total / total_draws

    raise RuntimeError("too many batches while conditioning on M=1")


def conditional_expected_size(probabilities: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    expected_size = float(probabilities.sum())
    nonempty_probability = float(1.0 - np.prod(1.0 - probabilities))
    if nonempty_probability <= 0:
        return 0.0
    return expected_size / nonempty_probability


def calibrate_selection_probabilities_conditional(
    indegrees: np.ndarray,
    k: int,
    *,
    centrality_power: float = 1.0,
) -> np.ndarray:
    """Choose Bernoulli probabilities with E[M|M>0]=k for k>1."""
    n = len(indegrees)
    k_effective = min(int(k), n)
    if k_effective <= 1:
        raise ValueError("k=1 uses conditioning on M=1")
    if k_effective >= n:
        return np.ones(n, dtype=float)
    if centrality_power <= 0:
        raise ValueError("centrality_power must be positive")

    centrality = np.maximum(np.asarray(indegrees, dtype=float), 1e-12)
    base = centrality ** centrality_power
    base /= base.sum()

    low, high = 0.0, 1.0
    while conditional_expected_size(np.minimum(high * base, 1.0)) < k_effective:
        high *= 2.0
        if high > 1e12:
            raise RuntimeError("failed to bracket conditional-size calibration")

    for _ in range(70):
        midpoint = (low + high) / 2.0
        candidate = np.minimum(midpoint * base, 1.0)
        if conditional_expected_size(candidate) < k_effective:
            low = midpoint
        else:
            high = midpoint
    return np.minimum(high * base, 1.0)


def sample_representatives_corrected(
    indegrees: np.ndarray,
    rng: np.random.Generator,
    k: int,
    *,
    centrality_power: float = 1.0,
) -> np.ndarray | None:
    """Sample representatives under the singleton/conditional-size convention."""
    n = len(indegrees)
    k_effective = min(int(k), n)
    if k_effective < 1:
        raise ValueError("k must be positive")

    if k_effective == 1:
        total = float(indegrees.sum())
        probabilities = (
            np.ones(n, dtype=float) / n
            if total <= 0
            else np.asarray(indegrees, dtype=float) / total
        )
        representatives = np.flatnonzero(
            rng.random(n) < probabilities
        ).astype(np.int32)
        return representatives if representatives.size == 1 else None

    probabilities = calibrate_selection_probabilities_conditional(
        indegrees,
        k_effective,
        centrality_power=centrality_power,
    )
    representatives = np.flatnonzero(
        rng.random(n) < probabilities
    ).astype(np.int32)
    return representatives if representatives.size > 0 else None
