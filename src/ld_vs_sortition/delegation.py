from __future__ import annotations

import numpy as np


def inverse_distance_weight_matrix(
    points: np.ndarray,
    *,
    metric: str = "l1",
    eps: float = 1e-12,
) -> np.ndarray:
    """Return off-diagonal inverse-distance delegation weights."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]

    differences = points[:, None, :] - points[None, :, :]
    if metric == "l1":
        distances = np.abs(differences).sum(axis=2)
    elif metric == "l2":
        distances = np.sqrt((differences * differences).sum(axis=2))
    else:
        raise ValueError("metric must be 'l1' or 'l2'")

    np.fill_diagonal(distances, np.inf)
    weights = 1.0 / (distances + eps)
    np.fill_diagonal(weights, 0.0)
    return weights


def probability_cdf_from_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every agent must have at least one positive delegation weight")
    probabilities = weights / row_sums
    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.0
    return cdf


def clique_probability_cdf(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return probability_cdf_from_weights(
        inverse_distance_weight_matrix(points, metric="l1", eps=eps)
    )


def sample_targets_batch_from_cdf(
    cdf: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent clique delegation graphs from row-wise categorical laws."""
    n = cdf.shape[0]
    uniforms = rng.random((batch_size, n))
    targets = np.empty((batch_size, n), dtype=np.int32)
    for agent in range(n):
        targets[:, agent] = np.searchsorted(
            cdf[agent], uniforms[:, agent], side="right"
        )
    return targets


def compute_indegrees_batch(targets: np.ndarray, n: int | None = None) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int32)
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch_size, n)")
    if n is None:
        n = targets.shape[1]
    batch_size = targets.shape[0]
    indegrees = np.zeros((batch_size, n), dtype=float)
    rows = np.repeat(np.arange(batch_size), targets.shape[1])
    np.add.at(indegrees, (rows, targets.ravel()), 1.0)
    return indegrees


def compute_indegrees(targets: np.ndarray, n: int | None = None) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int32)
    if n is None:
        n = len(targets)
    return np.bincount(targets, minlength=n).astype(float)


def make_blocks(n: int, num_blocks: int) -> np.ndarray:
    if num_blocks < 1 or num_blocks > n:
        raise ValueError("num_blocks must lie between 1 and n")
    base, remainder = divmod(n, num_blocks)
    blocks = np.empty(n, dtype=np.int32)
    start = 0
    for block in range(num_blocks):
        size = base + (1 if block < remainder else 0)
        blocks[start:start + size] = block
        start += size
    return blocks


def block_probability_matrix(
    blocks: np.ndarray,
    p_in: float,
    p_out: float,
) -> np.ndarray:
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError("SBM probabilities must lie in [0,1]")
    same_block = blocks[:, None] == blocks[None, :]
    probabilities = np.where(same_block, p_in, p_out).astype(float)
    np.fill_diagonal(probabilities, 0.0)
    return probabilities


def sample_sbm_adjacency(
    probability_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = probability_matrix.shape[0]
    uniforms = rng.random((n, n))
    upper = np.triu(uniforms < probability_matrix, k=1)
    return upper | upper.T


def sample_targets_from_allowed_weights(
    distance_weights: np.ndarray,
    adjacency: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one categorical target per row using an exponential race.

    Isolated vertices fall back to the full inverse-distance clique. For positive
    weights w_ij, argmin E_ij / w_ij with E_ij~Exp(1) samples target j with
    probability w_ij / sum_l w_il.
    """
    allowed = np.asarray(distance_weights, dtype=float) * adjacency
    isolated = allowed.sum(axis=1) <= 0
    if np.any(isolated):
        allowed = allowed.copy()
        allowed[isolated] = distance_weights[isolated]

    exponentials = rng.exponential(scale=1.0, size=allowed.shape)
    race_times = np.divide(
        exponentials,
        allowed,
        out=np.full_like(allowed, np.inf, dtype=float),
        where=allowed > 0,
    )
    return np.argmin(race_times, axis=1).astype(np.int32)


def basin_weights_to_representatives(
    targets: np.ndarray,
    representatives: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Follow delegation chains and assign resolved agents to representatives.

    Agents whose path enters a directed cycle containing no representative are
    counted as unresolved and contribute no basin weight. If no agent resolves
    at all, representatives receive equal fallback weights. This preserves the
    convention of the supplied simulation while making it explicit and
    measurable.
    """
    targets = np.asarray(targets, dtype=np.int32)
    representatives = np.asarray(representatives, dtype=np.int32)
    n = len(targets)
    representative_position = np.full(n, -1, dtype=np.int32)
    representative_position[representatives] = np.arange(
        len(representatives), dtype=np.int32
    )
    weights = np.zeros(len(representatives), dtype=float)
    unresolved = 0

    for start in range(n):
        current = start
        seen: set[int] = set()
        while True:
            rep_index = representative_position[current]
            if rep_index >= 0:
                weights[rep_index] += 1.0
                break
            if current in seen:
                unresolved += 1
                break
            seen.add(current)
            current = int(targets[current])

    if weights.sum() <= 0:
        weights[:] = 1.0
    return weights, unresolved
