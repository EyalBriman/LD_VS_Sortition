import numpy as np

from ld_vs_sortition.core import aggregate, grid_points_cube_gamma
from ld_vs_sortition.delegation import (
    clique_probability_cdf,
    compute_indegrees_batch,
    sample_targets_batch_from_cdf,
)


def test_gamma_grid_has_expected_shape_and_range():
    points = grid_points_cube_gamma(n=17, d=2, gamma=0.5)
    assert points.shape == (17, 2)
    assert np.all((0.0 <= points) & (points <= 1.0))


def test_aggregation_rules_return_vectors():
    points = np.array([[0.0], [0.5], [1.0]])
    weights = np.array([1.0, 2.0, 1.0])
    assert np.allclose(aggregate(points, weights, "mean"), [0.5])
    assert np.allclose(aggregate(points, weights, "median"), [0.5])


def test_clique_sampler_has_no_self_delegation_and_indegrees_sum_to_n():
    points = np.linspace(0.0, 1.0, 8)
    cdf = clique_probability_cdf(points)
    rng = np.random.default_rng(4)
    targets = sample_targets_batch_from_cdf(cdf, 100, rng)
    assert targets.shape == (100, 8)
    assert np.all(targets != np.arange(8)[None, :])
    indegrees = compute_indegrees_batch(targets)
    assert np.allclose(indegrees.sum(axis=1), 8.0)
