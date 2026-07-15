import numpy as np

from ld_vs_sortition.delegation import clique_probability_cdf
from ld_vs_sortition.selection import (
    calibrate_selection_probabilities_conditional,
    conditional_expected_size,
    estimate_singleton_distribution_analytic,
    singleton_raw_probabilities,
)


def test_singleton_raw_probabilities_known_row():
    indegrees = np.array([[2.0, 1.0, 0.0]])
    q_raw, p_m1 = singleton_raw_probabilities(indegrees)
    p = indegrees[0] / 3.0
    expected = np.array([
        p[0] * (1 - p[1]) * (1 - p[2]),
        p[1] * (1 - p[0]) * (1 - p[2]),
        0.0,
    ])
    assert np.allclose(q_raw[0], expected)
    assert np.allclose(p_m1[0], expected.sum())


def test_analytic_singleton_distribution_is_a_probability_vector():
    cdf = clique_probability_cdf(np.linspace(0.0, 1.0, 7))
    q, p_m1 = estimate_singleton_distribution_analytic(
        cdf, 500, np.random.default_rng(12), batch_size=100
    )
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q >= 0.0)
    assert 0.0 < p_m1 <= 1.0


def test_conditional_size_calibration_hits_target():
    indegrees = np.array([5.0, 3.0, 2.0, 0.0, 0.0])
    probabilities = calibrate_selection_probabilities_conditional(indegrees, 3)
    assert np.isclose(conditional_expected_size(probabilities), 3.0, atol=1e-8)
