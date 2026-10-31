"""
Tests for feature_engineer.normal_stockout_prob — the one definition of the
Normal-approximation P(stockout ≤ horizon) that serves as the
continuous_risk_score label column, the evaluation baseline, and the
stockout_risk_score the batch scorer publishes (it is no longer a model head).
"""
import numpy as np
from scipy.stats import norm

from ml_training.datasets.feature_engineer import normal_stockout_prob
from ml_training.datasets.label_generator import _normal_stockout_prob


def test_edge_cases_match_the_original_scalar_rules():
    assert normal_stockout_prob([100.0], [0.0], [5.0], 7)[0] == 0.0     # no demand -> no risk
    assert normal_stockout_prob([0.0], [10.0], [2.0], 7)[0] == 1.0      # already out of stock
    assert normal_stockout_prob([50.0], [10.0], [0.0], 7)[0] == 1.0     # deterministic, 5 days < 7
    assert normal_stockout_prob([100.0], [10.0], [0.0], 7)[0] == 0.0    # deterministic, 10 days >= 7


def test_matches_independent_scipy_formula_and_scalar_wrapper():
    rng = np.random.default_rng(0)
    q   = rng.exponential(300.0, 500)
    mu  = rng.exponential(30.0, 500) + 0.5
    sd  = mu * rng.uniform(0.05, 0.6, 500)

    mu_t     = q / mu
    sig_t    = q * sd / mu**2
    expected = norm.cdf(7, loc=mu_t, scale=sig_t)

    got = normal_stockout_prob(q, mu, sd, 7)
    np.testing.assert_allclose(got, expected, atol=1e-12)

    scalar = np.array([_normal_stockout_prob(a, b, c, 7) for a, b, c in zip(q, mu, sd, strict=True)])
    np.testing.assert_allclose(scalar, expected, atol=1e-12)


def test_bounded_and_monotone_in_stock():
    rng = np.random.default_rng(1)
    out = normal_stockout_prob(rng.exponential(300, 1000), rng.exponential(30, 1000), rng.exponential(5, 1000), 7)
    assert np.all((out >= 0.0) & (out <= 1.0))
    low_stock, high_stock = normal_stockout_prob([50.0, 150.0], [10.0, 10.0], [3.0, 3.0], 7)
    assert low_stock > high_stock
