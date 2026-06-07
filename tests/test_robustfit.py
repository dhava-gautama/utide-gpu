"""Regression tests for robustfit edge cases."""

import numpy as np

from utide.robustfit import robustfit


def test_robustfit_rank_deficient_does_not_crash():
    # A collinear column makes the design rank-deficient, so np.linalg.lstsq
    # returns an empty residual array. robustfit must handle that instead of
    # raising IndexError on rsumsq[0].
    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal((n, 5))
    X[:, 4] = X[:, 0]  # exact collinearity
    y = X @ np.arange(5.0) + 0.1 * rng.standard_normal(n)
    rf = robustfit(X, y)
    assert np.isfinite(rf.b).all()
    assert rf.iterations >= 1
    assert np.isfinite(rf.rms_resid)


def test_robustfit_full_rank_unchanged():
    # Full-rank path must be unaffected by the rank-deficient guard.
    rng = np.random.default_rng(1)
    n = 500
    X = rng.standard_normal((n, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ beta + 0.05 * rng.standard_normal(n)
    rf = robustfit(X, y)
    assert np.allclose(rf.b, beta, atol=0.05)
