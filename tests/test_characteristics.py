"""Tests for empirical tidal datums (utide.tidal_characteristics)."""

import numpy as np

from utide import (
    solve,
    tidal_characteristics,
    tidal_characteristics_many,
    tidal_form_factor,
)


def _cosine(nt=45 * 24, amp=1.5, mean=0.5, period_h=12.42):
    t = np.arange(nt) / 24.0
    h = mean + amp * np.cos(2 * np.pi * (1 / period_h) * 24 * t)
    return t, h


def test_pure_cosine_datums():
    amp, mean, period_h = 1.5, 0.5, 12.42
    t, h = _cosine(amp=amp, mean=mean, period_h=period_h)
    c = tidal_characteristics(t, h)
    assert abs(c.MHW - (mean + amp)) < 0.1
    assert abs(c.MLW - (mean - amp)) < 0.1
    assert abs(c.MTL - mean) < 0.05
    assert abs(c.MTR - 2 * amp) < 0.15
    # ebb and flood durations are each ~half a period for a symmetric tide
    assert abs(c.ED - period_h / 2) < 0.7
    assert abs(c.FD - period_h / 2) < 0.7
    assert c.n_high >= 80 and c.n_low >= 80


def test_batched_matches_single():
    amps = np.array([1.0, 1.5, 2.0])
    nt = 45 * 24
    t = np.arange(nt) / 24.0
    X = amps[None, :] * np.cos(2 * np.pi * (1 / 12.42) * 24 * t[:, None])
    out = tidal_characteristics_many(t, X)
    assert out.MTR.shape == (3,)
    assert np.allclose(out.MTR, 2 * amps, atol=0.15)
    for s in range(3):
        c = tidal_characteristics(t, X[:, s])
        assert out.MHW[s] == c.MHW  # identical to the single-series call


def test_flat_series_returns_none_and_nan():
    t = np.arange(200) / 24.0
    assert tidal_characteristics(t, np.ones(200)) is None
    out = tidal_characteristics_many(t, np.ones((200, 2)))
    assert np.all(np.isnan(out.MTR))
    assert np.all(out.n_high == 0)


def test_form_factor_classification():
    nt = 120 * 24
    t = np.arange(nt) / 24.0
    cph = {"M2": 1 / 12.42, "S2": 1 / 12.0, "K1": 1 / 23.93, "O1": 1 / 25.82}
    kw = {
        "lat": 45,
        "constit": ["M2", "S2", "K1", "O1"],
        "method": "ols",
        "conf_int": "none",
        "epoch": "2000-01-01",
        "verbose": False,
    }
    # diurnal-dominant -> F large
    h = (
        1.0 * np.cos(2 * np.pi * cph["K1"] * 24 * t)
        + 0.8 * np.cos(2 * np.pi * cph["O1"] * 24 * t)
        + 0.1 * np.cos(2 * np.pi * cph["M2"] * 24 * t)
    )
    F, regime = tidal_form_factor(solve(t, h, **kw), classify=True)
    assert F > 3 and regime == "diurnal"
    # semidiurnal-dominant -> F small
    h = (
        1.0 * np.cos(2 * np.pi * cph["M2"] * 24 * t)
        + 0.3 * np.cos(2 * np.pi * cph["S2"] * 24 * t)
        + 0.05 * np.cos(2 * np.pi * cph["K1"] * 24 * t)
    )
    assert tidal_form_factor(solve(t, h, **kw)) < 0.25


def test_nan_values_ignored():
    t, h = _cosine(amp=1.5, mean=0.0)
    h = h.copy()
    h[100:160] = np.nan
    c = tidal_characteristics(t, h)
    assert c is not None
    assert abs(c.MTR - 3.0) < 0.15
