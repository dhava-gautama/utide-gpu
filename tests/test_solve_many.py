"""CPU tests for solve_many (no GPU required)."""

import numpy as np

from utide import reconstruct, reconstruct_many, solve, solve_many

EPOCH = "2000-01-01"
CONSTIT = ["M2", "S2", "N2", "K1", "O1", "P1", "Q1", "M4"]


def _series(nt=2 * 365 * 24, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / 24.0
    fr = {"M2": 1 / 12.42, "S2": 1 / 12.0, "K1": 1 / 23.93, "O1": 1 / 25.82}
    u = (
        np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.3)
        + 0.4 * np.cos(2 * np.pi * fr["S2"] * 24 * t + 1.0)
        + 0.5 * np.cos(2 * np.pi * fr["K1"] * 24 * t + 2.0)
        + 0.3 * np.cos(2 * np.pi * fr["O1"] * 24 * t)
        + 0.1 * rng.standard_normal(nt)
    )
    return t, u


def test_solve_many_cpu_matches_solve():
    t, u = _series()
    X = np.column_stack([u, 0.7 * u, 1.3 * u])
    om = solve_many(
        t,
        X,
        lat=45,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    for s in range(3):
        c = solve(
            t,
            X[:, s],
            lat=45,
            constit=CONSTIT,
            method="ols",
            conf_int="none",
            epoch=EPOCH,
            verbose=False,
        )
        order = [list(om.name).index(n) for n in c["name"]]
        assert np.allclose(c["A"], om.A[order, s], rtol=1e-5, atol=1e-8)


def test_solve_many_solver_equivalence():
    # normal-equations (auto) must match lstsq on a well-conditioned design.
    rng = np.random.default_rng(2)
    t, u = _series()
    X = np.column_stack([(0.5 + 0.2 * i) * u for i in range(8)])
    X = X + 0.05 * rng.standard_normal(X.shape)
    kw = {"lat": 45, "constit": CONSTIT, "gpu": False, "epoch": EPOCH, "verbose": False}
    a = solve_many(t, X, solver="normal", **kw)
    b = solve_many(t, X, solver="lstsq", **kw)
    assert np.allclose(a.A, b.A, rtol=1e-6, atol=1e-8)
    assert np.allclose(a.g, b.g, rtol=1e-6, atol=1e-8)


def test_solve_many_cpu_gappy():
    rng = np.random.default_rng(5)
    t, u = _series()
    X = np.column_stack([(0.5 + 0.3 * i) * u for i in range(4)])
    X = X + 0.05 * rng.standard_normal(X.shape)
    for s in range(4):
        X[rng.choice(len(t), len(t) // 5, replace=False), s] = np.nan
    om = solve_many(
        t,
        X,
        lat=45,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    for s in range(4):
        c = solve(
            t,
            X[:, s],
            lat=45,
            constit=CONSTIT,
            method="ols",
            conf_int="none",
            epoch=EPOCH,
            verbose=False,
        )
        order = [list(om.name).index(n) for n in c["name"]]
        assert np.allclose(c["A"], om.A[order, s], rtol=1e-5, atol=1e-8)
    Xn = X.copy()
    Xn[:, 0] = np.nan
    om2 = solve_many(
        t,
        Xn,
        lat=45,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    assert np.all(np.isnan(om2.A[:, 0]))


def test_reconstruct_many_matches_reconstruct():
    t, u = _series()
    X = np.column_stack([u, 0.7 * u, 1.3 * u])
    om = solve_many(
        t,
        X,
        lat=45,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    H = reconstruct_many(t, om, epoch=EPOCH, gpu=False)
    assert H.shape == (len(t), 3)
    # series 0 must match a per-series solve + reconstruct
    c = solve(
        t,
        u,
        lat=45,
        constit=CONSTIT,
        method="ols",
        conf_int="none",
        epoch=EPOCH,
        verbose=False,
    )
    tide = reconstruct(t, c, epoch=EPOCH, min_SNR=0, verbose=False)
    assert np.allclose(H[:, 0], tide.h, rtol=1e-6, atol=1e-8)


def test_solve_many_per_station_lat():
    # An array of latitudes groups series into bands; each must match a
    # per-series solve at its own latitude.
    t, u = _series()
    lats = np.array([20.0, 40.0, 60.0])
    X = np.column_stack([u, 0.9 * u, 1.1 * u])
    om = solve_many(
        t,
        X,
        lat=lats,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    for s in range(3):
        c = solve(
            t,
            X[:, s],
            lat=float(lats[s]),
            constit=CONSTIT,
            method="ols",
            conf_int="none",
            epoch=EPOCH,
            verbose=False,
        )
        order = [list(om.name).index(n) for n in c["name"]]
        assert np.allclose(c["A"], om.A[order, s], rtol=1e-8, atol=1e-10)


def test_solve_many_2d():
    t, u = _series()
    v = 0.5 * np.roll(u, 13)
    U = np.column_stack([u, 0.8 * u])
    V = np.column_stack([v, 0.8 * v])
    om = solve_many(
        t,
        U,
        V,
        lat=45,
        constit=CONSTIT,
        gpu=False,
        epoch=EPOCH,
        verbose=False,
    )
    c = solve(
        t,
        u,
        v,
        lat=45,
        constit=CONSTIT,
        method="ols",
        conf_int="none",
        epoch=EPOCH,
        verbose=False,
    )
    order = [list(om.name).index(n) for n in c["name"]]
    assert np.allclose(c["Lsmaj"], om.Lsmaj[order, 0], rtol=1e-5, atol=1e-8)
