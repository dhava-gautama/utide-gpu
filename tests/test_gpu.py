"""
GPU-backend tests. These are skipped automatically when CuPy and a working
CUDA device are not available, so the suite still passes on CPU-only machines.
"""

import numpy as np
import pytest

from utide import solve, solve_many

# Skip the whole module unless CuPy + a device are present.
cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)


CONSTIT = ["M2", "S2", "N2", "K1", "O1", "P1", "Q1", "M4", "M6", "MM", "MF"]
# Float datenum input must be paired with an epoch, otherwise _normalize_time
# interprets the values as datetime64[ms] and collapses the time span.
EPOCH = "2000-01-01"


def _series(nt=2 * 365 * 24, twodim=False, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) / 24.0
    fr = {"M2": 1 / 12.42, "S2": 1 / 12.0, "K1": 1 / 23.93, "O1": 1 / 25.82}
    u = (
        np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.3)
        + 0.4 * np.cos(2 * np.pi * fr["S2"] * 24 * t + 1.0)
        + 0.5 * np.cos(2 * np.pi * fr["K1"] * 24 * t + 2.0)
        + 0.1 * rng.standard_normal(nt)
    )
    if twodim:
        v = 0.6 * np.cos(2 * np.pi * fr["M2"] * 24 * t + 0.7) + 0.1 * rng.standard_normal(nt)
        return t, u, v
    return t, u, None


def test_gpu_matches_cpu_1d():
    t, u, _ = _series()
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="linear", epoch=EPOCH, verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, **kw)
    assert np.allclose(c0["A"], c1["A"], rtol=1e-5, atol=1e-8)
    assert np.allclose(c0["g"], c1["g"], rtol=1e-5, atol=1e-6)
    assert np.allclose(c0["A_ci"], c1["A_ci"], rtol=1e-4, atol=1e-8, equal_nan=True)


def test_gpu_matches_cpu_2d():
    t, u, v = _series(twodim=True)
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="linear", epoch=EPOCH, verbose=False)
    c0 = solve(t, u, v, **kw)
    c1 = solve(t, u, v, gpu=True, **kw)
    assert np.allclose(c0["Lsmaj"], c1["Lsmaj"], rtol=1e-5, atol=1e-8)
    assert np.allclose(c0["g"], c1["g"], rtol=1e-5, atol=1e-6)


def test_gpu_single_precision_runs_and_is_close():
    # Mixed-precision (FP32) path should run and stay within a few digits on
    # the constituents that carry real signal (near-zero/noise constituents
    # are meaningless in FP32 and may also reorder).
    t, u, _ = _series()
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="none", epoch=EPOCH, verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, gpu_precision="single", **kw)
    order = [list(c1["name"]).index(n) for n in c0["name"]]
    A0, A1 = c0["A"], c1["A"][order]
    g0, g1 = c0["g"], c1["g"][order]
    big = A0 > 0.05
    assert np.allclose(A0[big], A1[big], rtol=5e-3, atol=1e-4)
    dg = (g0[big] - g1[big] + 180) % 360 - 180
    assert np.allclose(dg, 0, atol=0.1)


def test_gpu_robust_matches_cpu():
    # Robust IRLS runs entirely on the device and must match the CPU result.
    rng = np.random.default_rng(3)
    t, u, _ = _series()
    u = u.copy()
    idx = rng.choice(len(u), 30, replace=False)
    u[idx] += rng.uniform(-5, 5, 30)  # outliers for the robust weights to act on
    kw = dict(lat=45, constit=CONSTIT, method="robust", conf_int="none",
              epoch=EPOCH, verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, **kw)
    order = [list(c1["name"]).index(n) for n in c0["name"]]
    big = c0["A"] > 0.05
    assert c0["rf"].iterations == c1["rf"].iterations
    assert np.allclose(c0["A"][big], c1["A"][order][big], rtol=1e-4, atol=1e-6)


def test_gpu_fallback_unsupported_option():
    # nodal=False is not the GPU basis path; must fall back and still be correct.
    t, u, _ = _series()
    kw = dict(lat=45, constit=CONSTIT, method="ols", conf_int="none",
              nodal=False, verbose=False)
    c0 = solve(t, u, **kw)
    c1 = solve(t, u, gpu=True, **kw)
    assert np.allclose(c0["A"], c1["A"], rtol=1e-10, atol=1e-12)


def test_solve_many_gappy():
    # Per-series gaps: each series solved on its own valid samples, matching a
    # per-series solve; all-NaN series come back NaN.
    rng = np.random.default_rng(5)
    t, u, _ = _series()
    S = 4
    X = np.column_stack([(0.5 + 0.3 * i) * u for i in range(S)])
    X = X + 0.05 * rng.standard_normal(X.shape)
    for s in range(S):
        X[rng.choice(len(t), len(t) // 5, replace=False), s] = np.nan
    om = solve_many(t, X, lat=45, constit=CONSTIT, gpu=True, epoch=EPOCH, verbose=False)
    for s in range(S):
        c = solve(t, X[:, s], lat=45, constit=CONSTIT, method="ols",
                  conf_int="none", epoch=EPOCH, verbose=False)
        order = [list(om.name).index(n) for n in c["name"]]
        big = c["A"] > 0.05
        assert np.allclose(c["A"][big], om.A[order, s][big], rtol=1e-4, atol=1e-6)
    Xn = X.copy()
    Xn[:, 0] = np.nan
    om2 = solve_many(t, Xn, lat=45, constit=CONSTIT, gpu=True, epoch=EPOCH, verbose=False)
    assert np.all(np.isnan(om2.A[:, 0]))


def test_solve_many_chunked_matches_unchunked():
    # Forcing a small chunk size must give the same result as one big solve.
    rng = np.random.default_rng(7)
    t, u, _ = _series()
    X = np.column_stack([(0.5 + 0.2 * i) * u for i in range(20)])
    X = X + 0.05 * rng.standard_normal(X.shape)
    kw = dict(lat=45, constit=CONSTIT, gpu=True, epoch=EPOCH, verbose=False)
    a = solve_many(t, X, **kw)
    b = solve_many(t, X, chunk_size=3, **kw)  # 20 series -> 7 chunks
    assert np.allclose(a.A, b.A, rtol=1e-6, atol=1e-8, equal_nan=True)
    assert np.allclose(a.g, b.g, rtol=1e-6, atol=1e-8, equal_nan=True)


def test_solve_many_matches_solve():
    t, u, _ = _series()
    X = np.column_stack([u, 0.7 * u, 1.3 * u])
    om = solve_many(t, X, lat=45, constit=CONSTIT, gpu=True, epoch=EPOCH, verbose=False)
    c = solve(t, u, lat=45, constit=CONSTIT, method="ols", conf_int="none", epoch=EPOCH, verbose=False)
    order = [list(om.name).index(n) for n in c["name"]]
    assert np.allclose(c["A"], om.A[order, 0], rtol=1e-5, atol=1e-8)
    # phase wrapped difference
    dg = (c["g"] - om.g[order, 0] + 180) % 360 - 180
    assert np.allclose(dg, 0, atol=1e-4)
